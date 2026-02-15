import mujoco
import time
import threading
import random
from pathlib import Path
import re
import cv2
import numpy as np
from multiprocessing import shared_memory

from holosoma.simulator.mujoco.mujoco import MuJoCo
from holosoma.utils.rate import RateLimiter

# mainly an interface with camera.
class MujocoCameraRenderer:
    def __init__(self, simulator: MuJoCo, height: int, width: int, camera_names: list[str] | None = None, znear: float = 0.001) -> None:

        self.simulator: MuJoCo = simulator
        self._height = height
        self._width = width
        self._znear = znear
        self.camera_names = camera_names or [
            "robot_cam_front_depth",
            "robot_cam_back_depth"
        ]
        self._renderer: mujoco.Renderer | None = None
        self._renderer_thread_id: int | None = None


    def _get_renderer(self) -> mujoco.Renderer:
        current_thread_id = threading.get_ident()
        if self._renderer is None or self._renderer_thread_id != current_thread_id:
            self._renderer = mujoco.Renderer(
                self.simulator.root_model, height=self._height, width=self._width
            )
            self._renderer.enable_depth_rendering()
            self._renderer_thread_id = current_thread_id
            # Exclude head mesh (group 2) from depth rendering
            self._depth_scene_option = mujoco.MjvOption()
            self._depth_scene_option.geomgroup[2] = 0
        return self._renderer

    def get_frames(self):
        renderer = self._get_renderer()
        # Set znear before every render to prevent other components from overriding it
        extent = self.simulator.root_model.stat.extent
        self.simulator.root_model.vis.map.znear = self._znear / extent
        world_id = getattr(self.simulator, "current_world_id", 0)
        render_data = self.simulator.backend.get_render_data(world_id=world_id)

        frames = []
        for camera_name in self.camera_names:
            renderer.update_scene(render_data, camera=camera_name, scene_option=self._depth_scene_option)
            # depth, already in meters
            frame = renderer.render()
            frames.append(frame)
        return frames

class ImageServer:
    def __init__(
        self,
        simulator,
        camera_names: list[str] | None = None,
        image_type: str = "depth",
        renderer_height: int = 135,
        renderer_width: int = 240,
        resized_height: int = 27,
        resized_width: int = 48,
        near_clip: float = 0.1,
        far_clip: float = 2.0,
        frame_rate: int = 10,
        crop_y_start: int|None = None,
        crop_x_start: int|None = None,
        crop_x_end: int|None = None,
        crop_y_end: int|None = None,
        image_show: bool = False,
        render_near_plane: float = 0.001,
        latency_frame: int | tuple[int, int] = 0,
        buffer_len: int = 1,
    ):
        self.image_type = image_type
        self.near_clip = near_clip
        self.far_clip = far_clip
        self.crop_y_start = crop_y_start
        self.crop_x_start = crop_x_start
        self.crop_x_end = crop_x_end
        self.crop_y_end = crop_y_end
        self.image_show = image_show
        self.expected_shape = (resized_height, resized_width)
        self.frame_rate = frame_rate
        self.render_near_plane = render_near_plane

        # Latency buffer config
        if isinstance(latency_frame, (tuple, list)) and len(latency_frame) == 2:
            self.latency_frame_range = latency_frame
            self.latency_frame = None
        else:
            self.latency_frame_range = None
            self.latency_frame = latency_frame
        self.buffer_len = buffer_len

        if self.latency_frame_range is not None:
            assert self.latency_frame_range[1] < self.buffer_len, \
                f"Max latency frame ({self.latency_frame_range[1]}) must be less than buffer length ({self.buffer_len})"
        elif self.latency_frame is not None:
            assert self.latency_frame < self.buffer_len, \
                f"Latency frame ({self.latency_frame}) must be less than buffer length ({self.buffer_len})"

        # Initialize camera renderer
        self.camera = MujocoCameraRenderer(
            simulator,
            renderer_height,
            renderer_width,
            camera_names,
            znear=self.render_near_plane,
        )
        self.num_cameras = len(self.camera.camera_names)

        # Initialize shared memory
        self._init_shared_memory()

        # Initialize latency buffer: [buffer_len, num_cameras, channels, H, W]
        channels = 1 if self.image_type == "depth" else 3
        dtype = np.float32 if self.image_type == "depth" else np.uint8
        self._frame_buffer = np.zeros(
            (self.buffer_len, self.num_cameras, channels, self.expected_shape[0], self.expected_shape[1]),
            dtype=dtype,
        )

    def _init_shared_memory(self):
        img_shm_name = "depth_img_shm"

        # Initialize shared memory
        channels = 1 if self.image_type == "depth" else 3
        expected_shape = [self.num_cameras, channels, self.expected_shape[0], self.expected_shape[1]]
        dtype = np.float32 if self.image_type == "depth" else np.uint8

        try:
            memory_size = np.prod(expected_shape) * np.dtype(dtype).itemsize
            # Try to create new shared memory, if it exists, connect to existing one
            try:
                self.image_shm = shared_memory.SharedMemory(create=True, size=memory_size, name=img_shm_name)
                print(f"[Image Server] Created new shared memory: {img_shm_name}")
            except FileExistsError:
                self.image_shm = shared_memory.SharedMemory(name=img_shm_name)
                print(f"[Image Server] Connected to existing shared memory: {img_shm_name}")

            self.img_array = np.ndarray(expected_shape, dtype=dtype, buffer=self.image_shm.buf)
            print(f"[Image Server] Shared memory: shape={expected_shape}, dtype={dtype}")
        except Exception as e:
            print(f"[Image Server] Failed to initialize shared memory: {e}")
            raise

        print("ImageServer initialized")

    def _post_process_frame(self, frame):
        # crop
        if self.crop_y_start is not None or self.crop_y_end is not None or self.crop_x_start is not None or self.crop_x_end is not None: 
            frame = frame[..., self.crop_y_start:self.crop_y_end, self.crop_x_start:self.crop_x_end]
        # resize
        frame = cv2.resize(frame, (self.expected_shape[1], self.expected_shape[0]), cv2.INTER_CUBIC)

        # clip and scale to [-0.5, 0.5] range
        if self.image_type == "depth":
            frame = np.clip(frame, self.near_clip, self.far_clip)
            frame = (frame - self.near_clip) / (self.far_clip - self.near_clip) - 0.5
            frame = np.expand_dims(frame, axis=2)

        # [H, W, C] -> [C, H, W]
        frame = frame.transpose(2, 0, 1)
        return frame


    def _display_frame(self, name, frame):
        display = ((frame + 0.5) * 255.0).astype(np.uint8)
        cv2.imshow(f"Image Server Stream {name}", display)
        return cv2.waitKey(1) & 0xFF == ord("q")


    def send_process(self):
        rate_limiter = RateLimiter(self.frame_rate)

        step_count = 0
        while True:

            frames = self.camera.get_frames()
            frames = [self._post_process_frame(frame) for frame in frames]
            if self.image_show:
                self._display_frame("depth_processed", frames[0][0])

             # Concatenate frames before channel dimension (axis=0)
            # [C, H, W] -> [N, C, H, W] N is the number of cameras
            full_image = np.stack(frames, axis=0)

            # Shift buffer left and insert the newest frame at the end
            target_dtype = np.float32 if self.image_type == "depth" else np.uint8
            processed = full_image.astype(target_dtype)
            self._frame_buffer[:-1] = self._frame_buffer[1:]
            self._frame_buffer[-1] = processed

            # Select the delayed frame from the buffer
            if self.latency_frame_range is not None:
                current_latency = random.randint(self.latency_frame_range[0], self.latency_frame_range[1])
            else:
                current_latency = self.latency_frame
            delayed_frame = self._frame_buffer[-1 - current_latency]

            # copy to shared memory
            try:
                np.copyto(self.img_array, delayed_frame)
            except Exception as e:
                print(f"[Image Server] Failed to copy to shared memory: {e}")
                print(f"[Image Server] Input shape: {delayed_frame.shape}, Target shape: {self.img_array.shape}")
                print(f"[Image Server] Input dtype: {delayed_frame.dtype}, Target dtype: {self.img_array.dtype}")
                continue

            rate_limiter.sleep()
            step_count += 1

######################## temporary functions ########################

    def _depth_to_pixels(self, depth_frame): # Shift nearest values to the origin.
        depth = depth_frame.copy()
        depth -= depth.min()
        # Scale by 2 mean distances of near rays.
        depth /= 2*depth[depth <= 1].mean()
        # Scale to [0, 255]
        pixels = 255*np.clip(depth, 0, 1)
        return pixels


def write_depth_video_from_frames(
    frames_dir: str | Path,
    output_path: str | Path,
    fps: int = 10,
) -> None:
    """Create a depth video from concatenated_frames_*.png files."""
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    pattern = re.compile(r"concatenated_frames_(\d+)\.png$")

    frame_paths: list[Path] = []
    for path in frames_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match:
            frame_paths.append(path)

    if not frame_paths:
        raise FileNotFoundError(f"No concatenated_frames_*.png found in {frames_dir}")

    frame_paths.sort(key=lambda p: int(pattern.match(p.name).group(1)))

    first_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_UNCHANGED)
    if first_frame is None:
        raise RuntimeError(f"Failed to read frame: {frame_paths[0]}")

    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        isColor=(first_frame.ndim == 3),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at {output_path}")

    for path in frame_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise RuntimeError(f"Failed to read frame: {path}")
        writer.write(frame)

    writer.release()
