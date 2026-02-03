import mujoco
import time
import threading
from pathlib import Path
import re
import cv2
import numpy as np
from multiprocessing import shared_memory

from holosoma.simulator.mujoco.mujoco import MuJoCo
from holosoma.utils.rate import RateLimiter

# mainly an interface with camera.
class MujocoCameraRenderer:
    def __init__(self, simulator: MuJoCo, height: int, width: int) -> None:

        self.simulator: MuJoCo = simulator
        # TODO: get height/width of renderer from config
        self._height = height
        self._width = width
        self.camera_names = [
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
        return self._renderer
        
    def get_frames(self):
        renderer = self._get_renderer()
        world_id = getattr(self.simulator, "current_world_id", 0)
        render_data = self.simulator.backend.get_render_data(world_id=world_id)

        frames = []
        for camera_name in self.camera_names:
            renderer.update_scene(render_data, camera=camera_name)
            # depth, already in meters
            frame = renderer.render()
            frames.append(frame)
        return frames

class ImageServer:
    def __init__(self, simulator): 

        # TODO: A bunch of configs, refactor this to use a config class 
        self.image_type = "depth"
        self.near_clip = 0.1
        self.far_clip = 2.0
        self.expected_shape = (27, 48)

        renderer_height = 135  
        renderer_width = 240 

        # Initialize camera renderer
        self.camera = MujocoCameraRenderer(simulator, renderer_height, renderer_width)

        # Initialize shared memory
        self._init_shared_memory()
    
    def _init_shared_memory(self):
        img_shm_name = "depth_img_shm"

        # Initialize shared memory
        channels = 1 if self.image_type == "depth" else 3
        num_cameras = 2
        expected_shape = [num_cameras, channels, self.expected_shape[0], self.expected_shape[1]]
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
        # if self.image_type == "depth":
        #     # Normalize depth image from [-0.5, 0.5] to [0, 1] to [0, 255]
        #     display = ((frame + 0.5) * 255.0).astype(np.uint8)
        # else:
        #     display = frame
        display = ((frame + 0.5) * 255.0).astype(np.uint8)
        cv2.imshow(f"Image Server Stream {name}", display)
        return cv2.waitKey(1) & 0xFF == ord("q")
        
    
    def send_process(self):
        # this thread should be running at 10hz

        render_frequency = 10
        rate_limiter = RateLimiter(render_frequency)

        step_count = 0
        while True:

            frames = self.camera.get_frames()
            frames = [self._post_process_frame(frame) for frame in frames]

            # concatenated_frames = np.concatenate(processed_frames, axis=1)
            # self._display_frame("resized and clipped", concatenated_frames)

             # Concatenate frames before channel dimension (axis=0)
            # [C, H, W] -> [N, C, H, W] N is the number of cameras
            full_image = np.stack(frames, axis=0)
            # copy to shared memory
            try:
                target_dtype = np.float32 if self.image_type == "depth" else np.uint8
                processed = full_image.astype(target_dtype)
                np.copyto(self.img_array, processed)
            except Exception as e:
                print(f"[Image Server] Failed to copy to shared memory: {e}")
                print(f"[Image Server] Input shape: {full_image.shape}, Target shape: {self.img_array.shape}")
                print(f"[Image Server] Input dtype: {full_image.dtype}, Target dtype: {target_dtype}")
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
