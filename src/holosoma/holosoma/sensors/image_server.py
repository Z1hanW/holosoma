from doctest import FAIL_FAST
import time
import threading
from queue import Queue, Full
from dataclasses import dataclass, field
from pathlib import Path
import re
import cv2
import numpy as np
from multiprocessing import shared_memory
from typing import Literal
from holosoma.simulator.mujoco.mujoco import MujocoRendererWrapper
from holosoma.utils.rate import RateLimiter
from datetime import datetime

@dataclass(frozen=True)
class ZEDCameraConfig:
    """Configuration for ZED Camera initialization.
    
    """
    
    img_shape: tuple[int, int] = (720, 1280)
    """Image shape as (height, width). Common resolutions: (720, 1280), (1080, 1920), (1242, 2208), (376, 672)"""
    
    fps: int = 10
    """Frames per second"""
    
    serial_number: int = 32658215
    """Camera serial number."""
    
    depth_mode: Literal["NONE", "NEURAL", "NEURAL_LIGHT", "PERFORMANCE", "QUALITY", "ULTRA"] = "NEURAL"
    """Depth mode. Options: NONE, NEURAL, NEURAL_LIGHT, PERFORMANCE, QUALITY, ULTRA"""
    
    confidence_threshold: int = 100
    """Confidence threshold for depth measurement (0-100)"""

    positive_inf_depth_value: float = 2.0
    """How +inf should be mapped to in the depth image, in meters"""

    negative_inf_depth_value: float = 0.1
    """How -inf should be mapped to in the depth image, in meters"""

    nan_depth_value: float = 0.0
    """How nan should be mapped to in the depth image, in meters"""

default_terms = {
    "front": ZEDCameraConfig(
        serial_number=32658215,
    ),
    "back": ZEDCameraConfig(
        serial_number=33082869,
    ),
}
@dataclass(frozen=True)
class ZedCamerasConfig:
    """Configuration for ZED Cameras."""

    terms: dict[str, ZEDCameraConfig] = field(default_factory=lambda: default_terms.copy())

class ZEDCamera:
    def __init__(self, config: ZEDCameraConfig):
        # Lazy import ZED SDK, in case using simulator-camera
        try:
            import pyzed.sl as sl
        except ImportError:
            raise ImportError("pyzed.sl is not available. Please install the ZED SDK Python API to use ZED cameras.")
        
        self.sl = sl
        self.config = config
        self._init_zed()

    def _init_zed(self):
        """Initialize ZED camera"""
        # 
        sl = self.sl
        # Initialize ZED camera objects
        self.zed = sl.Camera()

        # Initialize image buffers
        self.rgb_mat_side_by_side = sl.Mat()
        self.depth_mat = sl.Mat()

        # Setup runtime parameters
        self.runtime_params = sl.RuntimeParameters()
        self.runtime_params.confidence_threshold = self.config.confidence_threshold

        # Setup init parameters
        self.init_params = sl.InitParameters()
        # self.init_params.camera_fps = self.config.fps
        # TODO: Understand why camera.get_frame() depends on this.

        self.init_params.camera_fps = 200
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.set_from_serial_number(self.config.serial_number)
        self.init_params.camera_resolution = self._get_zed_resolution_enum(self.config.img_shape)

        self.init_params.sdk_verbose = True

        self.init_params.depth_mode = self._get_depth_mode_enum()
        
        print(f"[ZED Camera] Init parameters: {self.init_params.depth_mode}")
        
        # Open camera
        status = self.zed.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {repr(status)}")
        
        # Get camera info

        cam_info = self.zed.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters
        left_fov = calib.left_cam.h_fov, calib.left_cam.v_fov
        
        print(f"[ZED Camera] Initialized successfully")
        print(f"[ZED Camera] Resolution: {cam_info.camera_configuration.resolution.width}x{cam_info.camera_configuration.resolution.height}")
        print(f"[ZED Camera] FOV: {left_fov[0]:.2f}° x {left_fov[1]:.2f}°")
        print(f"[ZED Camera] Depth mode: {self.config.depth_mode}")

    def _get_depth_mode_enum(self):
        """Convert depth_mode string from config to ZED SDK enum."""
        sl = self.sl
        depth_mode_map = {
            "NONE": sl.DEPTH_MODE.NONE,
            "NEURAL": sl.DEPTH_MODE.NEURAL,
            "NEURAL_LIGHT": sl.DEPTH_MODE.NEURAL_LIGHT,
            "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
            "QUALITY": sl.DEPTH_MODE.QUALITY,
            "ULTRA": sl.DEPTH_MODE.ULTRA,
        }
        return depth_mode_map.get(self.config.depth_mode, self.sl.DEPTH_MODE.NEURAL)
    
    def _get_zed_resolution_enum(self, img_shape):
        """Convert image shape to ZED resolution enum"""
        sl = self.sl
        height, width = img_shape[0], img_shape[1]
        
        # Common ZED resolutions
        if width == 1280 and height == 720:
            return sl.RESOLUTION.HD720
        elif width == 1920 and height == 1080:
            return sl.RESOLUTION.HD1080
        elif width == 2208 and height == 1242:
            return sl.RESOLUTION.HD2K
        elif width == 672 and height == 376:
            return sl.RESOLUTION.VGA
        else:
            # Default to HD720 if no exact match
            print(f"[ZED Camera] Warning: Resolution {width}x{height} not exactly matched, using HD720")
            return sl.RESOLUTION.HD720
    
    def _get_depth_data(self):
        """Get depth data from ZED camera, in meters"""
        sl = self.sl
        # Retrieve depth image
        self.zed.retrieve_measure(self.depth_mat, sl.MEASURE.DEPTH)

        # unit of depth_mat is in init_parameters.coordinate_units
        depth_data = self.depth_mat.get_data()

        # TODO: +inf should be 1. -inf should be 0. nan should be ? not sure...
        # TODO: nan should use interpolation to fill in the gaps (nearest neighbor interpolation?)
        depth_data = np.nan_to_num(depth_data, nan=self.config.nan_depth_value, posinf=self.config.positive_inf_depth_value, neginf=self.config.negative_inf_depth_value)
        return depth_data
    
    def _get_rgb_data(self):
        """Get RGB data from ZED camera"""
        sl = self.sl
        self.zed.retrieve_image(self.rgb_mat_side_by_side, sl.VIEW.SIDE_BY_SIDE_BGR)
        image_data = self.rgb_mat_side_by_side.get_data()
        return image_data
    
    def capture(self):
        """Capture rgb and depth data from ZED camera"""
        sl = self.sl
        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            depth_data = self._get_depth_data() if self.config.depth_mode != "NONE" else None
            rgb_data = self._get_rgb_data()
            return {"depth": depth_data, "rgb": rgb_data}
        else:
            print(f"[ZED Camera] Grab error: failed to grab frame")
            return None, None
    
        
    def release(self):
        """Release ZED camera resources"""
        if self.zed.is_opened():
            self.zed.close()
            print("[ZED Camera] Released")

class ZedCamerasWrapper:
    def __init__(self, config: ZedCamerasConfig):
       
        self.cameras = {name: ZEDCamera(config.terms[name]) for name in config.terms.keys()}
        self.num_cameras = len(self.cameras)

    def get_frames(self):
        depth_data: dict[str, np.ndarray] = {}
        rgb_data: dict[str, np.ndarray] = {}
        for name, camera in self.cameras.items():
            depth_data[name] = camera.capture()["depth"]
            rgb_data[name] = camera.capture()["rgb"]
        return {"depth": depth_data, "rgb": rgb_data}


@dataclass(frozen=True)
class ImageServerConfig:
    """Configuration for Image Server."""

    near_clip: float = 0.1
    far_clip: float = 2.0

    resized_height: int = 27
    resized_width: int = 48

    enable_save_images: bool = True 
    enable_visualize_images: bool = False 
    image_root_dir: str = "image_server_images"
    save_queue_maxsize: int = 0
    """Maximum queue size for image saving. 0 = unlimited. When full, oldest items are dropped."""
    save_workers: int = 2
    """Number of worker threads for parallel image saving."""

    gum_enabled: bool = False


class SaveImagesProfiler:
    """Profiler for tracking save_images operation statistics."""
    
    def __init__(self):
        self.times = []
        self.count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
    
    def record(self, elapsed_ms: float):
        """Record a timing measurement."""
        self.times.append(elapsed_ms)
        self.count += 1
        self.total_time += elapsed_ms
        self.min_time = min(self.min_time, elapsed_ms)
        self.max_time = max(self.max_time, elapsed_ms)
    
    def get_stats(self) -> dict[str, float]:
        """Get statistics about recorded timings."""
        if self.count == 0:
            return {}
        avg_time = self.total_time / self.count
        return {
            "count": self.count,
            "avg_ms": avg_time,
            "min_ms": self.min_time,
            "max_ms": self.max_time,
            "total_ms": self.total_time,
        }
    
    def reset(self):
        """Reset all statistics."""
        self.times.clear()
        self.count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0


class ImageServer:
    def __init__(self, camera_wrapper: list[MujocoRendererWrapper | ZedCamerasWrapper], config: ImageServerConfig, gum: None):
        self.config = config
        self.camera_wrapper: ZedCamerasWrapper | MujocoRendererWrapper = camera_wrapper
        self.gum = gum
        self._init_shared_memory()

        if self.config.enable_save_images:
            self._init_save_images_dir()
            self.save_images_profiler = SaveImagesProfiler()
            self.save_queue = Queue(maxsize=self.config.save_queue_maxsize if self.config.save_queue_maxsize > 0 else 0)
            self.save_queue_dropped_count = 0
            self.camera_dirs_created = False
            self._dirs_lock = threading.Lock()
            # Start multiple worker threads for parallel saving
            self.save_threads = []
            for i in range(self.config.save_workers):
                thread = threading.Thread(target=self._save_images_worker, daemon=True, name=f"SaveWorker-{i}")
                thread.start()
                self.save_threads.append(thread)
        
        self.frame_count = 0
    
    def _init_save_images_dir(self):
        # create a subdirectory in the save_images_dir for each session, and name the session with timestamp like 20260204_123456
        session_dir = Path(self.config.image_root_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir.mkdir(exist_ok=True, parents=True)
        self.save_images_dir = session_dir
        print(f"[Image Server] Initialized save images directory: {self.save_images_dir}")
    
    def _init_shared_memory(self):
        img_shm_name = "depth_img_shm"

        # Initialize shared memory
        expected_shape = [self.camera_wrapper.num_cameras, 1, self.config.resized_height, self.config.resized_width]
        dtype = np.float32
        
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

    def _resize_clip_expand_transpose(self, frame):
        # resize
        frame = cv2.resize(frame, (self.config.resized_width, self.config.resized_height), cv2.INTER_CUBIC)

        # clip and scale to [-0.5, 0.5] range
        frame = np.clip(frame, self.config.near_clip, self.config.far_clip)
        frame = (frame - self.config.near_clip) / (self.config.far_clip - self.config.near_clip) - 0.5

        # [H, W] -> [1, H, W]
        frame = np.expand_dims(frame, axis=0)
        return frame
    

    def _display_frame(self, name: str, frame: np.ndarray) -> bool:
        cv2.imshow(f"Image Server Stream {name}", frame)
        return cv2.waitKey(1) & 0xFF == ord("q")
        
    def _ensure_camera_dirs(self, camera_names):
        """Pre-create directories for all cameras (thread-safe, idempotent)."""
        if self.camera_dirs_created:
            return
        with self._dirs_lock:
            if not self.camera_dirs_created:
                for cam_name in camera_names:
                    (self.save_images_dir / cam_name / "depth").mkdir(exist_ok=True, parents=True)
                    (self.save_images_dir / cam_name / "rgb").mkdir(exist_ok=True, parents=True)
                self.camera_dirs_created = True
    
    def _save_images_worker(self):
        """Background thread worker that saves images from the queue."""
        while True:
            try:
                item = self.save_queue.get()
                if item is None:  # Sentinel to stop
                    break
                
                raw_images, step_count, timestamp = item
                
                # Pre-create directories if needed (only once)
                camera_names = list(raw_images["depth"].keys())
                self._ensure_camera_dirs(camera_names)
                
                # Save images in parallel (each camera can be saved independently)
                for cam_name in camera_names:
                    depth_path = self.save_images_dir / cam_name / "depth" / f"{cam_name}_{step_count}_{timestamp}.png"
                    rgb_path = self.save_images_dir / cam_name / "rgb" / f"{cam_name}_{step_count}_{timestamp}.png"
                    
                    # Save both depth and RGB as PNG (lossless, important for model prediction)
                    cv2.imwrite(str(depth_path), raw_images["depth"][cam_name])
                    cv2.imwrite(str(rgb_path), raw_images["rgb"][cam_name])
                
                self.save_queue.task_done()
            except Exception as e:
                print(f"[Image Server] Error in save_images_worker: {e}")
    
    def _save_images(self, raw_images: dict, step_count: int):
        """Enqueue images for background saving. Blocks if queue is full to avoid dropping frames."""
        t0 = time.perf_counter()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        # Copy images to avoid modification while in queue
        images_copy = {
            "depth": {k: v.copy() for k, v in raw_images["depth"].items()},
            "rgb": {k: v.copy() for k, v in raw_images["rgb"].items()}
        }
        # Use blocking put to ensure no frames are dropped
        self.save_queue.put((images_copy, step_count, timestamp))
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.save_images_profiler.record(elapsed_ms)

    
    def _prepare_depth_for_visualization(self, depth: np.ndarray) -> np.ndarray:
        # clip and scale to [0, 1]
        depth = np.clip(depth, self.config.near_clip, self.config.far_clip)
        depth = (depth - self.config.near_clip) / (self.config.far_clip - self.config.near_clip) # [0, 1]

        # [0, 1] -> [0, 255]
        return (depth * 255.0).astype(np.uint8) # [0, 255] # uint8, for visualization

    def _visualize_images(self, raw_image: dict[str, np.ndarray]):
        depth_for_vis = [self._prepare_depth_for_visualization(frame) for frame in raw_image["depth"].values()]
        concatenated_depth = np.concatenate(depth_for_vis, axis=0)
        self._display_frame("depth", concatenated_depth)
        concatenated_rgb = np.concatenate(list(raw_image["rgb"].values()), axis=0)
        self._display_frame("rgb", concatenated_rgb)
    
    def send_process(self):
        # this thread should be running at 10hz

        render_frequency = 10
        rate_limiter = RateLimiter(render_frequency)

        step_count = 0
        while True:

            # 0. grab frames from cameras
            raw_image = self.camera_wrapper.get_frames()
            
            # 1. get depth frames
            if self.config.gum_enabled:
                depth_frames = [self.gum.predict(x) for x in raw_image["rgb"].values()]
            else:
                depth_frames = list(raw_image["depth"].values())  

            # 2. Process data to be sent to policy;
            depth_frames = [self._resize_clip_expand_transpose(frame) for frame in depth_frames]
             # Concatenate frames before channel dimension (axis=0)
            # [C, H, W] -> [N, C, H, W] N is the number of cameras; front camera is first, back camera is second
            full_image = np.stack(depth_frames, axis=0)

            # 3. copy to shared memory for policy
            try:
                np.copyto(self.img_array, full_image)
            except Exception as e:
                print(f"[Image Server] Failed to copy to shared memory: {e}")
                continue
                
            # 4. save and visualize images
            if self.config.enable_save_images:
                self._save_images(raw_image, step_count)

            if self.config.enable_visualize_images:
                self._visualize_images(raw_image)

            if step_count % 10 == 0:
                print(f"[Image Server] Rate limiter stats: {rate_limiter.get_stats()}")
                if self.config.enable_save_images:
                    stats = self.save_images_profiler.get_stats()
                    stats["queue_size"] = self.save_queue.qsize()
                    print(f"[Image Server] save_images stats: {stats}")

            rate_limiter.sleep()
            step_count += 1
        
    
######################## temporary functions ########################



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


if __name__ == "__main__":
    image_server = ImageServer(ZedCamerasWrapper(ZedCamerasConfig()), ImageServerConfig(), None)
    thread = threading.Thread(target=image_server.send_process)
    thread.start()
    time.sleep(10)
    thread.join()