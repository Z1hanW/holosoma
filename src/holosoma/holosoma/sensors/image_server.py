import time
import threading
from queue import Queue
from collections import deque
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TypedDict
import cv2
import numpy as np
import torch
from multiprocessing import shared_memory
import tyro
from typing_extensions import Annotated, NotRequired
import holosoma.config_values.image_server
from holosoma.simulator.mujoco.mujoco import MujocoRendererWrapper
from holosoma.utils.rate import RateLimiter
from datetime import datetime
from holosoma.models.gum.infer import GUM
from holosoma.config_types.image_server import (
    ImageSaverConfig,
    ImageServerConfig,
    ImageVisualizerConfig,
)
from holosoma.sensors.zed import ZedCamerasConfig, ZedCamerasWrapper
from holosoma.sensors.utils import _prepare_depth_for_visualization


class FrameBundle(TypedDict):
    rgb: NotRequired[dict[str, np.ndarray | None]]
    depth: NotRequired[dict[str, np.ndarray | None]]
    depth_gum: NotRequired[dict[str, np.ndarray | None]]
    calibration: NotRequired[dict[str, dict[str, np.ndarray]]]


ImageServerCliConfig = Annotated[
    ImageServerConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults(
            holosoma.config_values.image_server.DEFAULTS
        )
    ),
]


class TimeProfiler:
    """Profiler for tracking time statistics."""
    
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

    @contextmanager
    def measure(self):
        """Context manager for measuring and recording elapsed time."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.record(elapsed_ms)
    
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


class ImageSaver:
    """Handles image saving logic with background worker threads."""
    
    def __init__(self, config: ImageSaverConfig, near_clip: float, far_clip: float):
        """Initialize ImageSaver.
        
        Args:
            config: Configuration for image saving
            near_clip: Near clipping plane for depth visualization
            far_clip: Far clipping plane for depth visualization
        """
        self.config = config
        self.image_root_dir = config.image_root_dir
        self.save_queue_maxsize = config.save_queue_maxsize
        self.save_workers = config.save_workers
        self.near_clip = near_clip
        self.far_clip = far_clip
        
        # Initialize save directory
        self._init_save_images_dir()
        
        # Initialize profiler
        self.profiler = TimeProfiler()
        
        # Initialize queue
        self.save_queue = Queue(maxsize=self.save_queue_maxsize if self.save_queue_maxsize > 0 else 0)
        self.save_queue_dropped_count = 0
        
        # Directory management
        self.created_camera_channel_dirs: set[tuple[str, str]] = set()
        self._dirs_lock = threading.Lock()
        
        # Start worker threads
        self.save_threads = []
        for i in range(self.save_workers):
            thread = threading.Thread(target=self._save_images_worker, daemon=True, name=f"SaveWorker-{i}")
            thread.start()
            self.save_threads.append(thread)
    
    def _init_save_images_dir(self):
        """Create a subdirectory for each session with timestamp."""
        session_dir = Path(self.image_root_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir.mkdir(exist_ok=True, parents=True)
        self.save_images_dir = session_dir
        print(f"[Image Saver] Initialized save images directory: {self.save_images_dir}")
    
    def _ensure_camera_dirs(self, names: list[str], channels: list[str]):
        """Pre-create directories for all channels and cameras."""
        with self._dirs_lock:
            for cam_name in names:
                for channel in channels:
                    camera_channel = (cam_name, channel)
                    if camera_channel in self.created_camera_channel_dirs:
                        continue
                    (self.save_images_dir / cam_name / channel).mkdir(exist_ok=True, parents=True)
                    self.created_camera_channel_dirs.add(camera_channel)

    def save_calibration(self, calibration_by_camera: dict[str, dict[str, np.ndarray]]):
        """Save camera intrinsics/extrinsics under each camera calibration directory."""
        if not calibration_by_camera:
            return

        for cam_name, calibration in calibration_by_camera.items():
            calibration_dir = self.save_images_dir / cam_name / "calibration"
            calibration_dir.mkdir(exist_ok=True, parents=True)

            intrinsics = calibration.get("intrinsics")
            extrinsics = calibration.get("extrinsics")

            if intrinsics is not None:
                np.save(calibration_dir / "intrinsics.npy", intrinsics)
            if extrinsics is not None:
                np.save(calibration_dir / "extrinsics.npy", extrinsics)

        print(f"[Image Saver] Saved calibration for cameras: {list(calibration_by_camera.keys())}")
    
    def _save_images_worker(self):
        """Background thread worker that saves images from the queue."""
        while True:
            try:
                item = self.save_queue.get()
                if item is None:  # Sentinel to stop
                    break
                
                raw_images, step_count, timestamp = item
                rgb_by_camera = raw_images.get("rgb", {})
                names = list(rgb_by_camera.keys())
                channels = [channel for channel in ("rgb", "depth", "depth_gum") if raw_images.get(channel)]
                self._ensure_camera_dirs(names, channels)

                for cam_name in names:
                    for channel in channels:
                        channel_frames = raw_images.get(channel, {})
                        frame = channel_frames.get(cam_name)
                        if frame is None:
                            continue

                        save_path = self.save_images_dir / cam_name / channel / f"{cam_name}_{step_count}_{timestamp}.png"
                        frame_for_png = (
                            _prepare_depth_for_visualization(
                                frame,
                                near_clip=self.near_clip,
                                far_clip=self.far_clip,
                            )
                            if channel in {"depth", "depth_gum"}
                            else frame
                        )
                        cv2.imwrite(str(save_path), frame_for_png)
                
                self.save_queue.task_done()
            except Exception as e:
                print(f"[Image Saver] Error in save_images_worker: {e}")
    
    def save(self, raw_images: FrameBundle, step_count: int):
        """Enqueue images for background saving. Blocks if queue is full to avoid dropping frames.
        
        Args:
            raw_images: Dictionary with optional image channels
            step_count: Step count for naming files
        """
        with self.profiler.measure():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            images_copy: FrameBundle = {}
            for channel in ("rgb", "depth", "depth_gum"):
                channel_frames = raw_images.get(channel)
                if not channel_frames:
                    continue
                images_copy[channel] = {
                    name: (frame.copy() if frame is not None else None)
                    for name, frame in channel_frames.items()
                }

            self.save_queue.put((images_copy, step_count, timestamp))
    
    def get_stats(self) -> dict[str, float]:
        """Get statistics about image saving operations."""
        stats = self.profiler.get_stats()
        stats["queue_size"] = self.save_queue.qsize()
        return stats
    
    def shutdown(self):
        """Shutdown worker threads gracefully."""
        # Send sentinel to each worker thread
        for _ in self.save_threads:
            self.save_queue.put(None)
        
        # Wait for threads to finish
        for thread in self.save_threads:
            thread.join()


class ImageVisualizer:
    """Handles image visualization logic."""
    
    def __init__(self, config: ImageVisualizerConfig):
        """Initialize ImageVisualizer.
        
        Args:
            config: Visualization window sizing configuration
        """
        self.near_clip = config.near_clip
        self.far_clip = config.far_clip
        self.scale = config.scale
    
    def _display_frame(self, name: str, frame: np.ndarray) -> bool:
        """Display a frame in a window.
        
        Args:
            name: Window name
            frame: Image frame to display
            
        Returns:
            True if 'q' key was pressed, False otherwise
        """
        if self.scale < 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(
                frame,
                (max(1, int(w * self.scale)), max(1, int(h * self.scale))),
                interpolation=cv2.INTER_CUBIC,
            )

        # window_name = f"Image Server Stream {name}"
        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # cv2.imshow(window_name, frame)
        cv2.imshow(f"Image Server Stream {name}", frame)
        return cv2.waitKey(1) & 0xFF == ord("q")

    def _depth_or_placeholder(
        self,
        depth: np.ndarray | None,
        height: int,
        width: int,
        label: str,
    ) -> np.ndarray:
        """Return depth visualization or a labeled placeholder when depth is unavailable."""
        if depth is None:
            vis = np.zeros((height, width), dtype=np.uint8)
            label = f"{label} not predicted"
        else:
            vis = _prepare_depth_for_visualization(
                depth,
                near_clip=self.near_clip,
                far_clip=self.far_clip,
            )

        cv2.putText(
            vis,
            label,
            (10, min(height - 10, 40)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            255,
            2,
            cv2.LINE_AA,
        )
        return vis
    
    def visualize(self, raw_image: FrameBundle):
        """Visualize depth and RGB images.
        
        Args:
            raw_image: Dictionary with optional image channels
        """
        rgb_by_camera = raw_image.get("rgb", {})
        names = list(rgb_by_camera.keys())
        if not names:
            return

        depth_channels = [
            channel for channel in ("depth", "depth_gum") if raw_image.get(channel)
        ]
        combined_depth_rows = []
        for cam_name in names:
            rgb_frame = rgb_by_camera.get(cam_name)
            if rgb_frame is None:
                continue
            h, w = rgb_frame.shape[:2]

            depth_tiles = [
                self._depth_or_placeholder(
                    raw_image.get(channel, {}).get(cam_name),
                    h,
                    w,
                    f"{cam_name}: {channel}",
                )
                for channel in depth_channels
            ]
            if depth_tiles:
                combined_depth_rows.append(np.concatenate(depth_tiles, axis=1))

        if combined_depth_rows:
            concatenated_depth = np.concatenate(combined_depth_rows, axis=0)
            self._display_frame("depth", concatenated_depth)

        rgb_frames = [rgb_by_camera[name] for name in names if rgb_by_camera.get(name) is not None]
        if not rgb_frames:
            return

        concatenated_rgb = np.concatenate(rgb_frames, axis=0)
        self._display_frame("rgb", concatenated_rgb)


class ImageServer:
    def __init__(self, camera_wrapper: MujocoRendererWrapper | ZedCamerasWrapper, cfg: ImageServerConfig):
        self.cfg: ImageServerConfig = cfg

        # Initialize camera wrapper
        self.camera_wrapper: ZedCamerasWrapper | MujocoRendererWrapper = camera_wrapper

        # Initialize shared memory
        self._init_shared_memory()

        # Initialize depth prediction models
        self.gum = GUM(cfg=self.cfg.gum_config, dtype=torch.bfloat16) if self.cfg.enable_gum_depth_prediction else None
        
        # Initialize delay buffer for frame delay
        self.num_delay_frames = self.cfg.num_delay_frames
        if self.num_delay_frames > 0:
            # Buffer needs to hold num_delay_frames + 1 frames to store current + delayed frames
            # When buffer is full, buffer[0] contains the frame from num_delay_frames steps ago
            # Buffer stores tuples of (step_count, full_image) for debugging
            self.delay_buffer: deque[tuple[int, np.ndarray]] = deque(maxlen=self.num_delay_frames + 1)
            print(f"[Image Server] Initialized delay buffer with {self.num_delay_frames} delay frames (will send frames from {self.num_delay_frames} steps ago)")
        else:
            self.delay_buffer = None
        
        # Initialize profilers for capturing and depth prediction
        self.capture_profiler = TimeProfiler()
        if self.gum:
            self.gum_profiler = TimeProfiler()
        
        # Initialize image saver and visualizer, for debugging and visualization
        if self.cfg.save_images:
            self.image_saver = ImageSaver(
                self.cfg.image_saver_config,
                near_clip=self.cfg.near_clip,
                far_clip=self.cfg.far_clip,
            )
            self._save_camera_calibration()
        else:
            self.image_saver = None

        if self.cfg.visualize_images:
            self.image_visualizer = ImageVisualizer(
                config=self.cfg.image_visualizer_config,
            )
        else:
            self.image_visualizer = None

    def _save_camera_calibration(self):
        """Save camera calibration right after ImageSaver initialization."""
        cameras = getattr(self.camera_wrapper, "cameras", None)
        if cameras is None:
            return

        calibration_by_camera: dict[str, dict[str, np.ndarray]] = {}
        for cam_name, camera in cameras.items():
            calibration = getattr(camera, "calibration", None)
            if calibration is None:
                continue
            calibration_by_camera[cam_name] = calibration

        if not calibration_by_camera:
            print("[Image Server] No camera calibration found to save.")
            return

        self.image_saver.save_calibration(calibration_by_camera)
    
    def _init_shared_memory(self):
        img_shm_name = "depth_img_shm"

        # Initialize shared memory
        expected_shape = [self.camera_wrapper.num_cameras, 1, self.cfg.resized_height, self.cfg.resized_width]
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
        # crop
        if any(v is not None for v in (self.cfg.crop_y_start, self.cfg.crop_y_end,
                                        self.cfg.crop_x_start, self.cfg.crop_x_end)):
            frame = frame[self.cfg.crop_y_start:self.cfg.crop_y_end,
                          self.cfg.crop_x_start:self.cfg.crop_x_end]

        # resize
        frame = cv2.resize(frame, (self.cfg.resized_width, self.cfg.resized_height), cv2.INTER_CUBIC)

        # clip and scale to [-0.5, 0.5] range
        frame = np.clip(frame, self.cfg.near_clip, self.cfg.far_clip)
        frame = (frame - self.cfg.near_clip) / (self.cfg.far_clip - self.cfg.near_clip) - 0.5

        # [H, W] -> [1, H, W]
        frame = np.expand_dims(frame, axis=0)
        return frame

    def _predict_gum_depth(self, frames: FrameBundle) -> dict[str, np.ndarray]:
        rgb_by_camera = frames.get("rgb", {})
        calibration_by_camera = frames.get("calibration")

        gum_depth: dict[str, np.ndarray] = {}
        for name, rgb in rgb_by_camera.items():
            calibration = calibration_by_camera.get(name)
            gum_depth[name] = self.gum.predict(
                side_by_side_image=rgb,
                camera_intrinsics=calibration["intrinsics"],
                camera_extrinsics=calibration["extrinsics"],
            )
        return gum_depth

    def _get_policy_depth_frames(self, frames: FrameBundle) -> dict[str, np.ndarray]:
        return frames.get(self.cfg.depth_source, {})


    def send_process(self):
        render_frequency = self.cfg.frame_rate
        rate_limiter = RateLimiter(render_frequency)

        step_count = 0
        while True:

            # 0. grab frames from cameras
            with self.capture_profiler.measure():
                all_frames: FrameBundle = dict(self.camera_wrapper.get_frames())

            if self.cfg.enable_gum_depth_prediction:
                with self.gum_profiler.measure():
                    all_frames["depth_gum"] = self._predict_gum_depth(all_frames)
            
            # 2. Process data to be sent to policy;
            depth_for_policy = all_frames[self.cfg.depth_source]
            depth_for_policy = [self._resize_clip_expand_transpose(frame) for frame in depth_for_policy.values()]
             # Concatenate frames before channel dimension (axis=0)
            # [C, H, W] -> [N, C, H, W] N is the number of cameras; front camera is first, back camera is second
            full_depth_for_policy = np.stack(depth_for_policy, axis=0)

            # 2.5. Add to delay buffer and get delayed image
            if self.delay_buffer is not None:
                # Note: When buffer is full (maxlen reached), deque automatically removes
                # the oldest item (leftmost) to make room for the new frame
                self.delay_buffer.append((step_count, full_depth_for_policy.copy()))
                
                # Wait until buffer is full before sending delayed frames
                if len(self.delay_buffer) <= self.num_delay_frames:
                    # Buffer not full yet, skip this frame
                    rate_limiter.sleep()
                    step_count += 1
                    continue
                
                # Retrieve the delayed frame (from num_delay_frames steps ago)
                # buffer[0] always contains the oldest frame (num_delay_frames steps ago)
                # because deque with maxlen automatically maintains the size
                delayed_step_count, delayed_image = self.delay_buffer[0]  # Oldest frame in buffer
                # print delayed_step_count for debugging purposes
            else:
                # No delay, use current frame
                delayed_step_count, delayed_image = step_count, full_depth_for_policy

            # 3. copy to shared memory for policy
            try:
                # print(f"[Image Server] Current step count: {step_count}, delayed step count: {delayed_step_count}")
                np.copyto(self.img_array, delayed_image)
            except Exception as e:
                print(f"[Image Server] Failed to copy to shared memory: {e}")
                continue
                
            # 4. save and visualize images
            if self.cfg.save_images:
                self.image_saver.save(
                    all_frames,
                    step_count,
                )

            if self.cfg.visualize_images:
                self.image_visualizer.visualize(
                    all_frames,
                )

            if step_count % 50 == 0:
                print(f"[Image Server] Rate limiter stats: {rate_limiter.get_stats()}")
                rate_limiter.reset()
                print(f"[Image Server] capture stats: {self.capture_profiler.get_stats()}")
                if self.cfg.enable_gum_depth_prediction:
                    print(f"[Image Server] GUM prediction stats: {self.gum_profiler.get_stats()}")

            rate_limiter.sleep()
            step_count += 1
    
if __name__ == "__main__":
    # Parse command line arguments using subcommand presets from config_values.image_server.
    cfg = tyro.cli(ImageServerCliConfig, default=holosoma.config_values.image_server.real)
    
    # Set ZED depth mode based on config toggle.
    depth_mode = "NEURAL" if cfg.enable_zed_depth_prediction else "NONE"
    zed_cfg = ZedCamerasConfig(
        terms={
            name: replace(camera_cfg, depth_mode=depth_mode)
            for name, camera_cfg in ZedCamerasConfig().terms.items()
        }
    )

    # Create image server with parsed config
    image_server = ImageServer(ZedCamerasWrapper(zed_cfg), cfg)
    thread = threading.Thread(target=image_server.send_process)
    thread.start()
    time.sleep(2)
    thread.join()