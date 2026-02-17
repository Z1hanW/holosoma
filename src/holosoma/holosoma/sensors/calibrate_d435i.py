"""On-Chip Calibration (OCC) for Intel RealSense D435i.

Recalibrates the stereo module using the built-in IR projector pattern.
No external calibration target is needed. The updated calibration is
written to the camera's firmware so it persists across reboots.

Usage:
    python src/holosoma/holosoma/sensors/calibrate_d435i.py
    python src/holosoma/holosoma/sensors/calibrate_d435i.py --serial 1234567890
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import cv2
import numpy as np


def list_devices():
    """Print all connected RealSense devices and their serial numbers."""
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("No RealSense devices found.")
        return []

    print(f"Found {len(devices)} RealSense device(s):\n")
    serials = []
    for i, dev in enumerate(devices):
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        fw = dev.get_info(rs.camera_info.firmware_version)
        serials.append(serial)
        print(f"  [{i}] {name}  serial={serial}  firmware={fw}")
    print()
    return serials


def get_calibration_health(device) -> float | None:
    """Query the current calibration health score from the device.

    Returns a float where 0 = perfect, higher = worse.
    Returns None if the query is not supported.
    """
    import pyrealsense2 as rs

    try:
        auto_cal = rs.auto_calibrated_device(device)
        health_json = auto_cal.get_calibration_table()
        # The health score is not directly in the table; we return None
        # and rely on OCC to report it.
        return None
    except Exception:
        return None


_WINDOW_NAME = "D435i Calibration"
_window_created = False


def _show_preview(frames, colorizer, status: str = ""):
    """Display depth (colorized) and left IR side-by-side in an OpenCV window."""
    global _window_created

    depth_frame = frames.get_depth_frame()
    ir_frame = frames.get_infrared_frame(1)

    panels = []
    if depth_frame:
        depth_color = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        # Compute fill percentage (non-zero depth pixels)
        depth_raw = np.asanyarray(depth_frame.get_data())
        total = depth_raw.size
        valid = np.count_nonzero(depth_raw)
        fill_pct = 100.0 * valid / total if total > 0 else 0.0
        cv2.putText(
            depth_color, f"Depth fill: {fill_pct:.1f}%", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        panels.append(depth_color)
    if ir_frame:
        ir_arr = np.asanyarray(ir_frame.get_data())
        # Normalize IR to full 0-255 range so it's actually visible
        ir_norm = cv2.equalizeHist(ir_arr)
        ir_bgr = cv2.cvtColor(ir_norm, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            ir_bgr, "IR (left)", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        panels.append(ir_bgr)

    if panels:
        canvas = np.concatenate(panels, axis=1)
        if status:
            cv2.putText(
                canvas, status, (10, canvas.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
        if not _window_created:
            cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(_WINDOW_NAME, min(canvas.shape[1], 1280), min(canvas.shape[0], 480))
            _window_created = True
        cv2.imshow(_WINDOW_NAME, canvas)


def _preview_until_ready(pipeline, colorizer):
    """Show live preview until the user presses Enter or 'q'."""
    import select

    ready = False
    while not ready:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=200)
            _show_preview(frames, colorizer, "Press ENTER to start calibration")
        except Exception:
            pass
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            pipeline.stop()
            cv2.destroyAllWindows()
            print("Aborted by user.")
            sys.exit(0)
        # Non-blocking check for Enter on stdin
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                ready = True
        except Exception:
            # select may not work on all platforms; fall back to timeout
            pass
    print()


def run_on_chip_calibration(
    serial: str = "",
    speed: int = 2,
    write_to_firmware: bool = True,
):
    """Run On-Chip Calibration on a D435i device.

    Parameters
    ----------
    serial : str
        Camera serial number. Empty string uses the first available device.
    speed : int
        Calibration speed: 0=very fast, 1=fast, 2=medium (default), 3=slow.
        Slower speeds are more thorough.
    write_to_firmware : bool
        If True, write the new calibration to the device firmware (persistent).
    """
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        print("ERROR: No RealSense devices found.")
        sys.exit(1)

    # Find the target device
    device = None
    for dev in devices:
        dev_serial = dev.get_info(rs.camera_info.serial_number)
        if serial == "" or dev_serial == serial:
            device = dev
            serial = dev_serial
            break

    if device is None:
        print(f"ERROR: Device with serial '{serial}' not found.")
        list_devices()
        sys.exit(1)

    name = device.get_info(rs.camera_info.name)
    fw = device.get_info(rs.camera_info.firmware_version)
    print(f"Target device: {name}  serial={serial}  firmware={fw}")
    print()

    # Hardware reset to ensure clean state (previous failed OCC can leave
    # the device in a bad state, causing "HW not ready" errors).
    print("Resetting device hardware...")
    device.hardware_reset()
    time.sleep(5)  # Wait for device to reconnect

    # Re-discover device after reset
    devices = ctx.query_devices()
    device = None
    for dev in devices:
        if dev.get_info(rs.camera_info.serial_number) == serial:
            device = dev
            break
    if device is None:
        print("ERROR: Device not found after hardware reset.")
        sys.exit(1)
    print("  Device reconnected.")
    print()

    # Cast to auto_calibrated_device
    try:
        auto_cal = rs.auto_calibrated_device(device)
    except Exception as exc:
        print(f"ERROR: Device does not support auto-calibration: {exc}")
        sys.exit(1)

    # Start depth-only pipeline (no explicit IR — OCC needs to control
    # the stereo/IR hardware itself).
    print("Starting depth pipeline...")
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device(serial)
    rs_config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    colorizer = rs.colorizer()
    pipeline.start(rs_config)

    # Warm up: read frames to let auto-exposure settle, show depth preview
    print("  Warming up with depth preview...")
    print("  Press 'q' in the preview window to abort.")
    print()
    for i in range(90):
        frames = pipeline.wait_for_frames()
        _show_preview(frames, colorizer, "Warming up...")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            pipeline.stop()
            cv2.destroyAllWindows()
            print("Aborted by user.")
            sys.exit(0)
        if i in (0, 44, 89):
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                d = np.asanyarray(depth_frame.get_data())
                fill = 100.0 * np.count_nonzero(d) / d.size
                print(f"  [frame {i}] depth fill: {fill:.1f}%")
    print("  Pipeline ready.")
    print()

    # Save the current calibration table as a backup
    print("Saving current calibration table as backup...")
    original_table = auto_cal.get_calibration_table()
    print(f"  Backup size: {len(original_table)} bytes")
    print()

    # Show depth preview until user presses Enter
    print("Position the camera at a textured scene (0.5-1.5 m away).")
    print("Press ENTER in the terminal to start calibration, or 'q' in the window to abort.")
    _preview_until_ready(pipeline, colorizer)
    cv2.destroyAllWindows()

    # Run On-Chip Calibration — pipeline stays running, we just stop
    # calling wait_for_frames() so OCC has uncontested access.
    speed_names = {0: "very fast", 1: "fast", 2: "medium", 3: "slow"}
    print(f"Starting On-Chip Calibration (speed={speed_names.get(speed, speed)})...")
    print("  Keep the camera stationary. This may take a while...")
    print()

    def _occ_progress(progress: float):
        print(f"  OCC progress: {progress:.0f}%")

    try:
        # API: run_on_chip_calibration(json, callback, timeout_ms)
        #   returns (calibration_table, (health_1, health_2))
        new_table, (health, health2) = auto_cal.run_on_chip_calibration(
            json.dumps({"speed": speed}),
            _occ_progress,
            120000,  # timeout (ms)
        )
    except Exception as exc:
        pipeline.stop()
        print(f"ERROR: On-Chip Calibration failed: {exc}")
        sys.exit(1)

    pipeline.stop()

    print(f"Calibration completed.")
    print(f"  Health score: {health:.4f} (secondary: {health2:.4f})")
    print(f"    (0 = perfect; < 0.25 = good; > 0.75 = consider recalibrating)")
    print()

    if health > 0.75:
        print("WARNING: Health score is high. The calibration result may not be ideal.")
        print("  Consider running again with a more textured scene or slower speed.")
        print()

    if write_to_firmware:
        print("Writing new calibration to device firmware...")
        try:
            auto_cal.set_calibration_table(new_table)
            auto_cal.write_calibration()
            print("  Calibration saved to firmware successfully.")
        except Exception as exc:
            print(f"ERROR: Failed to write calibration: {exc}")
            print("  Restoring original calibration...")
            try:
                auto_cal.set_calibration_table(original_table)
                auto_cal.write_calibration()
                print("  Original calibration restored.")
            except Exception as restore_exc:
                print(f"  WARNING: Failed to restore original calibration: {restore_exc}")
            sys.exit(1)
    else:
        print("Skipping firmware write (dry run).")

    print()
    print("Done. You can verify the calibration by running:")
    print(f"  realsense-viewer")
    print(f"  or: rs-enumerate-devices -c")


def main():
    parser = argparse.ArgumentParser(
        description="On-Chip Calibration (OCC) for Intel RealSense D435i",
    )
    parser.add_argument(
        "--serial", type=str, default="",
        help="Camera serial number. Omit to use the first available device.",
    )
    parser.add_argument(
        "--speed", type=int, default=2, choices=[0, 1, 2, 3],
        help="Calibration speed: 0=very fast, 1=fast, 2=medium (default), 3=slow.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List connected RealSense devices and exit.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run calibration but don't write to firmware.",
    )

    args = parser.parse_args()

    try:
        import pyrealsense2  # noqa: F401
    except ImportError:
        print("ERROR: pyrealsense2 is not installed.")
        print("  Install with: pip install pyrealsense2")
        sys.exit(1)

    if args.list:
        list_devices()
        return

    run_on_chip_calibration(
        serial=args.serial,
        speed=args.speed,
        write_to_firmware=not args.dry_run,
    )


if __name__ == "__main__":
    main()
