import sys
import threading
import time
from threading import Thread

import mujoco
import mujoco.viewer
import numpy as np
from loguru import logger
from mujoco import MjvCamera, Renderer

from holosoma_inference.utils.misc import resolve_holosoma_inference_path

# from loop_rate_limiters import RateLimiter
from holosoma_inference.utils.rate import RateLimiter

sys.path.append("../")

import imageio
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from holosoma_inference.sdk.bridge import ElasticBand, create_sdk2py_bridge
from holosoma_inference.utils.clock import ClockPub


class BaseSimulator:
    def __init__(self, config):
        self.config = config
        self.init_config()
        self.init_scene()

        self.init_factory()
        self.init_robot_bridge()

        # for more scenes
        self.init_subscriber()
        self.init_publisher()

        self.sim_thread = Thread(target=self.simulation_thread)
        self.record_thread = None
        if self.config.get("USE_RENDERER", False):
            self.record_thread = Thread(target=self.record_video, args=("simulation.mp4",), daemon=True)

        # Initialize clock publisher
        self.clock_pub = ClockPub()
        self.clock_pub.start()

    def init_subscriber(self):
        pass

    def init_publisher(self):
        pass

    def init_config(self):
        self.robot_config = self.config["ROBOT_CFG"]
        self.sdk_type = self.robot_config.get("SDK_TYPE", "unitree")
        self.num_dof = self.robot_config.NUM_JOINTS
        self.num_motor = self.robot_config.NUM_MOTORS
        self.sim_dt = self.config["SIMULATE_DT"]
        self.rate_limit_dt = self.config.get("RATE_LIMIT_DT", self.sim_dt)
        self.viewer_dt = self.config["VIEWER_DT"]
        self.torques = np.zeros(self.num_motor)
        self.node = None
        if self.config.get("USE_ROS", False):
            import rclpy

            rclpy.init(args=None)
            self.node = rclpy.create_node("simulator")
            self.logger = self.node.get_logger()
            self.rate = self.node.create_rate(1 / self.rate_limit_dt)
            thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
            thread.start()
        else:
            self.logger = logger
            self.rate = RateLimiter(1 / self.rate_limit_dt)

    def init_factory(self):
        self.lcm = None
        if self.sdk_type == "unitree":
            if self.config.get("INTERFACE", None):
                if sys.platform == "linux":
                    self.config["INTERFACE"] = "lo"
                elif sys.platform == "darwin":
                    self.config["INTERFACE"] = "lo0"
                else:
                    raise NotImplementedError("Only support Linux and MacOS.")
                ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.config["DOMAIN_ID"])
        elif self.sdk_type == "ros2":
            pass
        elif self.sdk_type == "booster":
            from booster_robotics_sdk import ChannelFactory

            ChannelFactory.Instance().Init(self.config["DOMAIN_ID"])
        else:
            raise NotImplementedError(f"SDK type {self.sdk_type} is not supported yet")
        self.logger.info(str.format("SDK TYPE: {0}", self.sdk_type))

    def init_scene(self):
        robot_scene_path = self.config["ROBOT_SCENE"]
        robot_scene_path = resolve_holosoma_inference_path(robot_scene_path)

        self.mj_model = mujoco.MjModel.from_xml_path(robot_scene_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt

        # Enable the elastic band
        if self.config["ENABLE_ELASTIC_BAND"]:
            self.elastic_band = ElasticBand()
            band_attached_link_name = self.config.get("BAND_ATTACHED_LINK", "torso_link")
            self.band_attached_link = self.mj_model.body(band_attached_link_name).id
            self.viewer = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, key_callback=self.combined_key_callback
            )
        else:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data, key_callback=self.key_callback)

    def init_robot_bridge(self):
        self.robot_bridge = create_sdk2py_bridge(self.mj_model, self.mj_data, self.robot_config, self.lcm)
        if self.config["USE_JOYSTICK"]:
            if sys.platform == "linux" and self.sdk_type == "unitree":
                # TODO [Yuanhang]: add other joystick support
                self.robot_bridge.setup_joystick(
                    device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"]
                )
            else:
                self.logger.warning("Joystick is not supported on Windows or MacOS.")

    def compute_torques(self):
        if self.robot_bridge.low_cmd:
            motor_cmd = list(self.robot_bridge.low_cmd.motor_cmd)
            try:
                for i in range(self.robot_bridge.num_motor):
                    if self.robot_bridge.use_sensor:
                        self.torques[i] = (
                            motor_cmd[i].tau
                            + motor_cmd[i].kp * (motor_cmd[i].q - self.mj_data.sensordata[i])
                            + motor_cmd[i].kd * (motor_cmd[i].dq - self.mj_data.sensordata[i + self.num_motor])
                        )
                    else:
                        self.torques[i] = (
                            motor_cmd[i].tau
                            + motor_cmd[i].kp * (motor_cmd[i].q - self.mj_data.qpos[7 + i])
                            + motor_cmd[i].kd * (motor_cmd[i].dq - self.mj_data.qvel[6 + i])
                        )
            except Exception as e:
                self.logger.error(str.format("Joint {0} not found in motor_cmd: {1}", i, e))
        # Set the torque limit
        self.torques = np.clip(self.torques, -self.robot_bridge.torque_limit, self.robot_bridge.torque_limit)

    def sim_step(self):
        self.robot_bridge.publish_low_state()
        if self.robot_bridge.joystick:
            self.robot_bridge.publish_wireless_controller()
        if self.config["ENABLE_ELASTIC_BAND"]:
            if self.elastic_band.enable:
                self.mj_data.xfrc_applied[self.band_attached_link, :3] = self.elastic_band.advance(
                    self.mj_data.qpos[:3], self.mj_data.qvel[:3]
                )
        self.compute_torques()
        self.mj_data.ctrl = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)

        # Publish clock sync
        self.clock_pub.publish(self.mj_data.time)

    def simulation_thread(self):
        sim_cnt = 0
        start_time = time.time()
        while self.viewer.is_running():
            self.sim_step()
            if sim_cnt % (self.viewer_dt / self.sim_dt) == 0:
                self.viewer.sync()

            # Get FPS
            sim_cnt += 1
            if sim_cnt % 100 == 0:
                end_time = time.time()
                self.logger.info(str.format("FPS: {0:.2f}", 100 / (end_time - start_time)))
                start_time = end_time
            self.rate.sleep()

    # -------------------------------------------------------------
    # Video recording
    # -------------------------------------------------------------
    def record_video(self, filename="simulation.mp4"):
        renderer = Renderer(self.mj_model, height=480, width=640)
        fps = int(1 / self.viewer_dt)

        # Create a camera object
        cam = MjvCamera()
        mujoco.mjv_defaultCamera(cam)

        # Get the body id of the root (adjust name if needed)
        root_body = self.mj_model.body("torso_link")
        root_id = root_body.id

        self.logger.info("Video recording thread started.")

        with imageio.get_writer(filename, fps=fps) as video:
            while self.viewer.is_running():
                root_pos = self.mj_data.xpos[root_id].copy()

                # Update camera to follow the root
                cam.lookat[:] = root_pos
                cam.distance = 3.0
                cam.azimuth = 90.0
                cam.elevation = -20.0

                # Render using custom camera
                renderer.update_scene(self.mj_data, camera=cam)
                frame = renderer.render()

                if getattr(self.elastic_band, "video_recording", False):
                    video.append_data(frame)

                time.sleep(self.viewer_dt)

        self.logger.info("Video saved to %s", filename)

    def key_callback(self, key):
        """Handle keyboard input for simulation control."""
        import glfw

        if key == glfw.KEY_BACKSPACE:
            self.clock_pub.restart()
            self.logger.info("Clock restarted (backspace pressed)")

    def combined_key_callback(self, key):
        """Combined key callback for elastic band and clock control."""
        self.key_callback(key)
        if hasattr(self, "elastic_band"):
            self.elastic_band.mujoco_key_callback(key)
