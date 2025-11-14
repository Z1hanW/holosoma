# Inference Pipeline API 

# Nodes

## `policy_inference_server`

**Purpose**: Runs ONNX policy inference and manages observation history.

**Subscribes**:
* `/robot_state` (`RobotState.msg`)
* `/user_commands` (`UserCommands.msg`)

**Publishes**:
* `/policy_actions` (`PolicyActions.msg`) @ configurable rate (default 50Hz)

**Services**:
* `~/policy/start` (`std_srvs/Empty`) - Start policy inference. Will perform `~/policy/reset` if not in reset pose already.
* `~/policy/stop `(`std_srvs/Empty`) - Stop policy inference. Set actions to 0.
* `~/policy/reset` (`std_srvs/Empty`) - Reset policy state & perform safe reset to default joint angles.

NOTE: Reset behavior is not yet implemented and will be a part of a future CR.

**Parameters**:
* `policy_path`: Path to ONNX policy model file (required)
* `inference_rate`: Policy inference rate in Hz (default: 50.0)

---


## `human_input_handler`

**Purpose**: Handles & unifies controller input and publishes user commands

**Publishes**:
* `/user_commands` (`UserCommands.msg`) @ configurable rate (default 10Hz)
`
**Service Clients**:
* `/policy_inference_server/policy/start` - Triggered by controller start button
* `/policy_inference_server/policy/stop` - Triggered by controller stop button
* `/policy_inference_server/policy/reset` - Triggered by controller reset button

**Parameters**:
* `input_device`: Controller type - "xbox360", "keyboard", "ps4" (default: "xbox360")
* `scale_lin_velocity_ms`: Linear velocity scaling factor (default: 1.0)
* `scale_ang_velocity_rads`: Angular velocity scaling factor (default: 1.0)
* `publish_rate_hz`: Publishing rate for user commands (default: 10.0)

---

## `robot_node`
**Purpose**: Abstract the HW implementation of the robot

NOTE: This API is implemented for sim2sim scenario by `Ros2Bridge`. It's still TODO for a sim2real setting for booster & unitree.

**Subscribes**:
* `/policy_actions` (`PolicyActions.msg`)
* `/user_commands` (`UserCommands.msg`)
`
**Publishes**:
* `/robot_state` (`RobotState.msg`)

---

# Messages

| Message Type | Description |
|------------|-------------|
| `RobotState.msg` | Encapsulates the complete state of the robot including base pose, velocities, and joint states. May be extended with sensor readings in the future. |
| `PolicyActions.msg` | Contains the output actions from the policy inference, including desired joint states and control mode. |
| `UserCommands.msg` | Represents high-level user input commands for controlling the robot's behavior. |

## Code Organization

ROS2 is chosen for the runtime code structuring & environment. There are a lot of benefits of using a framework like ROS2 to this approach (pub-sub, code organization, familiar to most people, existing tooling). One key benefit for FAR team specifically will be the improved collaboration across different projects in the future. For example, structuring ROS2 will enable cross-team collaboration with the already established GMP/FAR-pi workspace: https://code.amazon.com/packages/FAR-pi/trees/mainline/--/rfmpi/ros_workspace/src, as well as with the Nav2 codebase.


