from __future__ import annotations

import sys

KNOWN_MODES = ("rollout", "replay", "perception", "physics", "smpl", "joints")

USAGE = """\
Usage: python -m holosoma.visualize [mode] [args...]
Modes:
  rollout     Replay physics rollout .npz files (default)
  replay      Replay motion dataset files
  perception  Visualize perception rollouts
  physics     Physics rollout with policy checkpoint + motion input
  smpl        Visualize SMPL human mesh with optional geometry
  joints      Visualize joint positions from npz (global_joint_positions)

Rollout (default) example, config-first like eval_agent.py:
  python -m holosoma.visualize exp:g1-29dof-wbt-videomimic-mlp \\
    --rollout-dir /abs/path/to/rollouts \\
    # --rollout-file /abs/path/to/rollout_0001.npz \\
    # --terrain-obj-path /abs/path/to/obj_dir \\
    # --recenter=False

Replay (motion) example:
  python -m holosoma.visualize replay exp:g1-29dof-wbt-videomimic-mlp \\
    # --command.setup_terms.motion_command.params.motion_config.motion_file /abs/path \\
    # --command.setup_terms.motion_command.params.motion_config.motion_clip_name clip_0001

Perception example:
  python -m holosoma.visualize perception exp:g1-29dof-wbt-videomimic-mlp \\
    # perception:camera_depth_d435i_rendered

Physics rollout example:
  python -m holosoma.visualize physics \\
    --checkpoint /abs/path/to/model.pt \\
    --motion-dir /abs/path/to/motion_folder \\
    --geometry-dir /abs/path/to/obj_folder \\
    # --geometry-metadata /abs/path/to/metadata.json

SMPL mesh example:
  python -m holosoma.visualize smpl \\
    --motion-dir /abs/path/to/motion_folder \\
    --geometry-dir /abs/path/to/obj_folder \\
    --smpl-model-path /abs/path/to/body_models

Joints example:
  python -m holosoma.visualize joints \\
    --motion-dir /abs/path/to/joint_npz_folder \\
    # --geometry-dir /abs/path/to/obj_or_dir \\
    # --start-clip clip_name

Note: ExperimentConfig overrides are the same as eval_agent.py and can be appended here.
"""


def _print_usage() -> None:
    print(USAGE)


def _run(module_main, argv: list[str]) -> None:
    prev_argv = sys.argv
    sys.argv = [prev_argv[0], *argv]
    try:
        module_main()
    finally:
        sys.argv = prev_argv


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_usage()
        return

    if argv[0] in KNOWN_MODES:
        mode = argv[0]
        rest = argv[1:]
        if rest and rest[0] in ("-h", "--help", "help"):
            _print_usage()
            return
    else:
        mode = "rollout"
        rest = argv

    if mode == "rollout":
        from holosoma.viser_rollout import main as rollout_main

        _run(rollout_main, rest)
        return

    if mode == "replay":
        from holosoma.viser_replay import main as replay_main

        _run(replay_main, rest)
        return

    if mode == "perception":
        from holosoma.viser_perception import main as perception_main

        _run(perception_main, rest)
        return

    if mode == "physics":
        from holosoma.viser_physics_rollout import main as physics_main

        _run(physics_main, rest)
        return

    if mode == "smpl":
        from holosoma.viser_smpl_geometry import main as smpl_main

        _run(smpl_main, rest)
        return

    if mode == "joints":
        from holosoma.viser_joint_positions import main as joints_main

        _run(joints_main, rest)
        return

    print(f"Unknown mode: {mode}\n")
    _print_usage()


if __name__ == "__main__":
    main()
