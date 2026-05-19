"""Whole Body Tracking curriculum presets for the G1 robot."""

from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg

_average_episode_tracker_term = CurriculumTermCfg(
    func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker",
    params={},
)

g1_29dof_wbt_curriculum = CurriculumManagerCfg(
    params={
        "num_compute_average_epl": 1000,
    },
    setup_terms={
        "average_episode_tracker": _average_episode_tracker_term,
    },
    reset_terms={},
    step_terms={},
)

g1_29dof_wbt_curriculum_w_object = CurriculumManagerCfg(
    params={
        "num_compute_average_epl": 1000,
    },
    setup_terms={
        "average_episode_tracker": _average_episode_tracker_term,
        "w_object_difficulty_curriculum": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:WObjectDifficultyCurriculum",
            params={
                # Keep curriculum logic available, but disable by default for w-object training.
                "enabled": False,
                "initial_lambda": 0.0,
                "lambda_step_up": 0.01,
                "lambda_step_down": 0.01,
                "early_termination_threshold": 0.30,
                "similarity_metric_key": "motion/error_object_ref_pos",
                "similarity_sigma": 0.50,
                "similarity_threshold": 0.60,
                "assist_beta_max": 1.0,
                "object_pos_kp": 4.0,
                "object_lin_vel_kd": 2.0,
                "object_rot_kp": 3.0,
                "object_ang_vel_kd": 1.5,
                "object_force_to_velocity": 1.0,
                "object_torque_to_ang_velocity": 1.0,
                # Keep assist increments conservative to avoid physics instability in large-batch training.
                "object_max_delta_lin": 0.05,
                "object_max_delta_ang": 0.10,
                "object_max_lin_vel_abs": 2.5,
                "object_max_ang_vel_abs": 6.0,
                "object_max_pos_err": 0.50,
                "object_max_rot_err": 1.20,
            },
        ),
    },
    reset_terms={},
    step_terms={},
)

__all__ = ["g1_29dof_wbt_curriculum", "g1_29dof_wbt_curriculum_w_object"]
