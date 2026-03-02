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
                "enabled": True,
                "initial_lambda": 0.0,
                "lambda_step_up": 0.01,
                "lambda_step_down": 0.01,
                "early_termination_threshold": 0.30,
                "similarity_metric_key": "motion/error_body_pos",
                "similarity_sigma": 0.50,
                "similarity_threshold": 0.60,
                "enable_task_mixing": False,
                "imitation_prob_start": 1.0,
                "imitation_prob_target": 0.5,
                "assist_beta_max": 1.0,
                "assist_kp_pos": 4.0,
                "assist_kd_lin_vel": 2.0,
                "assist_kd_ang_vel": 1.5,
                "assist_max_delta_lin": 0.20,
                "assist_max_delta_ang": 0.20,
                "generalization_noise_scale_min": 1.0,
                "generalization_noise_scale_max": 2.0,
                "generalization_start_zero_prob_scale_min": 1.0,
                "generalization_start_zero_prob_scale_max": 0.25,
            },
        ),
    },
    reset_terms={},
    step_terms={},
)

__all__ = ["g1_29dof_wbt_curriculum", "g1_29dof_wbt_curriculum_w_object"]
