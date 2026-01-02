import torch


def get_obs_and_act_sizes(config):
    robot = config.asset.asset_file_name.split("/")[-1].split(".")[0]

    if robot == "amp_humanoid":
        num_key_bodies = 4  # TODO: currently hardcoded.
        dof_obs_size = 72
        num_act = 28

        if config.use_max_coords_obs:
            num_obs = 1 + 15 * (3 + 6 + 3 + 3) - 3
        else:
            num_obs = (
                13 + dof_obs_size + 28 + 3 * num_key_bodies
            )  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]

    elif (
        robot == "amp_humanoid_relaxed"
        or robot == "amp_humanoid_3d"
    ):
        num_key_bodies = 4  # TODO: currently hardcoded.
        dof_obs_size = 72
        num_act = 36

        if config.use_max_coords_obs:
            num_obs = 1 + 15 * (3 + 6 + 3 + 3) - 3
        else:
            num_obs = (
                13 + dof_obs_size + 36 + 3 * num_key_bodies
            )  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]

    elif (
            robot == "amp_humanoid_sword_shield"
            or robot == "amp_humanoid_sword_shield_3d"
    ):
        num_key_bodies = 6  # TODO: currently hardcoded.
        dof_obs_size = 78
        num_act = 31

        if config.use_max_coords_obs:
            num_obs = 1 + 17 * (3 + 6 + 3 + 3) - 3
        else:
            num_obs = 13 + dof_obs_size + 31 + 3 * num_key_bodies

    elif robot == "ov_humanoid":
        num_key_bodies = 4  # TODO: currently hardcoded.
        dof_obs_size = 90
        num_obs = (
            13 + dof_obs_size + 37 + 3 * num_key_bodies
        )  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        num_act = 37

    elif robot == "ov_humanoid_sword_shield":
        num_key_bodies = 4  # TODO: currently hardcoded.
        dof_obs_size = 96
        num_act = 40

        num_obs = (
            13 + dof_obs_size + 40 + 3 * num_key_bodies
        )  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]

    elif (
            robot == "smpl_humanoid"
            or robot == "smpl_humanoid_mesh"
            or robot == "smpl_humanoid_no_limits"
            or robot == "smpl_box_humanoid"
            or robot == "smpl_humanoid_pulse"
    ):
        num_key_bodies = 4  # TODO: currently hardcoded.
        dof_obs_size = 138
        num_act = 69

        if config.use_max_coords_obs:
            num_obs = 1 + 24 * (3 + 6 + 3 + 3) - 3
        else:
            raise NotImplementedError
    elif (
            robot == "smplx_box_humanoid"
    ):
        num_key_bodies = 4  # TODO: currently hardcoded.
        dof_obs_size = 306
        num_act = 153

        if config.use_max_coords_obs:
            num_obs = 1 + 52 * (3 + 6 + 3 + 3) - 3  # height + num_bodies * 15 (pos + vel + rot + ang_vel) - root_pos
        else:
            raise NotImplementedError
    elif robot == "g1_29dof":
        num_key_bodies = len(config.key_bodies)
        dof_obs_size = 174
        num_act = 29

        if config.use_max_coords_obs:
            num_bodies = getattr(config, "num_bodies", 32)
            num_obs = 1 + num_bodies * (3 + 6 + 3 + 3) - 3
        else:
            num_obs = 13 + dof_obs_size + num_act + 3 * num_key_bodies
    else:
        raise ValueError(f"Unsupported character config file: {robot}")

    if config.enable_task_obs:
        num_obs += config.num_task_obs
    # import pdb;pdb.set_trace()
    return num_obs, num_act


def get_num_jd_obs(config):
    robot = config.asset.asset_file_name.split("/")[-1].split(".")[0]

    if robot == "amp_humanoid":
        dof_obs_size = 72
        num_act = 28

    elif (
            robot == "amp_humanoid_relaxed"
            or robot == "amp_humanoid_3d"
    ):
        dof_obs_size = 72
        num_act = 36

    elif (
            robot == "amp_humanoid_sword_shield"
            or robot == "amp_humanoid_sword_shield_3d"
    ):
        dof_obs_size = 78
        num_act = 31

    elif robot == "ov_humanoid":
        dof_obs_size = 90
        num_act = 37

    elif robot == "ov_humanoid_sword_shield":
        dof_obs_size = 96
        num_act = 40


    elif (
            robot == "smpl_humanoid"
            or robot == "smpl_humanoid_mesh"
            or robot == "smpl_humanoid_no_limits"
            or robot == "smpl_box_humanoid"
            or robot == "smpl_humanoid_pulse"
    ):
        dof_obs_size = 138
        num_act = 69

    elif (
            robot == "smplx_box_humanoid"
    ):
        dof_obs_size = 306
        num_act = 153
    elif robot == "g1_29dof":
        dof_obs_size = 174
        num_act = 29

    else:
        raise ValueError(f"Unsupported character config file: {robot}")

    num_disc_obs_per_step = 13 + dof_obs_size + num_act + 3 * len(config.key_bodies)

    return num_disc_obs_per_step


def isaacgym_asset_file_to_stats(asset_file: str, num_key_bodies: int, use_max_coords: bool):
    if asset_file == "mjcf/amp_humanoid.xml":
        dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
        dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
        dof_obs_size = 72
        num_act = 28

        if use_max_coords:
            num_obs = 1 + 15 * (3 + 6 + 3 + 3) - 3
        else:
            num_obs = (
                13 + dof_obs_size + 28 + 3 * num_key_bodies
            )  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]

    elif (
        asset_file == "mjcf/amp_humanoid_relaxed.xml"
        or asset_file == "mjcf/amp_humanoid_3d.xml"
    ):
        # pelvis, torso, head, right_upper_arm, right_lower_arm, right_hand,
        # left_upper_arm, left_lower_arm, left_hand, right_thigh, right_shin,
        # right_foot, left_thigh, left_shin, left_foot
        dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
        dof_offsets = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]
        dof_obs_size = 72
        num_act = 36

        if use_max_coords:
            num_obs = 1 + 15 * (3 + 6 + 3 + 3) - 3
        else:
            num_obs = (
                13 + dof_obs_size + 36 + 3 * num_key_bodies
            )  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]

    elif asset_file == "mjcf/amp_humanoid_sword_shield.xml":
        dof_body_ids = [1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16]
        dof_offsets = [0, 3, 6, 9, 10, 13, 16, 17, 20, 21, 24, 27, 28, 31]
        dof_obs_size = 78
        num_act = 31

        if use_max_coords:
            num_obs = 1 + 17 * (3 + 6 + 3 + 3) - 3
        else:
            num_obs = 13 + dof_obs_size + 31 + 3 * num_key_bodies

    elif asset_file == "mjcf/amp_humanoid_sword_shield_3d.xml":
        dof_body_ids = [1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16]
        dof_offsets = [0, 3, 6, 9, 10, 13, 16, 17, 20, 21, 24, 27, 28, 31]
        dof_obs_size = 78
        num_act = 31

        if use_max_coords:
            num_obs = 1 + 17 * (3 + 6 + 3 + 3) - 3
        else:
            num_obs = 13 + dof_obs_size + 31 + 3 * num_key_bodies

    elif asset_file == "mjcf/ov_humanoid.xml":
        dof_body_ids = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        dof_offsets = [0, 3, 6, 9, 12, 15, 16, 19, 22, 23, 26, 27, 30, 33, 34, 37]
        dof_obs_size = 90
        num_obs = (
            13 + dof_obs_size + 37 + 3 * num_key_bodies
        )  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        num_act = 37
    elif asset_file == "mjcf/ov_humanoid_sword_shield.xml":
        dof_body_ids = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 14, 15, 16, 17, 18, 19]
        dof_offsets = [0, 3, 6, 9, 12, 15, 16, 19, 22, 25, 26, 29, 30, 33, 36, 37, 40]
        dof_obs_size = 96
        num_obs = (
            13 + dof_obs_size + 40 + 3 * num_key_bodies
        )  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        num_act = 40

    elif (
            asset_file == "mjcf/smpl_humanoid_mesh.xml"
            or asset_file == "mjcf/smpl_humanoid.xml"
            or asset_file == "mjcf/smpl_humanoid_no_limits.xml"
            or asset_file == "mjcf/smpl_box_humanoid.xml"
            or asset_file == "mjcf/smpl_humanoid_pulse.xml"
    ):
        # bodies_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
        #                 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist',
        #                 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
        dof_body_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        dof_offsets = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69]
        dof_obs_size = 138
        num_act = 69

        if use_max_coords:
            num_obs = 1 + 24 * (3 + 6 + 3 + 3) - 3
        else:
            # TODO: fix this number
            num_obs = 13 + dof_obs_size + 31 + 3 * num_key_bodies
    elif (
            asset_file == "mjcf/smplx_box_humanoid.xml"
    ):
        # bodies_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
        # 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Index1',
        # 'L_Index2', 'L_Index3', 'L_Middle1', 'L_Middle2', 'L_Middle3', 'L_Pinky1', 'L_Pinky2', 'L_Pinky3', 'L_Ring1',
        # 'L_Ring2', 'L_Ring3', 'L_Thumb1', 'L_Thumb2', 'L_Thumb3', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist',
        # 'R_Index1', 'R_Index2', 'R_Index3', 'R_Middle1', 'R_Middle2', 'R_Middle3', 'R_Pinky1', 'R_Pinky2', 'R_Pinky3',
        # 'R_Ring1', 'R_Ring2', 'R_Ring3', 'R_Thumb1', 'R_Thumb2', 'R_Thumb3']
        dof_body_ids = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
        dof_offsets = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 135, 138, 141, 144, 147, 150, 153]
        dof_obs_size = 306
        num_act = 153

        if use_max_coords:
            num_obs = 1 + 52 * (3 + 6 + 3 + 3) - 3
        else:
            raise NotImplementedError
    elif asset_file == "mjcf/g1_29dof.xml":
        dof_body_ids = [
            1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13,
            15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31,
        ]
        dof_offsets = list(range(0, 30))
        dof_obs_size = 174
        num_act = 29

        if use_max_coords:
            num_obs = 1 + 32 * (3 + 6 + 3 + 3) - 3
        else:
            num_obs = 13 + dof_obs_size + num_act + 3 * num_key_bodies
    else:
        raise ValueError(f"Unsupported character config file: {asset_file}")

    return dof_body_ids, dof_offsets, dof_obs_size, num_obs, num_act


def isaacsim_asset_file_to_stats(robot_type, device):
    humanoid_positions = torch.tensor([0, 0, 0.89])
    if robot_type == "amp_humanoid_sword_shield":
        bodies_names = ['pelvis', 'torso', 'head', 'right_upper_arm', 'right_lower_arm', 'right_hand',
                        'sword', 'left_upper_arm', 'left_lower_arm', 'shield', 'left_hand', 'right_thigh',
                        'right_shin', 'right_foot', 'left_thigh', 'left_shin', 'left_foot']
        num_obs = 1 + 17 * (3 + 6 + 3 + 3) - 3
        num_act = 31
        num_dof = 31
        num_bodies = 17
        dof_obs_size = 78
        dof_body_ids_gym = [1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16]
        dof_offsets_gym = [0, 3, 6, 9, 10, 13, 16, 17, 20, 21, 24, 27, 28, 31]
        dof_offsets_sim = [0, 3, 6, 9, 12, 15, 18, 19, 20, 21, 22, 25, 28, 31]
        dof_offset_indices_isaac_gym_to_sim = [0, 1, 2, 17, 18, 19, 24, 25, 26, 3, 4, 5, 6, 7, 8, 13, 14, 15, 20, 27,
                                               9, 16, 21, 22, 23, 28, 29, 30, 10, 11, 12]
        # TODO: dof_body_ids_sim = ?
        # TODO: dof_body_indices_isaac_gym_to_sim = ?

        contact_body_ids = torch.tensor([13, 16], device=device, dtype=torch.long)  # ["right_foot", "left_foot"]
        key_body_ids = torch.tensor([5, 10, 13, 16, 6, 9], device=device, dtype=torch.long)  # ["right_hand", "left_hand", "right_foot", "left_foot", "sword", "shield"]

    elif robot_type == "amp_humanoid":
        bodies_names = ['pelvis', 'torso', 'head', 'right_upper_arm', 'right_lower_arm', 'right_hand',
                        'left_upper_arm', 'left_lower_arm', 'left_hand', 'right_thigh', 'right_shin',
                        'right_foot', 'left_thigh', 'left_shin', 'left_foot']
        num_obs = 1 + 15 * (3 + 6 + 3 + 3) - 3
        num_act = 28
        num_dof = 28
        num_bodies = 15
        dof_obs_size = 72
        dof_body_ids_gym = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
        # dof_body_ids_sim = [1, 4, 5, 9, 6, 10, 2, 7, 11, 3, 8, 12]
        # dof_body_indices_isaac_gym_to_sim = [0, 1, 9, 12, 2, 3, 6, 10, 13, 4, 7, 11, 14, 5, 8]
        dof_offsets_gym = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
        dof_offsets_sim = [0, 3, 6, 9, 12, 15, 18, 19, 20, 21, 22, 25, 28]
        dof_offset_indices_isaac_gym_to_sim = [0, 1, 2, 14, 15, 16, 21, 22, 23, 3, 4, 5, 6, 7, 8, 10, 11, 12, 17, 24,
                                               9, 13, 18, 19, 20, 25, 26, 27]

        contact_body_ids = torch.tensor([11, 14], device=device, dtype=torch.long)  # ["right_foot", "left_foot"]
        key_body_ids = torch.tensor([5, 8, 11, 14], device=device, dtype=torch.long)  # ["right_hand", "left_hand", "right_foot", "left_foot"]

    elif robot_type == "amp_humanoid_3d":
        bodies_names = ['pelvis', 'torso', 'head', 'right_upper_arm', 'right_lower_arm', 'right_hand',
                        'left_upper_arm', 'left_lower_arm', 'left_hand', 'right_thigh', 'right_shin',
                        'right_foot', 'left_thigh', 'left_shin', 'left_foot']
        num_obs = 1 + 15 * (3 + 6 + 3 + 3) - 3
        num_act = 36
        num_dof = 36
        num_bodies = 15
        dof_obs_size = 72
        dof_body_ids_gym = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
        # dof_body_ids_sim = [1, 4, 5, 9, 6, 10, 2, 7, 11, 3, 8, 12]
        # dof_body_indices_isaac_gym_to_sim = [0, 1, 9, 12, 2, 3, 6, 10, 13, 4, 7, 11, 14, 5, 8]
        dof_offsets_gym = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]
        dof_offsets_sim = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]
        dof_offset_indices_isaac_gym_to_sim = [0, 1, 2, 18, 19, 20, 27, 28, 29, 3, 4, 5, 6, 7, 8, 12, 13, 14,
                                               21, 22, 23, 30, 31, 32, 9, 10, 11, 15, 16, 17, 24, 25, 26, 33, 34, 35]

        contact_body_ids = torch.tensor([11, 14], device=device, dtype=torch.long)  # ["right_foot", "left_foot"]
        key_body_ids = torch.tensor([5, 8, 11, 14], device=device, dtype=torch.long)  # ["right_hand", "left_hand", "right_foot", "left_foot"]

    else:
        raise ValueError(f"Unsupported robot type: {robot_type}")

    return bodies_names, num_dof, num_bodies, humanoid_positions, dof_body_ids_gym, dof_offsets_gym, dof_offsets_sim, dof_offset_indices_isaac_gym_to_sim, dof_obs_size, num_obs, num_act, contact_body_ids, key_body_ids
