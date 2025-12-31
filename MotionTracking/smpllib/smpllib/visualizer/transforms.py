import torch
from torch_transform import quat_apply, normalize
from torch_utils import interp_tensor_with_scipy
from conversion import joints_to_smpl, recover_from_ric


def transform_global_motion_for_vis(global_pos_z_up, device, global_motion):
    if not global_pos_z_up:
        base_rot = torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=device, dtype=global_motion.dtype)
        g_joints = global_motion[:, :, 0].transpose(1, 2).view(global_motion.shape[0], global_motion.shape[-1], -1, 3)
        global_motion = quat_apply(base_rot.expand(g_joints.shape[:-1] + (4,)), g_joints)
        global_motion = global_motion.view(global_motion.shape[:-2] + (-1,)).transpose(1, 2).unsqueeze(2)
    return global_motion

def obtain_joints_and_smpl_pose(self, samples, smpl, infer_kwargs=None, interp=True, return_contacts=False):
    samples = self.obtain_full263_motion(samples)
    if self.motion_rep == 'global_position':
        smpl_pose = smpl_trans = None
        joints_pos = samples.squeeze(2).permute(0, 2, 1)  # needs interpolation
        if interp:
            joints_pos = interp_tensor_with_scipy(joints_pos, scale=1.5, dim=1)
        joints_pos = joints_pos.reshape(joints_pos.shape[:-1] + (-1, 3))
    elif self.motion_rep == 'global_root_local_joints':
        smpl_pose = smpl_trans = None
        base_rot = torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=self.device, dtype=samples.dtype)
        samples = samples.permute(0, 2, 3, 1)  # now permulted to: [batch, 1, seq_len, nfeat]
        root_samp = samples[..., :5]
        if self.normalize_global_pos:
            root_samp = root_samp * self.humanml_global_std.to(self.device) + self.humanml_global_mean.to(
                self.device)
        r_pos, rot_cos_sin, local_feats = root_samp[..., :3], root_samp[..., 3:5], samples[..., 5:]
        rot_cos_sin = normalize(rot_cos_sin)
        r_rot_quat = torch.cat(
            [rot_cos_sin[..., [0]], torch.zeros_like(rot_cos_sin[..., [0]]), rot_cos_sin[..., [1]],
             torch.zeros_like(rot_cos_sin[..., [0]])], dim=-1)

        local_feats_norm = local_feats * self.humanml_std[4:].to(self.device) + self.humanml_mean[4:].to(
            self.device)  # [batch, 1, seq_len, nfeat]
        local_feats_norm_pad = torch.cat([torch.zeros_like(local_feats_norm[..., :4]), local_feats_norm], dim=-1)

        joints_pos = recover_from_ric(local_feats_norm_pad, 22, r_rot_quat, r_pos)[:, 0]
        hand_joints = joints_pos[..., -2:, :].clone()
        hand_joints[..., 1] += 1e-2
        joints_pos = torch.cat([joints_pos, hand_joints], dim=-2)
        if interp:
            joints_pos = interp_tensor_with_scipy(joints_pos, scale=1.5, dim=1)
        smpl_pose, smpl_trans = joints_to_smpl(joints_pos, smpl, self.device)
        joints_pos = quat_apply(base_rot.expand(joints_pos.shape[:-1] + (4,)), joints_pos)
    elif self.motion_rep == 'global_root_vel_local_joints':
        smpl_pose = smpl_trans = None
        base_rot = torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=self.device, dtype=samples.dtype)
        samples = samples.permute(0, 2, 3, 1)
        root_samp = samples[..., :4]
        if self.normalize_global_pos:
            root_samp = root_samp * self.humanml_global_std.to(self.device) + self.humanml_global_mean.to(
                self.device)
        pos_v_xy, pos_y, ang_v, local_feats = root_samp[..., :2], root_samp[..., [2]], root_samp[..., [3]], samples[
                                                                                                            ..., 4:]

        r_rot_ang = torch.zeros_like(ang_v).to(self.device)
        r_rot_ang[:, :, 1:] = ang_v[:, :, :-1]
        r_rot_ang = r_rot_ang.cumsum(dim=2)
        r_rot_quat = torch.cat(
            [torch.cos(r_rot_ang), torch.zeros_like(r_rot_ang), torch.sin(r_rot_ang), torch.zeros_like(r_rot_ang)],
            dim=-1)

        r_pos = torch.zeros(pos_v_xy.shape[:-1] + (3,)).to(self.device)
        r_pos[..., 1:, [0, 2]] = pos_v_xy[..., :-1, :]
        r_pos = torch.cumsum(r_pos, dim=2)
        r_pos[..., [1]] = pos_y

        local_feats_norm = local_feats * self.humanml_std[4:].to(self.device) + self.humanml_mean[4:].to(
            self.device)  # [batch, 1, seq_len, nfeat]
        local_feats_norm_pad = torch.cat([torch.zeros_like(local_feats_norm[..., :4]), local_feats_norm], dim=-1)

        joints_pos = recover_from_ric(local_feats_norm_pad, 22, r_rot_quat, r_pos)[:, 0]
        hand_joints = joints_pos[..., -2:, :].clone()
        hand_joints[..., 1] += 1e-2
        joints_pos = torch.cat([joints_pos, hand_joints], dim=-2)
        joints_pos = quat_apply(base_rot.expand(joints_pos.shape[:-1] + (4,)), joints_pos)
        if interp:
            joints_pos = interp_tensor_with_scipy(joints_pos, scale=1.5, dim=1)
    elif self.motion_rep == 'global_root':
        body_model = self.ext_models['body_model']
        if body_model.device != self.device:
            body_model = body_model.to(self.device)
        root_motion_local = self.convert_root_global_to_local(samples)
        observed_motion = torch.zeros((samples.shape[0], body_model.motion_rep_dim, *samples.shape[2:]),
                                      device=samples.device)
        observed_motion[:, :root_motion_local.shape[1]] = root_motion_local
        res = body_model.generate_motion_mask(body_model.model_cfg.motion_mask, observed_motion, samples.shape[-1],
                                              use_mask_type='root_traj')
        res['rm_text_flag'][:] = 0.0
        body_infer_kwargs = infer_kwargs.copy()
        for key in ['motion_mask', 'observed_motion', 'rm_text_flag']:
            body_infer_kwargs[key] = res[key]
        body_samples = body_model.infer_texts(**body_infer_kwargs)
        joints_pos, smpl_pose, smpl_trans = body_model.obtain_joints_and_smpl_pose(body_samples, smpl,
                                                                                   infer_kwargs=body_infer_kwargs,
                                                                                   interp=interp)
    else:
        smpl_pose, smpl_trans, joints_pos = humanml_to_smpl(samples, mean=self.humanml_mean.to(self.device),
                                                            std=self.humanml_std.to(self.device), smpl=smpl)

    if return_contacts and self.motion_rep in {'global_root_local_joints', 'full263'}:
        print(samples.size())
        if not self.motion_rep in {'global_position', 'global_root_local_joints'}:
            samples = samples.permute(0, 2, 3, 1)  # now permulted to: [batch, 1, seq_len, nfeat]
        foot_contacts = samples[:, 0, :,
                        -4:]  # [batch, seq_len, 4] where 4 is ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe")
        print(foot_contacts.size())
        # should be between 0 and 1 unnormalized
        foot_contacts_norm = foot_contacts * self.humanml_std[-4:].to(samples.device) + self.humanml_mean[-4:].to(
            samples.device)
        foot_contacts_norm = interp_tensor_with_scipy(foot_contacts_norm, scale=1.5, dim=1)
        contacts = foot_contacts_norm > 0.5
    else:
        contacts = None

    if return_contacts:
        return joints_pos, smpl_pose, smpl_trans, contacts
    else:
        return joints_pos, smpl_pose, smpl_trans
