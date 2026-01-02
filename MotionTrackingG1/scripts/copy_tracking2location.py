import torch
#scene_1203_parkour_line12_seed130_noise1_pretrain
#scene_1209_parkour_line12_seed130_noise0_pretrain
# scene_1209_parkour_line12_seed130_noise1_noscene_disc_pretrain
# scene_1209_parkour2norm_line12_seed130_noise1_pretrain
parkour_pretrain_file = "results/chair_scene_ours_0414_line1_fix_seed210_noise1_2gpu_pretrain/lightning_logs/version_0/last.ckpt"
tracking_policy_file = "results/chair_tracking_0414_finetune_0409_line1_seed210_noise1_2gpu/lightning_logs/version_0/last.ckpt"
# tracking_policy_file = "results/tracking_1208_parkour_line12_seed130_noise0_4gpu/lightning_logs/version_0/last.ckpt"
data1 = torch.load(parkour_pretrain_file)
data2 = torch.load(tracking_policy_file)


# Get the state dictionaries from each data
state_dict1 = data1['state_dict']
state_dict2 = data2['state_dict']

# import pdb;pdb.set_trace()
actor_keys = [key for key in state_dict2.keys() if 'actor' in key]
# Copy missing keys from data2 to data1
for key in state_dict1:
    if key in actor_keys:
        print(key)
        state_dict1[key] = state_dict2[key]
# # You might want to save the updated state_dict back to data1's checkpoint or to a new one
# # Example: Saving to a new checkpoint
updated_checkpoint_path = parkour_pretrain_file
data1['state_dict'] = state_dict1
torch.save(data1, updated_checkpoint_path)

