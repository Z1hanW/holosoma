## Single sequence motion retargeting
```
# Robot-only (OMOMO)
python examples/robot_retarget.py --data_path demo_data/OMOMO_new --task-type robot_only --task_name sub3_largebox_003 --data_format smplh --retargeter.debug --retargeter.visualize

# Object interaction (OMOMO)
python examples/robot_retarget.py --data_path demo_data/OMOMO_new --task-type object_interaction --task-name sub3_largebox_003 --data_format smplh --retargeter.debug --retargeter.visualize

# Climbing
python examples/robot_retarget.py --data_path demo_data/climb --task-type climbing --task-name mocap_climb_seq_0 --data_format mocap --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf --retargeter.debug --retargeter.visualize
```

Add --augmentation for running sequence with augmentation (need to first run the original sequence before adding augmentation)

## Batch processing for motion retargeting
```
# Robot-only (OMOMO)
python examples/parallel_robot_retarget.py --data-dir demo_data/OMOMO_new --task-type robot_only  --data_format smplh --save_dir demo_results_parallel/g1/robot_only/omomo --task-config.object-name ground

# Object interaction (OMOMO)
python examples/parallel_robot_retarget.py --data-dir demo_data/OMOMO_new --task-type object_interaction  --data_format smplh --save_dir demo_results_parallel/g1/object_interaction/omomo --task-config.object-name largebox

# Climbing
python examples/parallel_robot_retarget.py --data-dir demo_data/climb --task-type climbing --data_format mocap --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf --task-config.object-name multi_boxes --save_dir demo_results_parallel/g1/climbing/mocap_climb
```

Add --augmentation for running sequence with augmentation (for object interaction and climbing task)

## Data Preparation
We provide demo_data/ for fast testing. To test on more motion sequences in OMOMO, please follow the instructions to download the data.


### OMOMO
Our pipeline used the processed dataset by InterMimic. The data format differs from the original OMOMO dataset. Please download the processed OMOMO data here https://drive.google.com/file/d/141YoPOd2DlJ4jhU2cpZO5VU5GzV_lm5j/view. And put the downloaded folder to demo_data/OMOMO_new.

### LAFAN
#### Download the original LAFAN data.

Dowoloading the [lafan1.zip](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip) by clicking View Raw.

Put lafan1.zip to your designed data folder and uncompress it (DATA_FOLDER_PATH/lafan1).

We need some data processing files from the [LAFAN github repo](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).
```
git clone https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git
```

#### Convert the original LAFAN data format for motion retargeting.
Copy the folder lafan1/ from your downloaded LAFAN repo to data_utils/.

```
cd data_utils/
python extract_global_positions.py --input_dir DATA_FOLDER_PATH/lafan1 --output_dir ../demo_data/lafan
```

#### Single sequence retargeting on LAFAN.
```
python examples/robot_retarget.py --data_path demo_data/lafan --task-type robot_only --task_name dance2_subject1 --data_format lafan --task-config.ground-range -10 10 --save_dir demo_results/g1/robot_only/lafan --retargeter.debug --retargeter.visualize
```

#### Batch processing for motion retargeting on LAFAN.
```
python examples/parallel_robot_retarget.py --data-dir demo_data/lafan --task-type robot_only  --data_format lafan --save_dir demo_results_parallel/g1/robot_only/lafan --task-config.object-name ground --task-config.ground-range -10 10
```

## Check visualizations of saved retargeting results
```
# Visualize object-interaction results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --object_urdf models/largebox/largebox.urdf \
    --qpos_npz demo_results_parallel/g1/object_interaction/omomo/sub3_largebox_003_original.npz

python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --object_urdf models/largebox/largebox.urdf \
    --qpos_npz demo_results_parallel/g1/object_interaction/omomo/sub3_largebox_003_trans_1.npz

# Visualize climbing results
python viser_player.py --robot_urdf models/g1/g1_29dof_spherehand.urdf \
    --object_urdf demo_data/climb/mocap_climb_seq_0/multi_boxes.urdf \
    --qpos_npz demo_results_parallel/g1/climbing/mocap_climb/mocap_climb_seq_0_original.npz

python viser_player.py --robot_urdf models/g1/g1_29dof_spherehand.urdf \
    --object_urdf demo_data/climb/mocap_climb_seq_0/multi_boxes_scaled_0.74_0.74_0.89.urdf \
    --qpos_npz demo_results_parallel/g1/climbing/mocap_climb/mocap_climb_seq_0_z_scale_1.2.npz

# Visualize robot only results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz demo_results_parallel/g1/robot_only/omomo/sub3_largebox_003_original.npz

# Visualize LAFAN robot only results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz demo_results/g1/robot_only/lafan/dance2_subject1.npz
```

## Quantitative Evaluation
```
# Evaluate robot-object interaction
python evaluation/eval_retargeting.py --res_dir demo_results_parallel/g1/object_interaction/omomo --data_dir demo_data/OMOMO_new --data_type "robot_object"

# Evaluate climbing sequence
python evaluation/eval_retargeting.py --res_dir demo_results_parallel/g1/climbing/mocap_climb --data_dir demo_data/climb --data_type "robot_terrain" --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf

# Evaluate robot only (OMOMO)
python evaluation/eval_retargeting.py --res_dir demo_results_parallel/g1/robot_only/omomo --data_dir demo_data/OMOMO_new --data_type "robot_only"
```

## Prepare data for training RL whole-body tracking policy
Note that if you run this code on Mac, please use mjpython. For example,
```
mjpython data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/robot_only/omomo/sub3_largebox_003.npz --output_fps 50 --output_name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz --data_format smplh --object_name "ground" --once

mjpython data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz --output_fps 50 --output_name converted_res/object_interaction/sub3_largebox_003_mj_w_obj.npz --data_format smplh --object_name "largebox"  --has_dynamic_object --once
```

Robot-only setting
```
python data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/robot_only/omomo/sub3_largebox_003.npz --output_fps 50 --output_name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz --data_format smplh --object_name "ground" --once

python data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/robot_only/lafan/dance2_subject1.npz --output_fps 50 --output_name converted_res/robot_only/dance2_subject1_mj_fps50.npz  --data_format lafan --object_name "ground" --once
```

Robot-object setting
```
python data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz --output_fps 50 --output_name converted_res/object_interaction/sub3_largebox_003_mj_w_obj.npz --data_format smplh --object_name "largebox"  --has_dynamic_object --once
```

For OmniRetarget data downloaded from HuggingFace, please add --use_omniretarget_data for data conversion.
```
python data_conversion/convert_data_format_mj.py --input_file OmniRetarget/robot-object/sub3_largebox_003_original.npz --output_fps 50 --output_name converted_res/object_interaction/sub3_largebox_003_mj_w_obj_omnirt.npz --data_format smplh --object_name "largebox"  --has_dynamic_object --use_omniretarget_data --once
```
