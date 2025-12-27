python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-fast-sac \
    --training.num-envs 2048 \
    logger:wandb \
    --logger.project locomotion-g1-29dof-ppo \
    --logger.video.interval 1000