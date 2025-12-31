import torch


class BaseInterface(object):
    def __init__(
            self,
            config,
            device: torch.device,
    ):
        self.config = config
        self.device = device
        self.headless = config.headless

        # double check!
        self.graphics_device_id = self.device.index
        if self.headless is True:
            self.graphics_device_id = -1

        self.num_envs = config.num_envs

        self.control_freq_inv = config.control_freq_inv

    def get_obs_size(self):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def pre_physics_step(self, actions):
        raise NotImplementedError

    def reset(self, env_ids=None):
        raise NotImplementedError

    def physics_step(self):
        for i in range(self.control_freq_inv):
            self.control_i = i
            self.simulate()
        return

    def simulate(self):
        raise NotImplementedError

    def post_physics_step(self):
        raise NotImplementedError

    def on_epoch_end(self, current_epoch: int):
        pass

    def close(self):
        raise NotImplementedError
