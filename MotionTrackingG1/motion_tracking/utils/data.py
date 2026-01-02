from motion_tracking.utils.motion_lib import MotionLib
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from hydra.utils import instantiate


class CountingDataset(Dataset):
    def __init__(self, config):
        self.config = config
        # assert config.num_envs * config.num_steps % config.batch_size == 0
        # self.length = (
        #     config.ngpu * config.num_envs * config.num_steps // config.batch_size
        # )
        self.length = config.num_batches * config.ngpu

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return index


class CountingDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ds = CountingDataset(config)

    def train_dataloader(self):
        return DataLoader(self.ds, batch_size=1, shuffle=False)


class ReferenceMotions(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.motion_lib: MotionLib = instantiate(config.motion_lib)

    def setup(self, stage=None):
        pass


MOTION_LIB: MotionLib = None


def global_motion_lib_wrapper(config):
    global MOTION_LIB
    if MOTION_LIB is None:
        MOTION_LIB = MotionLib(**config)
    return MOTION_LIB
