import os
from typing import Any, Type

from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only

try:
    import sys

    sys.path.append(os.environ.get("SUBMIT_SCRIPTS", "."))
    from userlib.auto_resume import AutoResume
except ModuleNotFoundError:
    AutoResume = None


class AdlrAutoresume(Callback):
    """
    Handles ADLR `submit_job` style timed autoresume on the slurm clusters. The callback checks for the timeout condition after every batch.

    It leverages the existing hpc checkpointing infrastructure in pytorch lightning. When the autoresume timer expires it will automatically
    save a checkpoint and request autoresume. When your run is restarted, lightning will pick up this checkpoint and automatically load it for you.

    This callback can be safely loaded in an environment without the ADLR libraries; it will do nothing.

    Usage:
        ```
        model = ...
        data = ...

        trainer = pl.Trainer(..., callbacks=[AdlrAutoresume()])
        trainer.fit(model, data)
        ```

        or to resume a logger by id

        ```
        autoresume = AdlrAutoresume()
        logger = WandbLogger(id=autoresume.details.get("id"))
        model = ...
        data = ...

        trainer = pl.Trainer(..., logger=logger, callbacks=[autoresume])
        trainer.fit(model, data)
        ```
    """

    def __new__(cls: Type["AdlrAutoresume"], *args, **kwargs) -> Callback:
        if AutoResume is not None:
            return super().__new__(cls, *args, **kwargs)
        else:
            blank = super().__new__(Callback)
            blank.details = {}
            return blank

    def __init__(self) -> None:
        AutoResume.init()
        self.details = AutoResume.get_resume_details() or {
            "id": os.environ.get("SLURM_JOB_ID")
        }
        self._autoresume_sent = False

    @rank_zero_only
    def _request_autoresume(self, trainer: Trainer, pl_module: LightningModule) -> None:
        hpc_save_path = trainer._checkpoint_connector.hpc_save_path(
            trainer.default_root_dir
        )
        trainer.save_checkpoint(hpc_save_path)
        # TODO: save env info and reload it after autoresume
        self.details["checkpoint"] = hpc_save_path

        AutoResume.request_resume(user_dict=self.details)

    @property
    def terminate(self) -> bool:
        return AutoResume.termination_requested()

    def _check_autoresume(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.terminate and not self._autoresume_sent:
            self._request_autoresume(trainer, pl_module)
            trainer.should_stop = True
            self._autoresume_sent = True

    # def on_train_batch_end(
    #         self,
    #         trainer: Trainer,
    #         pl_module: LightningModule,
    #         outputs: Any,
    #         batch: Any,
    #         batch_idx: int,
    # ) -> None:
    #     self._check_autoresume(trainer, pl_module)

    def on_train_epoch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule
    ) -> None:
        self._check_autoresume(trainer, pl_module)

    # def on_validation_batch_end(
    #         self,
    #         trainer: Trainer,
    #         pl_module: LightningModule,
    #         outputs: Any,
    #         batch: Any,
    #         batch_idx: int,
    #         dataloader_idx: int = 0,
    # ) -> None:
    #     self._check_autoresume(trainer, pl_module)
    #
    # def on_test_batch_end(
    #         self,
    #         trainer: Trainer,
    #         pl_module: LightningModule,
    #         outputs: Any,
    #         batch: Any,
    #         batch_idx: int,
    #         dataloader_idx: int = 0,
    # ) -> None:
    #     self._check_autoresume(trainer, pl_module)
    #
    # def on_predict_batch_end(
    #         self,
    #         trainer: Trainer,
    #         pl_module: LightningModule,
    #         outputs: Any,
    #         batch: Any,
    #         batch_idx: int,
    #         dataloader_idx: int = 0,
    # ) -> None:
    #     self._check_autoresume(trainer, pl_module)
