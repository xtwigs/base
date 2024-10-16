import os
import gc
import logging
from typing import Sequence

from omegaconf import DictConfig
import hydra

import lightning as L
import torch

from src.ds.base import BaseDataset
from src.utils.instantiators import instantiate_callbacks, instantiate_loggers

logging.basicConfig(level=logging.INFO)


# def get_trainer(
#     config,
#     ckpt_path=None,
#     devices=None,
#     eval_freq=None,
#     max_steps=-1,
#     max_epochs=None,
#     accumulate_grad_batches=1,
#     log_every_n_steps=10,
#     gradient_clip_val=1.0,
#     run_name="",
#     precision="bf16-mixed",
#     accelerator="gpu",
#     dryrun=False,
#     **kwargs,
# ) -> L.Trainer:
#     class GradientNormCallback(L.Callback):
#         def __init__(self, log_every_n_steps=1):
#             super().__init__()
#             self.log_every_n_steps = log_every_n_steps

#         def on_after_backward(
#             self, trainer: L.Trainer, L_module: L.LightningModule
#         ) -> None:
#             if trainer.global_step % self.log_every_n_steps == 0:
#                 total_norm = 0
#                 for p in L_module.parameters():
#                     if p.grad is not None:
#                         param_norm = p.grad.data.norm(2)
#                         total_norm += param_norm.item() ** 2
#                 total_norm = total_norm**0.5
#                 trainer.logger.experiment.log(
#                     {"grad_norm": total_norm}, step=trainer.global_step
#                 )

#     gradient_norm_callback = GradientNormCallback(
#         log_every_n_steps=log_every_n_steps,
#     )

#     model_checkpoint = ModelCheckpoint(
#         monitor="val_bleu",
#         dirpath=ckpt_path,
#         filename="{epoch:02d}-{val_bleu:.2f}",
#         save_last=True,
#         save_top_k=2,
#         mode="max",
#     )

#     return L.Trainer(
#         devices=devices,
#         accelerator=accelerator,
#         max_steps=max_steps,
#         max_epochs=max_epochs,
#         val_check_interval=eval_freq,
#         log_every_n_steps=log_every_n_steps,
#         callbacks=[model_checkpoint, gradient_norm_callback],
#         precision=precision,
#         accumulate_grad_batches=accumulate_grad_batches,
#         gradient_clip_val=gradient_clip_val,
#         gradient_clip_algorithm="norm",
#     )


@hydra.main(version_base="1.3", config_path="../configs", config_name="run.yaml")
def main(config: DictConfig):
    dataset: BaseDataset = hydra.utils.instantiate(config.ds)
    tokenizer = dataset.get_tokenizer()
    model: L.LightningModule = hydra.utils.instantiate(config.model)
    tokenizer.padding_side = "right" if model.model_name == "mamba2" else "left"
    model.tokenizer = tokenizer

    train_dl, val_dl, _ = dataset.get_dataloaders(
        tokenizer, input_padding_side=tokenizer.padding_side, test=False
    )

    logger: L.pytorch.loggers.WandbLogger = instantiate_loggers(config.logger)
    callbacks: Sequence[L.Callback] = instantiate_callbacks(config.callbacks)

    trainer: L.Trainer = hydra.utils.instantiate(
        config.trainer, logger=logger, callbacks=callbacks
    )
    logging.info(f"current config: \n{config}")

    resume_training_path = config.experiment.resume_path

    trainer.fit(model, train_dl, val_dl, ckpt_path=resume_training_path)

    max_retries = 5
    while max_retries > 0:
        try:
            trainer.fit(model, train_dl, val_dl, ckpt_path=resume_training_path)
            break
        except torch.cuda.OutOfMemoryError:
            logging.info("OOM, retrying...")
            torch.cuda.empty_cache()
            gc.collect()
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.error(e)
            max_retries -= 1
            resume_training_path = "last"
        finally:
            trainer.save_checkpoint(
                os.path.join(
                    config.paths.output_dir, f"final-{trainer.logger.version}.ckpt"
                )
            )


if __name__ == "__main__":
    main()
