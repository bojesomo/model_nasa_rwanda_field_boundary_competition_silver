import gc
import inspect
import shutil
import torch
import pytorch_lightning as pl


def remove_dir(path):
    try:
        shutil.rmtree(path)
    except:
        pass


def free_memory(to_delete: list):
    calling_namespace = inspect.currentframe().f_back

    for _var in to_delete:
        calling_namespace.f_locals.pop(_var, None)
        gc.collect()
        torch.cuda.empty_cache()


def get_callbacks(fold, patience=10):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
          # filename=f"fold_{fold}",
          # monitor='val_loss',
          # monitor="val_f1",
          verbose=False,
          save_last=True,
          # save_top_k=1,
          # mode='min',
          # mode='max',
          save_weights_only=True
    )
    return [
      checkpoint_callback,
    ]
