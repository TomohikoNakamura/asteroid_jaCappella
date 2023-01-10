import argparse
import copy
import itertools
import json
import os
import random
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.utils.data
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from asteroid.metrics import get_metrics
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from local import dataloader
from mrdla import MRDLA, crop
from losses import CombinedLoss

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs.
parser = argparse.ArgumentParser()


class MRDLAManager(System):
    default_monitor = "val_loss"
    def __init__(
        self,
        model,
        optimizer,
        loss_func,
        train_loader,
        val_loader=None,
        scheduler=None,
        config=None
    ):
        self.sources = copy.deepcopy(config["data"]["sources"])
        config["data"].pop("sources")
        config["data"].pop("source_augmentations")
        super().__init__(model, optimizer, loss_func, train_loader, val_loader, scheduler, config)

    def common_step(self, batch, batch_nb, train=True):
        inputs, targets = batch
        est_targets = self(inputs)
        if train:
            if self.model.context:
                cropped_targets = crop(targets, est_targets)
                loss = self.loss_func(est_targets, cropped_targets)
            else:
                loss = self.loss_func(est_targets, targets)
        else:
            raise NotImplementedError
        return loss

    def validation_step(self, batch, batch_nb, dataloader_idx):
        tag = "val"
        _, targets = batch[0], batch[1] # targets: batch x sources x channel x time
        inputs = targets.sum(dim=1) # batch x sources x channels
        est_targets = self(inputs) # sources, batch_size, channels, time_length
        loss = self.loss_func(est_targets, targets) 
        self.log(f"{tag}_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        #
        time_hat = est_targets # source x batch x channel x time
        inputs = inputs.reshape(-1, inputs.shape[-1])
        targets = targets.reshape(targets.shape[1], -1)
        time_hat = time_hat.reshape(time_hat.shape[1], -1)
        metrics = get_metrics(inputs.cpu().numpy(), targets.cpu().numpy(), time_hat.cpu().numpy(), sample_rate=self.config["data"]["sample_rate"], metrics_list=["si_sdr"], average=False)
        val_sisdr = metrics["si_sdr"].reshape(-1) - metrics["input_si_sdr"].reshape(-1)
        self.log(f"{tag}_sisdr/average", float(val_sisdr.mean()), on_epoch=True, prog_bar=True, sync_dist=True)
        for i, source in enumerate(self.sources):
            self.log(f"{tag}_sisdr/{source}", float(val_sisdr[i]), on_epoch=True, prog_bar=False, sync_dist=True)

def main(conf: Dict, args):
    # Set seed for random
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    os.environ["OMP_NUM_THREADS"] = f"{int(args.num_workers/4*3)}"
    os.environ["MKL_NUM_THREADS"] = f"{int(args.num_workers/4*3)}"

    # create output dir if not exist
    exp_dir = Path(args.output)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load Datasets
    train_dataset, valid_dataset = dataloader.load_datasets(parser, args)
    dataloader_kwargs = (
        {"num_workers": args.num_workers, "pin_memory": True} if torch.cuda.is_available() else {}
    )
    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs
    )
    dataloader_kwargs["num_workers"] = 1
    valid_sampler = torch.utils.data.DataLoader(valid_dataset, batch_size=1, **dataloader_kwargs)

    # Define model and optimizer
    model = MRDLA(n_srcs=len(conf["data"]["sources"]), sample_rate=conf["data"]["sample_rate"], **conf["model"])
    optimizer = make_optimizer(
        model.parameters(), lr=args.lr, optimizer="adam", weight_decay=args.weight_decay
    )

    # Define scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.lr_decay_gamma, patience=args.lr_decay_patience, cooldown=10
    )

    # Save config
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    es = EarlyStopping(monitor="val_loss", mode="min", patience=args.patience, verbose=True)

    # Define Loss function.
    loss_func = CombinedLoss(lambda_f=conf["loss_func"].get("lambda_f", 1.0), lambda_t=conf["loss_func"].get("lambda_t", 10.0), band=conf["loss_func"].get("band", "high"))
        
    system = MRDLAManager(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_sampler,
        val_loader=valid_sampler,
        config=conf,
    )
    
    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True,
        every_n_epochs=1, every_n_train_steps=None, train_time_interval=None,
        save_last=True
    )
    callbacks.append(checkpoint)
    callbacks.append(es)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp",
        devices="auto",
        limit_train_batches=1.0,  # Useful for fast experiment
        check_val_every_n_epoch=1,
        val_check_interval=None
    )
    if args.ckpt_path is None:
        trainer.fit(system)
    else:
        trainer.fit(system, ckpt_path=args.ckpt_path)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_dataset.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))

if __name__ == "__main__":
    import yaml
    from asteroid.utils import parse_args_as_dict, prepare_parser_from_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("local/conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)

    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    main(arg_dic, plain_args)
