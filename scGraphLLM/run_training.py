from pathlib import Path
import json 
import os
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

import argparse
import random
import string
import subprocess as sp
import time 
import pickle 
import wandb
import glob 
from torch.utils.data import DataLoader as torchDataLoader

# import torchsummary

from scGraphLLM.models import LitScGraphLLM
from scGraphLLM.data import GraphTransformerDataset
from scGraphLLM._globals import *
from scGraphLLM.config import *
from scGraphLLM.vocab import GeneVocab

def generate_random_string(length):
    alphanumeric = string.ascii_letters + string.digits
    return ''.join(random.choice(alphanumeric) for i in range(length))

def update_mconfig_from_dict(mconfig, sweep_dict, ignore_keys={}):
    sweep_keys = [k for k in sweep_dict.keys() if k not in ignore_keys ]
    for skey in sweep_keys:
        key_path = skey.split("-")
        c_dict = mconfig
        for _key in key_path[:-1]:
            c_dict = c_dict[_key]
        ## preserve original datatype of parameter
        orig_dtype = type(c_dict[key_path[-1]])
        c_dict[key_path[-1]]=orig_dtype(sweep_dict[skey])
    return mconfig

torch.set_float32_matmul_precision('medium') ## this sets the gpu precision for 32 bit ops, lower means less precision but faster 

user = os.environ["USER"]

parser = argparse.ArgumentParser(
                    prog='Lighting+W&B model training' ,
                    description='This script is used to train model via pytorch lightning',
                    )
parser.add_argument('--config', type=str, help='path to model config file',required=True)
parser.add_argument("--gene-to-node-file", type=str, help="File containing gene to node index mapping")
parser.add_argument('--version', type=str, help='run version', default=None)
parser.add_argument('--name', type=str, help='run name', default=None)
parser.add_argument('--mode', type=str, help=' valid modes: [train, resume, predict, validate]', default=None, required=True)
parser.add_argument('--ckpt-file', type=str, help='name of checkpoint file only, no paths', default=None)
parser.add_argument('--override-config', type=str, help='wandb sweep style cl args that will be parsed and will update config accordingly', default=None)
parser.add_argument("--output-dir", type=str, help="output directory for model checkpoints and logs", default=None)

args = parser.parse_args()


class GraphTransformerDataModule(pl.LightningDataModule):
    def __init__(self, data_config, vocab: GeneVocab):
        super().__init__()
        self.data_config = data_config
        self.train_ds = GraphTransformerDataset(vocab=vocab, **data_config.train)
        self.val_ds = [GraphTransformerDataset(vocab=vocab, **val) for val in data_config.val]
        if data_config.run_test:
            self.test_ds = [GraphTransformerDataset(vocab=vocab, **test) for test in data_config.test]
    
    def train_dataloader(self):
        return torchDataLoader(
            dataset=self.train_ds, 
            batch_size=self.data_config.batch_size, 
            num_workers=self.data_config.num_workers, 
            collate_fn=self.train_ds.collate_fn
        )
    
    def val_dataloader(self):
        return [
            torchDataLoader(
                dataset=val_ds, 
                batch_size=self.data_config.batch_size, 
                num_workers=self.data_config.num_workers, 
                collate_fn=val_ds.collate_fn) 
            for val_ds in self.val_ds
        ]
    
    def test_dataloader(self):
        return [
            torchDataLoader(
                dataset=test_ds, 
                batch_size=self.data_config.batch_size, 
                num_workers=self.data_config.num_workers, 
                collate_fn=test_ds.collate_fn) 
            for test_ds in self.test_ds
        ]


def main(args):

    
    mconfig = args.config
    mode = args.mode
    name = args.name
    if (mode in {"train", "resume", "validate"}) and (name is None):
        raise ValueError("Must specify name for training or resume mode")
    elif mode == "predict":
        name = "predict"
        
    ## some lite error handling
    if mode in {"resume", "validate"}:
        ## error so we dont accidentally overwrite a run
        if args.version is None or args.ckpt_file is None:
            raise ValueError("Must specify version and ckpt file for resume, predict, or validate mode")
        
    version = args.version
    if version is None:
        ## this does not break for ddp processes 
        timestamp = time.time()
        current_time_UTC = time.strftime("%Y-%m-%d@%H:%M:%S", time.gmtime(timestamp))
        version = f"{name}:{current_time_UTC}"
        
    ## setup output directory
    project = mconfig['wandb_project']
    repo_name = mconfig['repo_name']
    root_dir = f"{args.output_dir}/model_out"
    run_dir = Path(f"{root_dir}/{version}")
    ## don't overwrite existing runs
    # if run_dir.exists() and (mode not in {"resume", "validate"}):
    #     raise NotImplementedError(f"run_dir {str(run_dir)} already exists, bad input ")
    run_dir.mkdir(exist_ok=True, parents=True)

    mconfig["slurm_jobid"] = os.getenv("SLURM_JOBID")
    ## final version of non-python config, save to output directory
    with open(f"{str(run_dir)}/mconfig_used.json", 'wb+') as m_stream:
        pickle.dump(mconfig, m_stream)

    # load gene vocabulary
    vocab = GeneVocab.from_csv(args.gene_to_node_file, gene_col="gene_name", node_col="idx")
    
    ### load data
    # this should be a LightingDataModule, but because we only have 1 train loader for now, keep it a regular dataloader
    print("loading data...")

    transformer_data_module = GraphTransformerDataModule(mconfig.data_config, vocab=vocab)
    train_transformer_dl = transformer_data_module.train_dataloader()
    val_transformer_dl = transformer_data_module.val_dataloader()
    
    print("data loaded")

    model_fn = mconfig.model
    ## write intermediates outputs to scratch space in /pmglocal
    outdir = Path(f"/hpc/mydata/{user}")
    outdir.mkdir(exist_ok=True)
    trainer_conf = mconfig['trainer_config']
    copy_checkpoints = True
    
    
    ## proceed with setting up trainer 
    if mode in {"train", "resume", "debug", "validate"}:
        
        ### Set up PTL trainer callbacks
        callbacks = []
        ## do not checkpoint every epoch,  will save time 
        #callbacks.append(ModelCheckpoint(filename = "latest-{epoch}-{step}"))
        
        if "checkpoint_config" in trainer_conf:
            check_point_conf = trainer_conf['checkpoint_config']
            check_point_conf["dirpath"] = f"{run_dir}/checkpoints/"
            check_point_conf["filename"] = f"{{epoch}}-{{step}}"
            callbacks.append(ModelCheckpoint(**check_point_conf)) # Locally save checkpoint
            

        if "early_stopping" in trainer_conf:
            callbacks.append(EarlyStopping(**trainer_conf['early_stopping']))
        
        trainer_conf['callbacks'] = callbacks

        ### set up logger 
        wblg = WandbLogger(project = mconfig['wandb_project'],
                    name = name,
                    entity = mconfig['wandb_user'],
                    version = version,
                    id = version,
                    config=mconfig,
                    save_dir = str(outdir))     
        trainer_conf['logger'] = wblg
    else:
        raise NotImplementedError(f"mode {mode} not implemented")
    if "early_stopping" in trainer_conf:
        del trainer_conf["early_stopping"]
    if "checkpoint_config" in trainer_conf:
        del trainer_conf["checkpoint_config"]
    
    wandb.init(project=mconfig['wandb_project'], name=name)
    if (mode == "train") or (mode == "debug"):
        trainer = pl.Trainer(**trainer_conf, default_root_dir=str(outdir))
        litmodel = model_fn(mconfig)
        trainer.fit(litmodel, train_dataloaders = train_transformer_dl, val_dataloaders = val_transformer_dl)

    elif mode == "resume":
        trainer = pl.Trainer(**trainer_conf, default_root_dir=str(outdir))
        ckpt = args.ckpt_file
        ckpt_file = f"{run_dir}/checkpoints/{ckpt}"
        litmodel = model_fn.load_from_checkpoint(ckpt_file, config = mconfig)
        ### ckpt_path is required to resume training 
        trainer.fit(litmodel, train_dataloaders = train_transformer_dl,ckpt_path = ckpt_file)
        
    elif mode == "validate":
        trainer = pl.Trainer(**trainer_conf, default_root_dir=str(outdir))
        ckpt = args.ckpt_file
        ckpt_file = f"{run_dir}/checkpoints/{ckpt}"
        litmodel = model_fn.load_from_checkpoint(ckpt_file, config = mconfig)
        trainer.validate(model=litmodel, dataloaders=val_transformer_dl)

    else:
        raise NotImplementedError(f"mode {mode} not implemented")

    ##  copy checkpoints to shared dir and clean up only on 1 gpu 

    if ((mconfig['trainer_config']['devices'] == 1) or (torch.distributed.get_rank() == 0)) and (mode != "debug") :
        if copy_checkpoints:
            time.sleep(30) ## potentially wait for other processes to finish writing to disk
            local_ckptdir = f"{str(outdir)}/{project}/{version}/checkpoints"
            print("copying checkpoints to shared dir")
            shared_ckptdir = f"{str(run_dir)}/checkpoints"
            sp.run(f"mkdir -p {shared_ckptdir}", shell=True)
            sp.run(f"cp {local_ckptdir}/* {shared_ckptdir}/", shell=True)
            sp.run(f"rm -rf {local_ckptdir}", shell=True)
            print("Run Completed Succsessfully.")

if __name__ == "__main__":
    mconfig_str = args.config
    mconfig = eval(mconfig_str)

    ## update config with extras from commandline
    override_config = args.override_config
    if override_config is not None:
        ## pass in commandline style args to override config. 
        override_args = override_config.split(",")
        params_to_update = {}
        for a in override_args:
            k,v = a.split("=")
            params_to_update[k]=v
        mconfig = update_mconfig_from_dict(mconfig, params_to_update, set(['mconfig', 'project']) )
    args.config = mconfig
    main(args)