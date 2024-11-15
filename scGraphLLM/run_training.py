from models import LitScGraphLLM
from data import *
from pathlib import Path
import json 
import os
# from data import AracneGraphWithRanksDataset
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.nn.utils.rnn import pad_sequence
from utils import update_mconfig_from_dict
import argparse
import random
import string
import subprocess as sp
import time 
import pickle 
import wandb
import glob 
from config import *
from torch_geometric.loader import DataLoader

def generate_random_string(length):
    alphanumeric = string.ascii_letters + string.digits
    return ''.join(random.choice(alphanumeric) for i in range(length))

torch.set_float32_matmul_precision('medium') ## this sets the gpu precision for 32 bit ops, lower means less precision but faster 
# filesystem = os.environ["WHEREAMI"]
user = os.environ["USER"]
filesystem = f"/hpc/mydata/{user}"
## ^This makes it easier to switch between different machines;  WHEREAMI is set in the .bashrc file and is the location of where we store repos; 
## on manitou its /manitou/pmg/users/vss2134, exxmini its /data/vss2134, aws its /data and so on 

parser = argparse.ArgumentParser(
                    prog='Lighting+W&B model training' ,
                    description='This script is used to train model via pytorch lightning',
                    )
parser.add_argument('--config', type=str, help='path to model config file',required=True)
parser.add_argument('--version', type=str, help='run version', default=None)
parser.add_argument('--name', type=str, help='run name', default=None)
parser.add_argument('--mode', type=str, help=' valid modes: [train, resume, debug, predict]', default=None, required=True)
parser.add_argument('--ckpt-file', type=str, help='name of checkpoint file only, no paths', default=None)
parser.add_argument('--override-config', type=str, help='wandb sweep style cl args that will be parsed and will update config accordingly', default=None)

args = parser.parse_args()

def collate_fn(batch):
    data = { "orig_gene_id" : [], "orig_rank_indices" : [], "gene_mask" : [], "rank_mask" : [], "both_mask" : [], "dataset_name" : [] }

    # Make a dictionary of lists from the list of dictionaries
    for b in batch:
        for key in data.keys():
            data[key].append(b[key])

    # Pad these dictionaries of lists
    for key in data.keys():
        if key != "dataset_name":
            data[key] = pad_sequence(data[key], batch_first=True)

    return data

def main(args):
    ## config is now a dict
    mconfig = args.config
    print(mconfig)
    mode = args.mode
    name = args.name
    if (mode in {"train", "resume",}) and (name is None):
        raise ValueError("Must specify name for training or resume mode")
    elif mode == "predict":
        name = "predict"
        
    if mode == "debug":
        ### run a miminal debug run to make sure everything is working
        print("***debug***")
        mconfig.trainer_config.max_epochs=2
        mconfig.data_config.train["debug"] = True
        for i in range(len(mconfig.data_config.val)):
            mconfig.data_config.val[i]["debug"] = True
        mconfig.data_config.num_workers=1
        mconfig['wandb_project']='debug'
        name = "debug"
        mconfig['trainer_config']['devices']=1
        if "strategy" in mconfig.trainer_config:
            del mconfig['trainer_config']['strategy']
    ## some lite error handling
    if mode == "resume":
        ## error so we dont accidentally overwrite a run
        if args.version is None or args.ckpt_file is None:
            raise ValueError("Must specify version and ckpt file for resume or predict mode")
        
    version = args.version
    if version is None:
        ## this does not break for ddp processes 
        version = generate_random_string(8)
    ## setup output directory
    project = mconfig['wandb_project']
    repo_name = mconfig['repo_name']
    root_dir = f"{filesystem}/{repo_name}/model_out"
    run_dir = Path(f"{root_dir}/{project}/{version}")
    ## don't overwrite existing runs
    if run_dir.exists() and (mode not in {"resume"}):
        raise NotImplementedError(f"run_dir {str(run_dir)} already exists, bad input ")
    run_dir.mkdir(exist_ok=True, parents=True)

    mconfig["slurm_jobid"] = os.getenv("SLURM_JOBID")
    ## final version of non-python config, save to output directory
    with open(f"{str(run_dir)}/mconfig_used.json", 'wb+') as m_stream:
        pickle.dump(mconfig, m_stream)

    ### load data
    # this should be a LightingDataModule, but because we only have 1 train loader for now, keep it a regular dataloader
    print("loading data...")
    lit_data_module = LitDataModule(mconfig.data_config)
    train_dl = lit_data_module.train_dataloader()
    val_dl = lit_data_module.val_dataloader()

    def collate_fn(batch):
        data = { "orig_gene_id" : [], "orig_rank_indices" : [], "gene_mask" : [], "rank_mask" : [], "both_mask" : [], "dataset_name" : [] }
        
        # Make a dictionary of lists from the list of dictionaries
        for b in batch:
            for key in data.keys():
                data[key].append(b[key])

        # Pad these dictionaries of lists
        for key in data.keys():
            if key != "dataset_name":
                data[key] = pad_sequence(data[key], batch_first=True)

        return data

    transformer_data_module = GraphTransformerDataModule(mconfig.data_config, collate_fn=collate_fn)
    train_transformer_dl = transformer_data_module.train_dataloader()
    val_transformer_dl = transformer_data_module.val_dataloader()
    
    #transformer_data_module = LitDataModule(mconfig.data_config)
    #train_transformer_dl = transformer_data_module.train_dataloader()
    #val_transformer_dl = transformer_data_module.val_dataloader()
    
    print("data loaded")


    model_fn = mconfig.model
    ## write intermediates outputs to scratch space in /pmglocal
    outdir = Path(f"/pmglocal/{user}/model_out/")
    outdir.mkdir(exist_ok=True)
    trainer_conf = mconfig['trainer_config']
    copy_checkpoints = True
    
    
    ## proceed with setting up trainer 
    if mode in {"train", "resume", "debug"}:
        

        ### Set up PTL trainer callbacks
        callbacks = []
        ## do not checkpoint every epoch,  will save time 
        #callbacks.append(ModelCheckpoint(filename = "latest-{epoch}-{step}"))
        
        if "checkpoint_config" in trainer_conf:
            check_point_conf = trainer_conf['checkpoint_config']
            for cp_conf in check_point_conf:
                callbacks.append(ModelCheckpoint(**cp_conf))


        if "early_stopping" in trainer_conf:
            callbacks.append(EarlyStopping(**trainer_conf['early_stopping']))
        
        trainer_conf['callbacks'] = callbacks

        ### set up logger 
        wblg= WandbLogger(project = mconfig['wandb_project'],
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
    
    
    if (mode == "train") or (mode == "debug"):
        trainer = pl.Trainer(**trainer_conf, default_root_dir=str(outdir))
        litmodel = model_fn(mconfig)                                                                # HERE

        # trainer.fit(litmodel, train_dataloaders = train_dl, val_dataloaders = val_dl)
        trainer.fit(litmodel, train_dataloaders = train_transformer_dl, val_dataloaders = val_transformer_dl)
        # trainer.validate(model=litmodel, dataloaders=val_dl)

    elif mode == "resume":
        trainer = pl.Trainer(**trainer_conf, default_root_dir=str(outdir))
        ckpt = args.ckpt_file
        ckpt_file = f"{root_dir}/{project}/{version}/checkpoints/{ckpt}"
        litmodel = model_fn.load_from_checkpoint(ckpt_file, 
                    config = mconfig)
        ### ckpt_path is required to resume training 
        trainer.fit(litmodel, train_dataloaders = train_dl,ckpt_path = ckpt_file)
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