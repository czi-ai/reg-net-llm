from models import LitScGraphLLM
from data import *
from pathlib import Path
import json 
import os
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
from wandb_checkpoint import SaveModelEveryNSteps

torch.set_float32_matmul_precision('medium')
user = os.environ["USER"]
filesystem = f"/hpc/mydata/{user}"

parser = argparse.ArgumentParser(
                    prog='Lighting+W&B perturbation fine-tuning' ,
                    description='This script is used to fine-tune embeddings on perturbation data',
                    )
parser.add_argument('--fine-tune-config', type=str, help='path to model config file used in fine-tuning',required=True)
parser.add_argument('--version', type=str, help='run version', default=None)
parser.add_argument('--name', type=str, help='run name', default=None)
parser.add_argument('--mode', type=str, help=' valid modes: [train, debug, predict]', default=None, required=True)
parser.add_argument('--ckpt-path', type=str, help='path to pre-trained model weights', default=None, required=True)

args = parser.parse_args()

def main(args):
    mconfig = args.fine_tune_config
    mode = args.mode
    name = args.name
    version = args.version
    
   
    if mode == "train":
        name = f"{name}_fine_tuned"
    
    if version is None:
        timestamp = time.time()
        current_time_UTC = time.strftime("%Y-%m-%d@%H:%M:%S", time.gmtime(timestamp))
        version = f"{name}:{current_time_UTC}"
    
    project = "perturb_fine_tune_scGraphLM"
    #repo_name = mconfig['repo_name']
    root_dir = f"{filesystem}/GLM/model_fine_tune_out"
    run_dir = Path(f"{root_dir}/{version}")
    run_dir.mkdir(exist_ok=True, parents=True)
    
    print("loading fine-tune data...")

    transformer_data_module = PerturbationDataModule(mconfig.data_config)
    train_transformer_dl = transformer_data_module.train_dataloader()
    val_transformer_dl = transformer_data_module.val_dataloader()
    print("data loaded")
    
    model_fn = mconfig.model
    outdir = Path(f"/hpc/mydata/{user}")
    outdir.mkdir(exist_ok=True)
    trainer_conf = mconfig['trainer_config']
    copy_checkpoints = True
    
    if mode == 'train':
        
        wblg = WandbLogger(project = mconfig['wandb_project'],
                    name = name,
                    entity = mconfig['wandb_user'],
                    version = version,
                    id = version,
                    config=mconfig,
                    save_dir = str(outdir))     
        trainer_conf['logger'] = wblg
    
    wandb.init(project=mconfig['wandb_project'], name=name)
    
    if mode == 'train':
        if "checkpoint_config" in trainer_conf:
            del trainer_conf["checkpoint_config"]
        trainer = pl.Trainer(**trainer_conf, default_root_dir=str(outdir))
        pre_trained_path = args.ckpt_path
        print("loading pre-trained weights...")
        #checkpoint = torch.load(pre_trained_path, weights_only=False)["state_dict"]
        print(model_fn)
        fine_tune_model = model_fn.load_from_checkpoint(pre_trained_path, config=graph_kernel_attn_3L_4096, strict=False)
        print("pre-trained weights loaded")
        
        # fine-tune with separate trainer
        trainer.fit(fine_tune_model, train_dataloaders = train_transformer_dl, 
                    val_dataloaders=val_transformer_dl)
    
    if ((mconfig['trainer_config']['devices'] == 1) or (torch.distributed.get_rank() == 0)) and (mode != "debug") :
        if copy_checkpoints:
            time.sleep(30)
            local_ckptdir = f"{str(outdir)}/{project}/{version}/checkpoints"
            print("copying checkpoints to shared dir")
            shared_ckptdir = f"{str(run_dir)}/checkpoints"
            sp.run(f"mkdir -p {shared_ckptdir}", shell=True)
            sp.run(f"cp {local_ckptdir}/* {shared_ckptdir}/", shell=True)
            sp.run(f"rm -rf {local_ckptdir}", shell=True)
            print("Fine Tuning Completed Succsessfully.")
        
if __name__ == "__main__":
    mconfig_str = args.fine_tune_config
    mconfig = eval(mconfig_str)
    args.fine_tune_config = mconfig
    main(args)