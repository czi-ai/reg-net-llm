from models import LitScGraphLLM
from data import *
from pathlib import Path
import json 
import os
from data import AracneGraphWithRanksDataset
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.nn.utils.rnn import pad_sequence
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



def collate_fn(batch):
    data = { "orig_gene_id" : [], "orig_rank_indices" : [], "gene_mask" : [], 
            "rank_mask" : [], "both_mask" : [], "edge_index": [], "num_nodes" :[], 
            "perturb_regime": [], "dataset_name" : []}
    
    # Make a dictionary of lists from the list of dictionaries
    for b in batch:
        for key in data.keys():
            data[key].append(b[key])

    # Pad these dictionaries of lists
    for key in data.keys():
        if (key != "dataset_name") & (key != "edge_index") & (key != "num_nodes"):
            data[key] = pad_sequence(data[key], batch_first=True, padding_value=PAD_RANK_IDX)
    return data

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
    
    if mode == "debug":
        ### run a minimal debug run to make sure everything is working
        print("***debug***")
        mconfig.trainer_config.max_epochs=1
        mconfig.data_config.train["debug"] = True
        for i in range(len(mconfig.data_config.val)):
            mconfig.data_config.val[i]["debug"] = True
        mconfig.data_config.num_workers=1
        mconfig['wandb_project']='debug'
        name = "debug"
        mconfig['trainer_config']['devices']=1
        if "strategy" in mconfig.trainer_config:
            del mconfig['trainer_config']['strategy']
    elif mode == "train":
        name = f"{name}_fine_tuned"
    
    if version is None:
        timestamp = time.time()
        current_time_UTC = time.strftime("%Y-%m-%d@%H:%M:%S", time.gmtime(timestamp))
        version = f"{name}:{current_time_UTC}"
    
    project = mconfig['wandb_project']
    repo_name = mconfig['repo_name']
    root_dir = f"{filesystem}/GLM/model_fine_tune_out"
    run_dir = Path(f"{root_dir}/{version}")
    ## don't overwrite existing runs
    if run_dir.exists() and (mode not in {"resume", "validate"}):
        raise NotImplementedError(f"run_dir {str(run_dir)} already exists, bad input ")
    run_dir.mkdir(exist_ok=True, parents=True)
    
    
    mconfig["slurm_jobid"] = os.getenv("SLURM_JOBID")
    with open(f"{str(run_dir)}/mconfig_used.json", 'wb+') as m_stream:
        pickle.dump(mconfig, m_stream)
    
    print("loading fine-tune data...")

    transformer_data_module = GraphTransformerDataModule(mconfig.data_config, collate_fn=collate_fn)
    train_transformer_dl = transformer_data_module.train_dataloader()
    val_transformer_dl = transformer_data_module.val_dataloader()
    print("data loaded")
    
    model_fn = mconfig.model
    outdir = Path(f"/hpc/mydata/{user}")
    outdir.mkdir(exist_ok=True)
    trainer_conf = mconfig['trainer_config']
    copy_checkpoints = True
    
    if mode == 'train':
        callbacks = []
        if "checkpoint_config" in trainer_conf:
            check_point_conf = trainer_conf['checkpoint_config']
            check_point_conf["dirpath"] = f"{run_dir}/checkpoints/"
            check_point_conf["filename"] = f"{{epoch}}-{{step}}"
            callbacks.append(ModelCheckpoint(**check_point_conf)) 
        
        if "early_stopping" in trainer_conf:
            callbacks.append(EarlyStopping(**trainer_conf['early_stopping']))
        
        trainer_conf['callbacks'] = callbacks
        
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
        trainer = pl.Trainer(**trainer_conf, default_root_dir=str(outdir))
        pre_trained_path = args.ckpt_path
        
        # load pre-trained weights
        checkpoint = torch.load(pre_trained_path, weights_only=False)["state_dict"]
        fine_tune_model = model_fn.load_from_checkpoint(checkpoint, strict=False)
        
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
    main(args)
    
    