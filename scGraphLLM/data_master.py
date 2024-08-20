from data import *
from config import *
import argparse

print("Starting...")

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

if __name__ == "__main__":
    ## config is now a dict
    mconfig_str = args.config
    mconfig = eval(mconfig_str)

    transformer_data_module = TransformerDataModule(mconfig.data_config)
    train_dl = transformer_data_module.train_dataloader()

    batch = next(iter(train_dl))
    example_data = batch[0]
    print()
    print(example_data)
    print()