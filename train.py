import argparse
from omegaconf import OmegaConf

from diffusion.trainers import get_trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    hparams = OmegaConf.load(args.config)
    trainer = get_trainer(hparams)
    trainer.train()