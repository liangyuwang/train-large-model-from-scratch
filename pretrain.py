from __future__ import annotations

from training.arguments import parse_args
from training.config import build_config
from training.trainer import Trainer


def main():
    args = parse_args()
    cfg = build_config(args)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
