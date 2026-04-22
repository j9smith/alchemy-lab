import argparse
from pathlib import Path

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf

from alchemy.lab.training.checkpoints import CheckpointManager

class DecoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        return self.vae.decode(latent)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="output/onnx")
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    checkpoint = torch.load(Path(args.ckpt), map_location="cpu")
    cfg = OmegaConf.create(checkpoint["run_config"])

    model = instantiate(cfg.model).to(device)
    ema = None
    if args.use_ema and checkpoint["state"].get("ema") is not None:
        ema = instantiate(cfg.model).to(device)

    CheckpointManager.load(
        path=args.ckpt,
        model=model,
        ema=ema,
        optimiser=None,
        scheduler=None,
        strict_model=True,
        load_optimiser=False,
        load_scheduler=False,
        load_ema=args.use_ema,
    )

    denoiser = ema if (args.use_ema and ema is not None) else model
    denoiser.eval()

    img_cfg = cfg.data.image
    vae = None
    if "vae" in cfg:
        vae = instantiate(cfg.vae).to(device)
        vae.eval()

    if vae is not None:
        with torch.no_grad():
            dummy = torch.zeros(1, img_cfg.channels, img_cfg.resolution, img_cfg.resolution, device=device)
            _, C, H, W = vae.encode(dummy).shape
    else:
        C, H, W = img_cfg.channels, img_cfg.resolution, img_cfg.resolution

    dummy_xt = torch.randn(1, C, H, W, device=device)
    dummy_t = torch.zeros(1, dtype=torch.long, device=device)

    torch.onnx.export(
        denoiser,
        (dummy_xt, dummy_t),
        str(out / "denoiser.onnx"),
        input_names=["xt", "t"],
        output_names=["output"],
        dynamic_axes={
            "xt": {0: "batch"},
            "t": {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"exported {out / 'denoiser.onnx'}")

    if vae is not None:
        dummy_latent = torch.randn(1, C, H, W, device=device)

        torch.onnx.export(
            DecoderWrapper(vae),
            dummy_latent,
            str(out / "decoder.onnx"),
            input_names=["latent"],
            output_names=["output"],
            dynamic_axes={
                "latent": {0: "batch"},
                "output": {0: "batch"},
            },
            opset_version=17,
        )
        print(f"exported {out / 'decoder.onnx'}")
    else:
        print("no VAE in config, skipping decoder export")


if __name__ == "__main__":
    main()