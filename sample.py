# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse

from diffusers.models import AutoencoderKL
from torchvision.utils import save_image
from tqdm import tqdm

from diffusion import create_diffusion
from download import find_model
from models import DiT_models


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size, num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt
    # state_dict = find_model(ckpt_path)
    # model.load_state_dict(state_dict)
    model.eval()  # important!
    print(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    diffusion = create_diffusion(str(args.num_sampling_steps))
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = [0]

    # Create sampling noise:
    n = args.bs
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    print(f"z.shape: {z.shape}")
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    # z = torch.cat([z, z], 0)
    # y_null = torch.tensor([1000] * n, device=device)
    # y = torch.cat([y, y_null], 0)
    # model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    model_kwargs = dict(y=y)
    repetitions = 10
    timings = np.zeros((repetitions, 1))
    print(f"Running {repetitions} repetitions of {args.bench_type} benchmark")
    # GPU-WARM-UP
    with torch.no_grad():
        for _ in range(10):
            _ = model.forward(z, torch.ones((n), device=device), **model_kwargs)
        torch.cuda.synchronize()

        for rep in tqdm(range(repetitions)):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            starter.record()
            if args.bench_type == "sample":
                samples = diffusion.p_sample_loop(
                    model.forward,
                    z.shape,
                    z,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=True,
                    device=device,
                )
            else:
                _ = model.forward(z, torch.ones((n), device=device), **model_kwargs)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print("Inference time: {:.2f}+/-{:.2f}ms".format(mean_syn, std_syn))
    exit(0)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device,
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=list(DiT_models.keys()), default="DiT-L/2"
    )
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="vae")
    parser.add_argument(
        "--bench_type", type=str, choices=["forward", "sample"], default="forward"
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).",
    )
    args = parser.parse_args()
    main(args)
