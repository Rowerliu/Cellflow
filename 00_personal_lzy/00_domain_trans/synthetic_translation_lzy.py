"""
Synthetic domain translation from a source 2D domain to a target.
"""

import argparse
import os
import pathlib
import torch as th
import numpy as np
import torch.distributed as dist
import datetime
from scripts.common import read_model_and_diffusion
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    model_and_diffusion_defaults
)
from guided_diffusion.synthetic_datasets import scatter, heatmap
from guided_diffusion.image_datasets import load_data


def main():

    args = create_argparser().parse_args()
    dist_util.setup_dist()

    folder = r'F:\BUAA\02_Code\02_ZhangLab\02_ddib_lzy\01_results\20230915_PAIP_N2P'
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    time_start = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    logger.configure(dir=folder, log_suffix=time_start)
    logger.log("Title: " + "Transfer HE to IHC")
    logger.log(f"args: {args}")
    logger.log("time_start:", time_start)

    logger.log("\n---------------------" + '***********' + "---------------------\n")

    logger.log("starting to sample synthetic data.")

    i = args.source
    j = args.target

    logger.log(f"reading models for synthetic data...")

    source_dir = r'G:\03_BUAA_old\02_Code\21_Diffusion\07_guided-diffusion\02_results_saved\20230722_PAIP_neg_384-256_all\model'
    source_model, diffusion = read_model_and_diffusion(args, source_dir, synthetic=False)

    target_dir = r'G:\03_BUAA_old\02_Code\21_Diffusion\07_guided-diffusion\02_results_saved\20230723_PAIP_pos_384-256_all\model'
    target_model, _ = read_model_and_diffusion(args, target_dir, synthetic=False)


    image_subfolder = os.path.join(folder, f"translation_{i}_{j}")
    pathlib.Path(image_subfolder).mkdir(parents=True, exist_ok=True)

    sources = []
    latents = []
    targets = []


    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True
    )
    lens = len(os.listdir(args.data_dir))


    for k, (source, extra) in enumerate(data):

        if k < lens//args.batch_size:
            time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.log("\nepoch start:", time_now)

            logger.log(f"translating {i} -> {j}, batch {k}, shape {source.shape}...")
            logger.log(f"device: {dist_util.dev()}")

            source = source.to(dist_util.dev())

            noise_clip_denoised = False  # todo
            noise = diffusion.ddim_reverse_sample_loop(
                source_model,
                source,
                clip_denoised=noise_clip_denoised,
                device=dist_util.dev(),
                progress=True,
            )
            logger.log('noise_clip_denoise: ', noise_clip_denoised)
            logger.log(f"obtained latent representation for {source.shape[0]} samples...")
            logger.log(f"latent with mean {noise.mean()} and std {noise.std()}")

            target_clip_denoised = False
            target = diffusion.ddim_sample_loop(
                target_model,
                (args.batch_size, 3, args.image_size, args.image_size),
                noise=noise,
                clip_denoised=target_clip_denoised,
                device=dist_util.dev(),
                progress=True,
            )
            logger.log('target_clip_denoise: ', target_clip_denoised)
            logger.log(f"finished translation {target.shape}")

            # sources = np.concatenate(sources, axis=0)
            source_path = os.path.join(image_subfolder, 'source_np')
            pathlib.Path(source_path).mkdir(parents=True, exist_ok=True)
            source_path = os.path.join(source_path, f'source_{k:04d}.npy')
            np.save(source_path, source.cpu().numpy())


            # latents = np.concatenate(latents, axis=0)
            latent_path = os.path.join(image_subfolder, 'latent_np')
            pathlib.Path(latent_path).mkdir(parents=True, exist_ok=True)
            latent_path = os.path.join(latent_path, f'latent_{k:04d}.npy')
            np.save(latent_path, noise.cpu().numpy())


            # targets = np.concatenate(targets, axis=0)
            target = ((target + 1) * 127.5).clamp(0, 255).to(th.uint8)
            target_path = os.path.join(image_subfolder, 'target_np')
            pathlib.Path(target_path).mkdir(parents=True, exist_ok=True)
            target_path = os.path.join(target_path, f'target_{k:04d}.npy')
            np.save(target_path, target.cpu().numpy())

            sources.append(source.cpu().numpy())
            latents.append(noise.cpu().numpy())
            targets.append(target.cpu().numpy())
            logger.log("finished append")

            time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.log("epoch end:", time_now)

        else:
            break

    sources = np.concatenate(sources, axis=0)
    sources_path = os.path.join(image_subfolder, f'source.npy')
    np.save(sources_path, sources)

    latents = np.concatenate(latents, axis=0)
    latents_path = os.path.join(image_subfolder, f'latent.npy')
    np.save(latents_path, latents)

    targets = np.concatenate(targets, axis=0)
    targets_path = os.path.join(image_subfolder, f'target.npy')
    np.save(targets_path, targets)


    dist.barrier()
    logger.log(f"synthetic data translation complete: {i} -> {j}\n")
    time_complete = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.log("time_complete:", time_complete)


def create_argparser():
    defaults = dict(
        data_dir=r"E:\12_Data\03_Pathology\04_BCI\val\HE_Resize256_sample8",
        image_size=256,
        batch_size=4,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--source",
            type=str,
            default='HE',
            help="Source dataset HE."
        )
    parser.add_argument(
            "--target",
            type=str,
            default='IHC',
            help="Target dataset IHC."
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
