"""
Synthetic domain translation from a source 2D domain to a target.
"""

import argparse
import os
import pathlib
import datetime
import torch as th
import numpy as np
import torch.distributed as dist

from scripts.common import read_model_and_diffusion
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import add_dict_to_argparser, model_and_diffusion_defaults
from guided_diffusion.image_datasets import load_data

def main():
    args_source = create_argparser(type='source').parse_args()
    args_target = create_argparser(type='target').parse_args()
    args = args_source

    dist_util.setup_dist()

    folder = r'F:\BUAA\02_Code\02_ZhangLab\08_process-learning-new\01_results\20231028_pRCC_Type1-Type2_FFFF_amount10'  # todo
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    time_start = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    logger.configure(dir=folder, log_suffix=time_start)
    logger.log("\nargs_source: ", args_source)
    logger.log("\nargs_target: ", args_target)
    logger.log("\nstarting to synthesis data.")

    time_start = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.log("time_start:", time_start)

    i = args.source
    j = args.target
    logger.log(f"reading models for synthetic data...")

    source_dir = r'F:\BUAA\02_Code\02_ZhangLab\01_guided-diffusion-lzy\02_results_saved\20231005_pRCC_Type1_c256\False\model'
    source_model, diffusion = read_model_and_diffusion(args, source_dir)

    target_dir = r'F:\BUAA\02_Code\02_ZhangLab\01_guided-diffusion-lzy\02_results_saved\20231005_pRCC_Type2_c256\False\model'
    target_model, _ = read_model_and_diffusion(args, target_dir)

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
    # lens = 2
    # define amount (m) of middle images, m should be divisible by 1000
    amount = 10  # 10  todo

    for k, (source, extra) in enumerate(data):

        if k < lens//args.batch_size:

            source = source.to(dist_util.dev())
            # source_path_k = os.path.join(image_subfolder, 'source_np')
            # pathlib.Path(source_path_k).mkdir(parents=True, exist_ok=True)
            # source_path_k = os.path.join(source_path_k, f'source_{k:04d}.npy')
            # np.save(source_path_k, source.cpu().numpy())
            sources.append(source.cpu().numpy())

            diffusion_steps = args.diffusion_steps
            diffway_list = np.round(np.linspace(0, diffusion_steps, amount+1)).astype(int)

            logger.log(f"\ndevice: {dist_util.dev()}")
            logger.log(f"translating: {i}->{j}, batch: {k+1}, shape: {source.shape}, diffway_list: ", diffway_list)

            for m in range(amount):

                time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                logger.log("\ntime_now:", time_now)
                logger.log(f"[batch / amount / total]: [{k+1} / {m+1} / {amount}]")

                diffway_start = diffway_list[m]
                diffway_end = diffway_list[m+1]

                noise_clip_denoised = False  # todo
                noise = diffusion.ddim_reverse_sample_loop(
                    source_model,
                    source,
                    clip_denoised=noise_clip_denoised,
                    device=dist_util.dev(),
                    progress=True,
                    amount=amount,
                    diffway_start=diffway_start,
                    diffway_end=diffway_end,
                )
                source = noise

                logger.log('\n noise_clip_denoise: ', noise_clip_denoised)
                logger.log(f"obtained latent representation for {source.shape[0]} samples...")
                logger.log(f"latent with mean {noise.mean()} and std {noise.std()}")

                target_clip_denoised = False
                target = diffusion.ddim_sample_loop(
                    target_model, (args.batch_size, 3, args.image_size, args.image_size),
                    noise=noise,
                    clip_denoised=target_clip_denoised,
                    device=dist_util.dev(),
                    progress=True,
                    amount=amount,
                    diffway_end=diffway_end,
                )
                logger.log('\n target_clip_denoise: ', target_clip_denoised)
                logger.log(f"finished translation {target.shape}")

                noise = ((noise + 1) * 127.5).clamp(0, 255).to(th.uint8)
                # latent_path_k = os.path.join(image_subfolder, 'latent_np')
                # pathlib.Path(latent_path_k).mkdir(parents=True, exist_ok=True)
                # latent_path_k = os.path.join(latent_path_k, f'latent_{k:04d}_{m:04d}.npy')
                # np.save(latent_path_k, noise.cpu().numpy())

                target = ((target + 1) * 127.5).clamp(0, 255).to(th.uint8)
                # target_path_k = os.path.join(image_subfolder, 'target_np')
                # pathlib.Path(target_path_k).mkdir(parents=True, exist_ok=True)
                # target_path_k = os.path.join(target_path_k, f'target_{k:04d}_{m:04d}.npy')
                # np.save(target_path_k, target.cpu().numpy())


                latents.append(noise.cpu().numpy())
                targets.append(target.cpu().numpy())

        else:
            break


    sources = np.stack(sources, axis=1)
    sources_path = os.path.join(image_subfolder, 'source.npy')
    np.save(sources_path, sources)

    latents = np.stack(latents, axis=1)
    # grouped_latents = np.reshape(latents, (lens, amount, latents.shape[1], latents.shape[2], latents.shape[3]))
    latents_path = os.path.join(image_subfolder, 'latent.npy')
    np.save(latents_path, latents)

    targets = np.stack(targets, axis=1)
    # grouped_targets = np.reshape(targets, (lens, amount, targets.shape[1], targets.shape[2], targets.shape[3]))
    targets_path = os.path.join(image_subfolder, 'target.npy')
    np.save(targets_path, targets)

    dist.barrier()
    logger.log(f"synthetic data translation complete: {i}->{j}\n\n")
    time_complete = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.log("time_complete:", time_complete)


def create_argparser(type=None):
    defaults = dict(
        data_dir=r"E:\12_Data\03_Pathology\09_pRCC\crop_384\type1_120",  # todo
        image_size=256,
        batch_size=4,  # depends on the size of GPU
    )
    defaults.update(model_and_diffusion_defaults())
    if type == 'source':
        new_defaults = dict(
            use_new_attention_order=True,
        )
    elif type == 'target':
        new_defaults = dict(
            use_new_attention_order=False,
        )
    else:
        new_defaults = defaults
    defaults.update(new_defaults)

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--source",
            type=int,
            default=0,
            help="Source dataset pRCC Type1."
        )
    parser.add_argument(
            "--target",
            type=int,
            default=1,
            help="Target dataset pRCC Type2."
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
