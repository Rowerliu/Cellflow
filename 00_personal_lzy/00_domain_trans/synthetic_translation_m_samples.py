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
from guided_diffusion.script_util import model_and_diffusion_defaults_2d, add_dict_to_argparser, model_and_diffusion_defaults
from guided_diffusion.synthetic_datasets import scatter, heatmap, load_2d_data, Synthetic2DType
from guided_diffusion.image_datasets import load_data

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure(dir=r'H:\BUAA\02_Code\02_Diffusion\22_ddib\01_results')
    logger.log("starting to sample synthetic data.")
    logger.log("args: ", args)

    time_start = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.log("time_start:", time_start)

    # image_folder = os.path.join(code_folder, f"experiments/images")
    image_folder = os.path.join(rf'F:\BUAA\02_Code\21_Diffusion\12_ddib\01_results')
    pathlib.Path(image_folder).mkdir(parents=True, exist_ok=True)

    i = args.source
    j = args.target
    logger.log(f"reading models for synthetic data...")


    source_dir = r'H:\BUAA\02_Code\02_Diffusion\12_guided-diffusion-main\02_results_saved\20230606_ROSE_Negative_size256_sample100'
    source_model, diffusion = read_model_and_diffusion(args, source_dir, synthetic=False)

    target_dir = r'H:\BUAA\02_Code\02_Diffusion\12_guided-diffusion-main\02_results_saved\20230607_ROSE_Positive_size256_sample100'
    target_model, _ = read_model_and_diffusion(args, target_dir, synthetic=False)


    image_subfolder = os.path.join(image_folder, f"translation_{i}_{j}")
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

    # define amount (m) of middle images, m should be divisible by 1000
    amount = 10

    for k, (source, extra) in enumerate(data):

        if k < lens//args.batch_size:

            source = source.to(dist_util.dev())
            # source = ((source + 1) * 127.5).clamp(0, 255).to(th.uint8)
            source_path_k = os.path.join(image_subfolder, 'sources', f'source_{k}.npy')
            np.save(source_path_k, source.cpu().numpy())
            sources.append(source.cpu().numpy())

            m = amount
            while m > 0:

                logger.log(f"translating {i}->{j}, batch {k}, diffway {m}, shape {source.shape}...")
                logger.log(f"device: {dist_util.dev()}")

                noise = diffusion.ddim_reverse_sample_loop(
                    source_model,
                    source,
                    clip_denoised=False,
                    device=dist_util.dev(),
                    amount=amount,
                    diffway=m,
                )
                logger.log(f"obtained latent representation for {source.shape[0]} samples...")
                logger.log(f"latent with mean {noise.mean()} and std {noise.std()}")

                time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                logger.log("time_now:", time_now)

                target = diffusion.ddim_sample_loop(
                    target_model, (args.batch_size, 3, args.image_size, args.image_size),
                    noise=noise,
                    clip_denoised=False,
                    device=dist_util.dev(),
                    amount=amount,
                    diffway=m,
                )
                logger.log(f"finished translation {target.shape}")

                # noise = ((noise + 1) * 127.5).clamp(0, 255).to(th.uint8)
                latent_path_k = os.path.join(image_subfolder, 'latents', f'latent_{k}_{m}.npy')
                np.save(latent_path_k, noise.cpu().numpy())

                # target = ((target + 1) * 127.5).clamp(0, 255).to(th.uint8)
                target_path_k = os.path.join(image_subfolder, 'targets', f'target_{k}_{m}.npy')
                np.save(target_path_k, target.cpu().numpy())


                latents.append(noise.cpu().numpy())
                targets.append(target.cpu().numpy())

                m = m - 1

        else:
            break


    sources = np.concatenate(sources, axis=0)
    sources_path = os.path.join(image_subfolder, 'source.npy')
    np.save(sources_path, sources)


    latents = np.concatenate(latents, axis=0)
    grouped_latents = np.reshape(latents, (lens, amount, latents.shape[1], latents.shape[2], latents.shape[3]))
    latents_path = os.path.join(image_subfolder, 'latent.npy')
    np.save(latents_path, grouped_latents)


    targets = np.concatenate(targets, axis=0)
    grouped_targets = np.reshape(targets, (lens, amount, targets.shape[1], targets.shape[2], targets.shape[3]))
    targets_path = os.path.join(image_subfolder, 'target.npy')
    np.save(targets_path, grouped_targets)

    dist.barrier()
    logger.log(f"synthetic data translation complete: {i}->{j}\n\n")
    time_complete = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.log("time_complete:", time_complete)


def create_argparser():
    defaults = dict(
        data_dir=r"H:\BUAA\05_Dataset\02_Pathology\02_ROSE\fewshot\256\Negative_5",
        image_size=256,
        batch_size=1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--source",
            type=int,
            default=0,
            help="Source dataset ROSE Negative."
        )
    parser.add_argument(
            "--target",
            type=int,
            default=1,
            help="Target dataset ROSE Positive."
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
