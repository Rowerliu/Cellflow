"""
Gather latent domain images
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

    folder = r'F:\BUAA\02_Code\02_ZhangLab\02_ddib_lzy\01_results\FFTT\BCI_HE_40w\batch3'
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    time_start = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    logger.configure(dir=folder, log_suffix=time_start)
    logger.log("Title: " + "Gather latent images")
    logger.log(f"args: {args}")
    logger.log("time_start:", time_start)

    logger.log("\n---------------------" + '***********' + "---------------------\n")
    logger.log("starting to gather latent data.")


    i = args.source
    j = args.latent

    source_dir = r'F:\BUAA\02_Code\02_ZhangLab\01_guided-diffusion-lzy\02_results_saved\20230911_BCI_HE_40w\model'
    source_model, diffusion = read_model_and_diffusion(args, source_dir, synthetic=False)

    image_subfolder = os.path.join(folder, f"translation_{i}_{j}")
    pathlib.Path(image_subfolder).mkdir(parents=True, exist_ok=True)

    sources = []
    latents = []


    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True
    )
    lens = len(os.listdir(args.data_dir))


    for k, (source, extra) in enumerate(data):

        # if k < 2:
        if k < lens//args.batch_size:

            time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.log("\nepoch start:", time_now)

            logger.log(f"translating {i} -> {j}, batch {k}, shape {source.shape}...")
            logger.log(f"device: {dist_util.dev()}")

            source = source.to(dist_util.dev())

            # sources.append(source.cpu().numpy())
            # sources_path = os.path.join(image_subfolder, f'source_{k}.npy')
            # np.save(sources_path, sources)

            noise = diffusion.ddim_reverse_sample_loop(
                source_model,
                source,
                clip_denoised=False,
                device=dist_util.dev(),
                progress=True,
            )
            logger.log(f"obtained latent representation for {source.shape[0]} samples...")
            logger.log(f"latent with mean {noise.mean()} and std {noise.std()}")

            sources.append(source.cpu().numpy())
            latents.append(noise.cpu().numpy())
            logger.log("finished append")
            logger.log(f"finished loop: {k}")

            time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.log("epoch end:", time_now)
            logger.log("\n------------ next ------------")

            source_path_k = os.path.join(image_subfolder, 'source_np')
            pathlib.Path(source_path_k).mkdir(parents=True, exist_ok=True)
            source_path_k = os.path.join(source_path_k, f'source_{k}.npy')
            np.save(source_path_k, source.cpu().numpy())


            latent_path_k = os.path.join(image_subfolder, 'latent_np')
            pathlib.Path(latent_path_k).mkdir(parents=True, exist_ok=True)
            latent_path_k = os.path.join(latent_path_k, f'latent_{k}.npy')
            np.save(latent_path_k, noise.cpu().numpy())

        else:
            break

    sources = np.concatenate(sources, axis=0)
    sources_path = os.path.join(image_subfolder, f'source.npy')
    np.save(sources_path, sources)

    latents = np.concatenate(latents, axis=0)
    latents_path = os.path.join(image_subfolder, f'latent.npy')
    np.save(latents_path, latents)

    dist.barrier()

    time_complete = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.log("time_complete:", time_complete)


def create_argparser():
    defaults = dict(
        data_dir=r"G:\12_Data\03_Pathology\07_Stain-Transfer\BCI\test\03_crop\HE_120\v2\batch3",
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
            "--latent",
            type=str,
            default='latent',
            help="latent."
        )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
