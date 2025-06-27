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
import imageio
from scripts.common import read_model_and_diffusion
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    model_and_diffusion_defaults
)
from guided_diffusion.synthetic_datasets import scatter, heatmap
from guided_diffusion.image_datasets import load_data


def main():
    dist_util.setup_dist()

    image_folder = r'E:\LZY\02_ddib_lzy\01_results_3'
    pathlib.Path(image_folder).mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    logger.configure(dir=image_folder, log_suffix=now)

    time_start = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.log("time_start:", time_start)
    args = create_argparser().parse_args()
    logger.log(f"args: {args}")
    logger.log("starting to transfer.")

    i = args.latent
    j = args.target
    bs = args.batch_size
    logger.log(f"reading models for transfer...")
    target_dir = r'E:\LZY\01_guided-diffusion-lzy\02_results_saved\20230719_BCI_IHC_5w_all_384-256\model'
    target_model, diffusion = read_model_and_diffusion(args, target_dir, synthetic=False)

    image_subfolder = os.path.join(image_folder, f"translation_{i}_{j}")
    pathlib.Path(image_subfolder).mkdir(parents=True, exist_ok=True)

    latents = []
    targets = []

    # load data
    # data = load_data(
    #     data_dir=args.data_dir,
    #     batch_size=args.batch_size,
    #     image_size=args.image_size,
    #     class_cond=args.class_cond,
    #     deterministic=True
    # )
    # lens = len(os.listdir(args.data_dir))

    # load npy
    data_root = args.numpy_dir
    data = np.load(data_root)
    # lens = 8
    lens = data.shape[0]



    # std_data = []
    # for i in range(lens):
    #     sample = data[i].astype('float64')
    #     for j in range(sample.shape[0]):
    #         sample[j] = (sample[j] - sample[j].mean()) / sample[j].std()
    #     data[i] = sample.astype('float32')

    k = 0
    while k * bs < lens:
        lowerbound = k * bs
        if (lowerbound + bs) > lens:
            upperbound = lens - lowerbound
        else:
            upperbound = lowerbound + bs
        latent = data[lowerbound:upperbound, :, :, :]
        time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.log('\n' + "time_now:", time_now)
        logger.log(f"translating: {i}->{j}, batch: {k+1}, shape: {latent.shape}...")
        logger.log(f"device: {dist_util.dev()}")
        latent = th.tensor(latent)
        latent = latent.to(dist_util.dev())
        noise = latent

        logger.log(f"obtained latent representation for {latent.shape[0]} samples...")
        logger.log(f"latent with mean {noise.mean()} and std {noise.std()}")

        # standardization
        # for x in range(noise.shape[0]):
        #     noise[x] = (noise[x]-noise[x].mean()) / noise[x].std()
        # logger.log(f"latent with mean {noise.mean()} and std {noise.std()} after standardization")

        target = diffusion.ddim_sample_loop(
            target_model,
            (latent.shape[0], 3, args.image_size, args.image_size),
            noise=noise,
            clip_denoised=False,
            device=dist_util.dev(),
            progress=True,
        )
        logger.log(f"finished translation {target.shape}")

        latent = ((latent + 1) * 127.5).clamp(0, 255).to(th.uint8)
        target = ((target + 1) * 127.5).clamp(0, 255).to(th.uint8)

        latents.append(latent.cpu().numpy())
        targets.append(target.cpu().numpy())
        logger.log("finished append")

        latent_path_k = os.path.join(image_subfolder, 'latent_np')
        pathlib.Path(latent_path_k).mkdir(parents=True, exist_ok=True)
        latent_path = os.path.join(latent_path_k, f'latent_{k}.npy')
        np.save(latent_path, noise.cpu().numpy())

        target_path_k = os.path.join(image_subfolder, 'target_np')
        pathlib.Path(target_path_k).mkdir(parents=True, exist_ok=True)
        target_path = os.path.join(target_path_k, f'target_{k}.npy')
        np.save(target_path, target.cpu().numpy())

        k = k + 1


    latents = np.concatenate(latents, axis=0)
    latents_path = os.path.join(image_subfolder, f'latent.npy')
    np.save(latents_path, latents)

    targets = np.concatenate(targets, axis=0)
    targets_path = os.path.join(image_subfolder, f'target.npy')
    np.save(targets_path, targets)

    # save numpy as jpg for visual
    npy2img(data=targets, data_name='target_img', folder=image_subfolder)
    npy2img(data=latents, data_name='latent_img', folder=image_subfolder)

    dist.barrier()
    logger.log(f"\ndata translation complete: {i}->{j}")

    time_complete = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.log("time_complete:", time_complete)


def create_argparser():
    defaults = dict(
        data_dir=r"F:\BUAA\02_Code\21_Diffusion\12_ddib\05_data\HE2IHC\target_s",
        numpy_dir=r'E:\LZY\10_diffusion_latent_trans\02_results_saved\20230804_HE2IHC_sample64_ts980\output\output.npy',
        image_size=256,
        batch_size=2,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--latent",
            type=int,
            default=1,
            help="Latent dataset HE."
        )
    parser.add_argument(
            "--target",
            type=int,
            default=2,
            help="Target dataset IHC."
    )
    add_dict_to_argparser(parser, defaults)
    return parser

def npy2img(data, data_name, folder):

    n = len(data)
    for i in range(n):
        latent_data = data[i]
        latent_data = latent_data.transpose(1, 2, 0)
        # latent_data = ((latent_data + 1) * 127.5).astype('uint8')
        img_folder = os.path.join(folder + rf'/{data_name}')
        pathlib.Path(img_folder).mkdir(parents=True, exist_ok=True)
        imageio.imwrite(os.path.join(img_folder + rf'/{data_name}_{i}.jpeg'), latent_data)

        print('latent_i: ', i)


if __name__ == "__main__":
    main()
