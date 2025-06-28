"""
Class-conditional image transition from one ImageNet class to another.
"""

import argparse
import os
import pathlib
import datetime
import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from scripts.common_cfg import read_model_and_diffusion
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import (
    load_data,
)
from guided_diffusion.script_util_cfg import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.fourier_trans import get_adaptive_list


def main():
    args_source = create_argparser(type='source').parse_args()
    args_target = create_argparser(type='target').parse_args()
    args = args_source

    dist_util.setup_dist()

    folder = args.result_dir
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    i = args.source
    j = args.target
    source_domain = args.domain_name[i]
    target_domain = args.domain_name[j]
    image_subfolder = os.path.join(folder, f"{source_domain}_{target_domain}")
    pathlib.Path(image_subfolder).mkdir(parents=True, exist_ok=True)

    time_start = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    logger.configure(dir=image_subfolder, log_suffix=time_start)
    logger.log("\nargs_source: ", args_source)
    logger.log("\nargs_target: ", args_target)
    logger.log("\nstarting to synthesis data.")

    time_start = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.log("time_start:", time_start)
    logger.log(f"reading models for synthetic data...")

    source_dir = os.path.join(args.model_dir, str(args_source.use_new_attention_order))  # fixme
    source_model, diffusion = read_model_and_diffusion(args_source, source_dir)

    target_dir = os.path.join(args.model_dir, str(args_target.use_new_attention_order))  # fixme
    target_model, _ = read_model_and_diffusion(args_target, target_dir)

    weight = args.weight

    # Copies the source dataset
    logger.log("copying source dataset.")
    source = [int(args.source)]
    target = [int(args.target)]
    source_to_target_mapping = {s: t for s, t in zip(source, target)}  # cannot transfer to multi-domain once time

    source_img_dir = os.path.join(image_subfolder + r'/source_img')
    latent_img_dir = os.path.join(image_subfolder + r'/latent_img')
    target_img_dir = os.path.join(image_subfolder + r'/target_img')

    pathlib.Path(source_img_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(latent_img_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(target_img_dir).mkdir(parents=True, exist_ok=True)

    sources = []
    latents = []
    targets = []
    diffway_list_all = []

    logger.log("running image transition...")
    data = load_data(
        batch_size=args.batch_size,
        image_size=args.image_size,
        data_dir=args.data_dir,
        class_cond=True,
        deterministic=True
    )

    source_data_dir = os.path.join(args.data_dir, source_domain)
    lens = len(os.listdir(source_data_dir))
    # lens = 8

    # if resume from length point
    resume = 0

    # define amount (m) of progressive images
    amount = 10

    bs = 0
    for k, (source, extra) in enumerate(data):
        if bs < lens//args.batch_size:
            if bs < resume:
                bs = bs + 1
                continue
            if int(extra["y"]) == args.source:

                source = source.to(dist_util.dev())
                # source_path_k = os.path.join(image_subfolder, 'source_np')
                # pathlib.Path(source_path_k).mkdir(parents=True, exist_ok=True)
                # source_path_k = os.path.join(source_path_k, f'source_{k:04d}.npy')
                # np.save(source_path_k, source.cpu().numpy())
                sources.append(source.cpu().numpy())

                # save source image
                source_for_img = ((source + 1) * 127.5).clamp(0, 255).to(th.uint8)
                save_tensor_as_images(source_for_img, source_img_dir, "source", k)

                # Class labels for source and target sets
                source_y = dict(y=extra["y"].to(dist_util.dev()))
                target_y_list = [source_to_target_mapping[v.item()] for v in extra["y"]]
                target_y = dict(y=th.tensor(target_y_list).to(dist_util.dev()))

                out_fourier_magnitude_list = diffusion.fourier_frequency_list_loop(
                    source_model,
                    source,
                    model_kwargs=source_y,
                    clip_denoised=args.clip_denoised,
                    device=dist_util.dev(),
                    progress=True,
                    weight=weight,
                )

                diffway_list = get_adaptive_list(out_fourier_magnitude_list, amount)
                diffway_list_all.extend(diffway_list)
                diffway_list.insert(0, 0)  # initial time-step zero for diffway_start

                logger.log(f"\ndevice: {dist_util.dev()}")
                logger.log(f"translating: {i}->{j}, batch: {bs+1}, shape: {source.shape}, diffway_list: ", diffway_list)

                m = 0
                while m < amount:
                    time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    logger.log("\ntime_now:", time_now)
                    logger.log(f"[batch / amount / total]: [{bs+1} / {m+1} / {amount}]")

                    diffway_start = diffway_list[m]
                    diffway_end = diffway_list[m+1]

                    # First, use DDIM to encode to latents.
                    logger.log("encoding the source images.")
                    noise = diffusion.ddim_reverse_sample_loop(
                        source_model,
                        source,
                        clip_denoised=args.clip_denoised,
                        model_kwargs=source_y,
                        device=dist_util.dev(),
                        progress=True,
                        weight=weight,
                        amount=amount,
                        diffway_start=diffway_start,
                        diffway_end=diffway_end,
                    )
                    source = noise
                    logger.log(f"obtained latent representation for {noise.shape[0]} samples...")
                    logger.log(f"latent with mean {noise.mean()} and std {noise.std()}")

                    # Next, decode the latents to the target class.
                    time_noise = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    logger.log("time_noise:", time_noise)
                    target = diffusion.ddim_sample_loop(
                        target_model,
                        (args.batch_size, 3, args.image_size, args.image_size),
                        noise=noise,
                        clip_denoised=args.clip_denoised,
                        model_kwargs=target_y,
                        device=dist_util.dev(),
                        progress=True,
                        eta=args.eta,
                        weight=weight,
                        amount=amount,
                        diffway_end=diffway_end,
                    )
                    logger.log(f"finished transition {target.shape}")

                    noise = ((noise + 1) * 127.5).clamp(0, 255).to(th.uint8)
                    # latent_path_k = os.path.join(image_subfolder, 'latent_np')
                    # pathlib.Path(latent_path_k).mkdir(parents=True, exist_ok=True)
                    # latent_path_k = os.path.join(
                    # latent_path_k, f'latent_{k:04d}_{m:04d}.npy')
                    # np.save(latent_path_k, noise.cpu().numpy())


                    target = ((target + 1) * 127.5).clamp(0, 255).to(th.uint8)
                    # target_path_k = os.path.join(image_subfolder, 'target_np')
                    # pathlib.Path(target_path_k).mkdir(parents=True, exist_ok=True)
                    # target_path_k = os.path.join(target_path_k, f'target_{k:04d}_{m:04d}.npy')
                    # np.save(target_path_k, target.cpu().numpy())

                    # save latent & target image
                    save_tensor_as_images(noise, latent_img_dir, "latent", bs, diffway_end)
                    prefix = source_domain + '-' + target_domain
                    save_tensor_as_images(target, target_img_dir, prefix, bs, diffway_end)

                    latents.append(noise.cpu().numpy())
                    targets.append(target.cpu().numpy())

                    m = m + 1
                bs = bs + 1
            else:
                pass
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

    diffway_np = np.array(diffway_list_all).reshape(-1, amount)
    diffway_path = os.path.join(image_subfolder, 'diffway.npy')
    np.save(diffway_path, diffway_np)

    dist.barrier()
    logger.log(f"synthetic transition data complete: {i}->{j}\n\n")
    time_complete = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.log("time_complete:", time_complete)


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


def save_tensor_as_images(tensor, save_dir, prefix, batch_idx, step_idx=None):
    """
    将tensor保存为图像文件

    Args:
        tensor: shape ()batch_size, channels, height, width)
        save_dir:
        prefix: prefix of file ('source', 'latent', 'target')
        batch_idx: for name inorder
        step_idx: for name inorder
    """
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    if isinstance(tensor, th.Tensor):
        tensor = tensor.cpu().numpy()

    batch_size = tensor.shape[0]

    for i in range(batch_size):
        img_array = tensor[i]

        if len(img_array.shape) == 3 and img_array.shape[0] == 3:
            img_array = np.transpose(img_array, (1, 2, 0))

        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        if step_idx is not None:
            filename = f"{prefix}_batch{batch_idx:04d}_img{i:04d}_step{step_idx:04d}.jpg"
        else:
            filename = f"{prefix}_batch{batch_idx:04d}_img{i:04d}.jpg"

        filepath = os.path.join(save_dir, filename)

        if len(img_array.shape) == 3:
            img = Image.fromarray(img_array, 'RGB')
        else:
            img = Image.fromarray(img_array, 'L')

        img.save(filepath)


def create_argparser(type=None):
    defaults_all = dict()
    defaults = dict(
        # folder to save flow images
        result_dir=r"F:\12_Code\02_Zhanglab\04_Cellflow\02_Flow\01_result\20250627_ROSE\TTFF_cfg_ddim100_a10",  # fixme
        # folder of flow model
        model_dir=r"F:\12_Code\02_Zhanglab\04_Cellflow\01_Diffusion\01_result\20250618_ROSE-r640_diff-512_cfg_blur_10w",
        # folder of flow data
        data_dir=r"F:\13_Data\03_Pathology\a04_Cellflow\02_classification\01_ROSE\train",  # todo

        clip_denoised=False,
        batch_size=1,
        eta=0.0,
        image_size=512,
        class_cond=True,
        weight=1.8,  # weight for classifier-free guidance
        num_classes=2,
        out_channels=2,
        timestep_respacing="ddim100",
    )
    defaults_all.update(model_and_diffusion_defaults())
    defaults_all.update(defaults)

    if type == 'source':
        defaults_add = dict(
            use_new_attention_order=True,
        )
    elif type == 'target':
        defaults_add = dict(
            use_new_attention_order=False,
        )
    else:
        defaults_add = defaults

    defaults_all.update(defaults_add)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=int,
        default=0,
        help="Source dataset ROSE."
    )
    parser.add_argument(
        "--target",
        type=int,
        default=1,
        help="Target dataset ROSE."
    )
    parser.add_argument(
        "--domain_name",
        type=list,
        default=["negative", "positive"],
        help="domain name list"
    )
    add_dict_to_argparser(parser, defaults_all)
    return parser


if __name__ == "__main__":
    th.cuda.empty_cache()
    with th.cuda.device(0):
        main()
