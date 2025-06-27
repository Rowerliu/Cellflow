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
from PIL import Image
from scripts.common import read_model_and_diffusion
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import add_dict_to_argparser, model_and_diffusion_defaults
from guided_diffusion.image_datasets import load_data

def main():
    args_source = create_argparser(type='source').parse_args()
    args_target = create_argparser(type='target').parse_args()
    args = args_source

    dist_util.setup_dist()

    folder = r'E:\01_LZY\02_Code\04_Cellflow\01_results\20250528_MHIST_HP-SSA_TTFF_amount10'  # todo
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

    source_dir = r'E:\01_LZY\02_Code\02_ADM\01_results\20250518_MHIST_r256_Blur_10w\HP\True'
    source_model, diffusion = read_model_and_diffusion(args, source_dir)

    target_dir = r'E:\01_LZY\02_Code\02_ADM\01_results\20250518_MHIST_r256_Blur_10w\SSA\False'
    target_model, _ = read_model_and_diffusion(args, target_dir)

    image_subfolder = os.path.join(folder, f"translation_{i}_{j}")
    pathlib.Path(image_subfolder).mkdir(parents=True, exist_ok=True)

    source_img_dir = os.path.join(image_subfolder + r'/source_img')
    latent_img_dir = os.path.join(image_subfolder + r'/latent_img')
    target_img_dir = os.path.join(image_subfolder + r'/target_img')

    pathlib.Path(source_img_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(latent_img_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(target_img_dir).mkdir(parents=True, exist_ok=True)

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
    # lens = len(os.listdir(args.data_dir))
    lens = 8
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

            # 将source转换为图像格式并保存
            source_for_img = ((source + 1) * 127.5).clamp(0, 255).to(th.uint8)
            save_tensor_as_images(source_for_img, source_img_dir, "source", k)

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

                # 保存latent和target的图像文件
                save_tensor_as_images(noise, latent_img_dir, "latent", k, m)
                save_tensor_as_images(target, target_img_dir, "target", k, m)

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


def save_tensor_as_images(tensor, save_dir, prefix, batch_idx, step_idx=None):
    """
    将tensor保存为图像文件

    Args:
        tensor: 形状为 (batch_size, channels, height, width) 的tensor
        save_dir: 保存目录
        prefix: 文件名前缀 ('source', 'latent', 'target')
        batch_idx: 批次索引
        step_idx: 步骤索引（对于latent和target）
    """
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 确保tensor在CPU上并转换为numpy
    if isinstance(tensor, th.Tensor):
        tensor = tensor.cpu().numpy()

    batch_size = tensor.shape[0]

    for i in range(batch_size):
        # 获取单张图像 (C, H, W)
        img_array = tensor[i]

        # 转换为 (H, W, C) 格式
        if len(img_array.shape) == 3 and img_array.shape[0] == 3:
            img_array = np.transpose(img_array, (1, 2, 0))

        # 确保数值在0-255范围内
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        # 构造文件名
        if step_idx is not None:
            filename = f"{prefix}_batch{batch_idx:04d}_img{i:04d}_step{step_idx:04d}.jpg"
        else:
            filename = f"{prefix}_batch{batch_idx:04d}_img{i:04d}.jpg"

        filepath = os.path.join(save_dir, filename)

        # 保存图像
        if len(img_array.shape) == 3:
            img = Image.fromarray(img_array, 'RGB')
        else:
            img = Image.fromarray(img_array, 'L')

        img.save(filepath)


def create_argparser(type=None):
    defaults = dict(
        data_dir=r"E:\01_LZY\01_Data\z13_MHIST\02_resize\r256\HP",  # todo
        image_size=256,
        batch_size=4,  # depends on the size of GPU
    )
    defaults.update(model_and_diffusion_defaults())
    if type == 'source':
        new_defaults = dict(
            use_new_attention_order=False,
        )
    elif type == 'target':
        new_defaults = dict(
            use_new_attention_order=True,
        )
    else:
        new_defaults = defaults
    defaults.update(new_defaults)

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--source",
            type=str,
            default='HP',
            help="Source dataset MHIST HP."
        )
    parser.add_argument(
            "--target",
            type=str,
            default='SSA',
            help="Target dataset MHIST SSA."
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    th.cuda.empty_cache()
    with th.cuda.device(0):
        main()

