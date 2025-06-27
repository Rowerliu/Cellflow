"""
Class-conditional image transition from one ImageNet class to another.
"""

import argparse
import os
import pathlib
import datetime
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from scripts.common import read_model_and_diffusion, read_classifier
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import (
    load_data,
)
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.fourier_trans import get_adaptive_list


def main():
    args_source = create_argparser(type='source').parse_args()
    args_target = create_argparser(type='target').parse_args()
    args = args_source

    dist_util.setup_dist()

    # folder = os.path.join(rf'D:\01_LZY\02_Code\08_process-learning-V3\01_results\20240103_Gleason_transition_PTL\benign-gleason3\TTFF_ddim100_a10_classifier')
    folder = os.path.join(rf'D:\01_LZY\02_Code\08_process-learning-V3\01_results\20240119_Gleason_transition_PTL_512\TTFF_ddim100_a10_classifier')
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

    source_dir = r'D:\01_LZY\02_Code\01_guided-diffusion\02_results_saved\20231223_gleason_4class_r320-256_Blur\True\model'
    source_model, diffusion = read_model_and_diffusion(args_source, source_dir)

    target_dir = r'D:\01_LZY\02_Code\01_guided-diffusion\02_results_saved\20231223_gleason_4class_r320-256_Blur\False\model'
    target_model, _ = read_model_and_diffusion(args_target, target_dir)

    classifier_dir = r'D:\01_LZY\02_Code\01_guided-diffusion\02_results_saved\20231223_gleason_4class_r320-256_Blur\Classifier'
    classifier = read_classifier(args, classifier_dir)

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_source(x, t, y=None):
        assert y is not None
        return source_model(x, t, y if args.class_cond else None)

    def model_target(x, t, y=None):
        assert y is not None
        return target_model(x, t, y if args.class_cond else None)

    # Copies the source dataset
    logger.log("copying source dataset.")
    source = [int(v) for v in args.source.split(",")]
    target = [int(v) for v in args.target.split(",")]
    source_to_target_mapping = {s: t for s, t in zip(source, target)}  # todo  cannot transfer to multi-domain once time

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

    lens = len(os.listdir(args.data_dir))
    # lens = 8
    # define amount (m) of middle images
    amount = 10

    for k, (source, extra) in enumerate(data):

        if k < lens//args.batch_size:

            source = source.to(dist_util.dev())
            # source_path_k = os.path.join(image_subfolder, 'source_np')
            # pathlib.Path(source_path_k).mkdir(parents=True, exist_ok=True)
            # source_path_k = os.path.join(source_path_k, f'source_{k:04d}.npy')
            # np.save(source_path_k, source.cpu().numpy())
            sources.append(source.cpu().numpy())

            # Class labels for source and target sets
            source_y = dict(y=extra["y"].to(dist_util.dev()))
            target_y_list = [source_to_target_mapping[v.item()] for v in extra["y"]]
            target_y = dict(y=th.tensor(target_y_list).to(dist_util.dev()))

            out_fourier_magnitude_list = diffusion.fourier_frequency_list_loop(
                model_source,
                source,
                model_kwargs=source_y,
                clip_denoised=args.clip_denoised,
                device=dist_util.dev(),
                progress=True,
            )

            diffway_list = get_adaptive_list(out_fourier_magnitude_list, amount)
            diffway_list_all.extend(diffway_list)
            diffway_list.insert(0, 0)  # initial time-step zero for diffway_start

            logger.log(f"\ndevice: {dist_util.dev()}")
            logger.log(f"translating: {i}->{j}, batch: {k+1}, shape: {source.shape}, diffway_list: ", diffway_list)

            m = 0
            while m < amount:
                time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                logger.log("\ntime_now:", time_now)
                logger.log(f"[batch / amount / total]: [{k+1} / {m+1} / {amount}]")

                diffway_start = diffway_list[m]
                diffway_end = diffway_list[m+1]

                # First, use DDIM to encode to latents.
                logger.log("encoding the source images.")
                noise = diffusion.ddim_reverse_sample_loop(
                    model_source,
                    source,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=source_y,
                    device=dist_util.dev(),
                    progress=True,
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
                    model_target,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    noise=noise,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=target_y,
                    cond_fn=cond_fn,
                    device=dist_util.dev(),
                    progress=True,
                    eta=args.eta,
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

                latents.append(noise.cpu().numpy())
                targets.append(target.cpu().numpy())

                m = m + 1

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


def create_argparser(type=None):
    defaults = dict(
        # data_dir=r"D:\01_LZY\01_Data\03_Pathology\05_Gleason\images_test_no_divergence\02_resize\256\train\gleason3",
        data_dir=r"D:\01_LZY\01_Data\03_Pathology\05_Gleason\images_test_no_divergence_clshead_png\01_origin\train\benign",
        clip_denoised=False,
        batch_size=1,
        classifier_scale=1.0,
        eta=0.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())

    if type == 'source':
        new_defaults = dict(
            use_new_attention_order=True,
            class_cond=True,
            timestep_respacing="ddim100",
        )
    elif type == 'target':
        new_defaults = dict(
            use_new_attention_order=False,
            class_cond=True,
            timestep_respacing="ddim100",
        )
    else:
        new_defaults = defaults

    defaults.update(new_defaults)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Source domains. (0: benign, 1: gleason3, 2: gleason4, 3: gleason5)"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="1",
        help="Target domains. (0: benign, 1: gleason3, 2: gleason4, 3: gleason5)"
    )
    parser.add_argument(
        "--domain_name",
        type=list,
        default=["benign", "gleason3", "gleason4", "gleason5"],
        help="domain name list"
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    with th.cuda.device(0):
        main()
