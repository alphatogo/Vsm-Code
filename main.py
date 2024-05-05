"""
Train a diffusion model on images.
"""
import pickle
import argparse
from model.set_diffusion import dist_util, logger
import os.path as osp
from dataset import create_loader
from model import select_model
from model.set_diffusion import logger
from model.set_diffusion.resample import create_named_schedule_sampler
from model.set_diffusion.script_util import (add_dict_to_argparser,
                                                args_to_dict,
                                                create_model_and_diffusion,
                                                model_and_diffusion_defaults)
from model.set_diffusion.train_util import TrainLoop
from utils.util import count_params, set_seed
from utils.path import set_folder
import torch
DIR=set_folder()

def main():
    args = create_argparser().parse_args()
    print()
    dct = vars(args)
    for k in sorted(dct):
        print(k, dct[k])
    print()
    
    # dist_util.setup_dist()
    logger.configure(dir=DIR, mode="training", args=args, tag='')

    logger.log("creating model and diffusion...")
    model = select_model(args)(args)

    print(count_params(model))
    model.to(args.device)

    # model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, model.diffusion
    )

    logger.log("creating data loader...")

    train_loader = create_loader(args, split="train", shuffle=True)
    # evaluation is expensive...perform it only when saving models
    val_loader = create_loader(args, split="test", shuffle=True)

    logger.log("training...")
    TrainLoop(
        model=model,
        data=train_loader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        val_loader=val_loader,
        args=args
    ).run_loop()


def create_argparser():
    defaults = dict(
        model='vfsddpm',
        dataset='celeba',
        image_size=64,
        sample_size=5,
        patch_size=4,
        hdim=256,
        in_channels=3,
        encoder_mode='unet',
        pool='mean_patch', # mean, mean_patch
        context_channels=256,
        mode_context="deterministic",
        mode_conditioning='lag', 
        augment=False,
        device="cuda",
        data_dir="./data",
        schedule_sampler="uniform",
        num_classes=1,
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=10,
        batch_size_eval=100,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.995",  # comma-separated list of EMA values
        log_interval=5000,
        save_interval=20000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        clip_denoised=True,
        use_ddim=False,
        tag=None,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    s = set_seed(0)
    main()
