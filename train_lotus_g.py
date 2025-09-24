#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from PIL import Image
from glob import glob
from easydict import EasyDict

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.nn as nn
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from pipeline import LotusGPipeline
from utils.image_utils import concatenate_images, colorize_depth_map
from utils.hypersim_dataset import get_hypersim_dataset_depth_normal
from utils.vkitti_dataset import VKITTIDataset, VKITTITransform, collate_fn_vkitti

from torchvision import transforms

from eval import evaluation_depth, evaluation_normal

import tensorboard

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__, log_level="INFO")
    
TOP5_STEPS_DEPTH = []
TOP5_STEPS_NORMAL = []

import cv2

from torchvision.transforms.functional import to_tensor, normalize
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DIS5K_Test_Dataset(Dataset):
    def __init__(self, data_dir):
        self.image_dir = os.path.join(data_dir, "im")
        self.gt_dir = os.path.join(data_dir, "gt")
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_name)
        
        base_name_no_ext = os.path.splitext(image_name)[0]
        gt_path = os.path.join(self.gt_dir, f"{base_name_no_ext}.png")
        
        image_pil = Image.open(image_path).convert("RGB")
        gt_mask_pil = Image.open(gt_path).convert("L")
            
        return image_pil, gt_mask_pil
    
def get_disk_kernel(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))

def compute_boundary_acc(gt, mask):
    gt = gt.astype(np.uint8)
    mask = mask.astype(np.uint8)

    h, w = gt.shape

    min_radius = 1
    max_radius = (w + h) / 300
    num_steps = 5

    mask_acc = np.zeros(num_steps)

    for i in range(num_steps):
        curr_radius = min_radius + int((max_radius - min_radius) / num_steps * i)

        kernel = get_disk_kernel(curr_radius)
        boundary_region = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, kernel) > 0

        gt_in_bound = gt[boundary_region]
        mask_in_bound = mask[boundary_region]

        num_edge_pixels = boundary_region.sum()
        num_mask_gd_pix = ((gt_in_bound) * (mask_in_bound) + (1 - gt_in_bound) * (1 - mask_in_bound)).sum()

        mask_acc[i] = num_mask_gd_pix / (num_edge_pixels + 1e-8)

    return mask_acc.mean()

def compute_iou(gt, mask):
    intersection = np.logical_and(gt, mask)
    union = np.logical_or(gt, mask)
    iou = np.sum(intersection) / (np.sum(union) + 1e-8)
    return iou
    
class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class IoU(AvgMeter):
    def compute_metric(self, pred, gt):
        pred_norm = pred.astype(np.float32) / 255.0 if pred.max() > 1.0 else pred.astype(np.float32)
        gt_norm = gt.astype(np.float32) / 255.0 if gt.max() > 1.0 else gt.astype(np.float32)

        pred_bin = pred_norm > 0.5
        gt_bin = gt_norm > 0.5

        return compute_iou(gt_bin, pred_bin)
    
class mBA(AvgMeter):
    def compute_metric(self, pred, gt):
        pred_norm = pred.astype(np.float32) / 255.0 if pred.max() > 1.0 else pred.astype(np.float32)
        gt_norm = gt.astype(np.float32) / 255.0 if gt.max() > 1.0 else gt.astype(np.float32)

        pred_bin = pred_norm > 0.5
        gt_bin = gt_norm > 0.5

        return compute_boundary_acc(gt_bin, pred_bin)

def run_example_validation(pipeline, task, args, step, accelerator, generator):
    validation_image_dir = "/vip_media/jinshan/Data/DIS5K/test/im"
    validation_gt_dir = "/vip_media/jinshan/Data/DIS5K/test/gt"

    validation_images = sorted(glob(os.path.join(validation_image_dir, "*.jpg")) + glob(os.path.join(validation_image_dir, "*.png")))
    # 为了防止验证时间过长，只取前5张作为可视化样本
    validation_images = validation_images[:5]

    pred_annos = []
    input_images = []
    gt_images = []
    
    if task == "depth":
        for i in range(len(validation_images)):
            if torch.backends.mps.is_available():
                autocast_ctx = nullcontext()
            else:
                autocast_ctx = torch.autocast(accelerator.device.type)

            with autocast_ctx:
                image_path = validation_images[i]
                # 根据图片名构造对应的GT路径
                base_name = os.path.basename(image_path)
                base_name_no_ext = os.path.splitext(os.path.basename(image_path))[0]
                gt_path = os.path.join(validation_gt_dir, f"{base_name_no_ext}.png")
                # gt_path = os.path.join(validation_gt_dir, base_name)
                gt_images.append(Image.open(gt_path).convert("L"))

                # Preprocess validation image 预处理数据：转换为RGB、归一化
                validation_image = Image.open(validation_images[i]).convert("RGB")
                input_images.append(validation_image)
                validation_image = np.array(validation_image).astype(np.float32)
                validation_image = torch.tensor(validation_image).permute(2,0,1).unsqueeze(0)
                validation_image = validation_image / 127.5 - 1.0 
                validation_image = validation_image.to(accelerator.device)

                # 执行预测深度图的任务
                task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(accelerator.device)
                task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)

                # Run # 预测
                pred_depth = pipeline(
                    rgb_in=validation_image, 
                    task_emb=task_emb,
                    prompt="", 
                    num_inference_steps=1, 
                    timesteps=[args.timestep],
                    generator=generator, 
                    output_type='np',
                    ).images[0]
                
                # Post-process the prediction
                pred_depth = pred_depth.mean(axis=-1) # 取平均值变成单通道

                pred_depth = (pred_depth > 0.5).astype(np.uint8)
                pred_depth_to_save = Image.fromarray(pred_depth * 255).convert("L")

                # is_reverse_color = "disparity" in args.norm_type
                # depth_color = colorize_depth_map(pred_depth, reverse_color=is_reverse_color)
                
                pred_annos.append(pred_depth_to_save)

    else:
        raise ValueError(f"Not Supported Task: {args.task_name}!")

    # Save output 保存
    save_output = concatenate_images(input_images, pred_annos, gt_images)
    save_dir = os.path.join(args.output_dir,'images')
    os.makedirs(save_dir, exist_ok=True)
    save_output.save(os.path.join(save_dir, f'{step:05d}.jpg'))

def run_evaluation(pipeline, task, args, step, accelerator, weight_dtype):
    if task == "depth" and step > 0:
        logger.info(f"Running EFFICIENT, BATCHED, DISTRIBUTED evaluation on DIS5K for step {step}...")
        
        metrics = {
            "iou": IoU(),
            "mba": mBA(),
        }

        eval_batch_size = 1
        
        test_dataset = DIS5K_Test_Dataset(data_dir="/vip_media/jinshan/Data/DIS5K/test/")
        # num_debug_samples = 20
        # debug_indices = range(min(num_debug_samples, len(test_dataset)))
        # test_dataset = Subset(test_dataset, debug_indices)
        
        def collate_fn_eval(batch):
            images, gt_masks = zip(*batch)
            return list(images), list(gt_masks)

        test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_eval)
        test_loader = accelerator.prepare(test_loader)
        
        local_metrics = {
            'iou_values': [],
            'mba_values': [],
            'sample_count': 0
        }

        for batch in tqdm(test_loader, desc=f"Evaluating DIS5K", disable=not accelerator.is_main_process):
            images_pil, gt_masks_pil = batch

            # 预处理
            processed_images = []
            for img in images_pil:
                img = transforms.functional.resize(img, (1024, 1024), interpolation=Image.BILINEAR) # resize一下，加速评估
                
                img_np = np.array(img).astype(np.float32)
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
                img_tensor = img_tensor / 127.5 - 1.0

                processed_images.append(img_tensor)
            images = torch.stack(processed_images).to(dtype=weight_dtype)

            # 推理
            autocast_ctx = torch.autocast(accelerator.device.type)
            with torch.no_grad(), autocast_ctx:
                batch_size = images.shape[0]
                task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(batch_size, 1).to(accelerator.device)
                task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)
                
                pred_masks_np = pipeline(
                    rgb_in=images,
                    task_emb=task_emb,
                    prompt="", 
                    num_inference_steps=1, 
                    timesteps=[args.timestep],
                    output_type='np',
                ).images
            
            # 后处理和立即计算指标
            pred_masks_binary = (pred_masks_np.mean(axis=-1) > 0.5).astype(np.uint8)
            
            for i, (pred_mask, mask_pil) in enumerate(zip(pred_masks_binary, gt_masks_pil)):
                mask_pil = transforms.functional.resize(mask_pil, (1024, 1024), interpolation=Image.NEAREST) # resize一下，加速评估
                gt_mask = np.array(mask_pil)
                if gt_mask.max() > 1:
                    gt_mask = (gt_mask > 127).astype(np.uint8)
                
                iou_val = metrics['iou'].compute_metric(pred_mask, gt_mask)
                mba_val = metrics['mba'].compute_metric(pred_mask, gt_mask)
                
                local_metrics['iou_values'].append(iou_val)
                local_metrics['mba_values'].append(mba_val)
                local_metrics['sample_count'] += 1

        # logger.info(f"Process {accelerator.process_index}: Processed {local_metrics['sample_count']} samples")

        # 同步所有进程
        accelerator.wait_for_everyone()

        local_iou_tensor = torch.tensor(local_metrics['iou_values'], device=accelerator.device)
        local_mba_tensor = torch.tensor(local_metrics['mba_values'], device=accelerator.device)
        local_count_tensor = torch.tensor([local_metrics['sample_count']], device=accelerator.device)

        # Gather
        gathered_ious = accelerator.gather_for_metrics(local_iou_tensor)
        gathered_mbas = accelerator.gather_for_metrics(local_mba_tensor)
        gathered_counts = accelerator.gather_for_metrics(local_count_tensor)
        
        if accelerator.is_main_process:
            total_samples = gathered_counts.sum().item()
            avg_iou = gathered_ious.mean().item()
            avg_mba = gathered_mbas.mean().item()
            
            logger.info(f"Total samples evaluated: {total_samples}")
            logger.info(f"----------------IoU: {avg_iou}------------------")
            logger.info(f"----------------mBA: {avg_mba}------------------")
            
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    tracker.writer.add_scalar("evaluation/IoU", avg_iou, step)
                    tracker.writer.add_scalar("evaluation/mBA", avg_mba, step)
                    

def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, step):
    logger.info("Running validation for task: %s... " % args.task_name[0])
    task = args.task_name[0]

    # Load pipeline # 加载模型
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    scheduler.register_to_config(prediction_type=args.prediction_type)
    pipeline = LotusGPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        scheduler=scheduler,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
  
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    
    
    # Run example-validation 可视化结果
    run_example_validation(pipeline, task, args, step, accelerator, generator)

    # Run evaluation 模型评估
    run_evaluation(pipeline, task, args, step, accelerator, weight_dtype)

    del pipeline
    torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir_hypersim",
        type=str,
        default=None,
        help=(
            "A folder containing the training data for hypersim"
        ),
    )
    parser.add_argument(
        "--train_data_dir_vkitti",
        type=str,
        default=None,
        help=(
            "A folder containing the training data for vkitti"
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--base_test_data_dir",
        type=str,
        default="datasets/eval/"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=["depth","normal"],
        nargs="+"
    )
    parser.add_argument(
        "--validation_images",
        type=str,
        default=None,
        help=("A set of images evaluated every `--validation_steps` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution_hypersim",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--resolution_vkitti",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--prob_hypersim",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--mix_dataset",
        action="store_true",
        help='Whether to mix the training data from hypersim and vkitti'
    )
    parser.add_argument(
        "--norm_type",
        type=str,
        choices=['instnorm','truncnorm','perscene_norm','disparity','trunc_disparity'],
        default='trunc_disparity',
        help='The normalization type for the depth prediction'
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--align_cam_normal",
        action="store_true",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--truncnorm_min",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=1
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="sample",
        help="The prediction_type that shall be used for training. ",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument("--FULL_EVALUATION", action="store_true")
    parser.add_argument("--save_pred_vis", action="store_true")
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_lotus_g",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    # if args.train_data_dir_hypersim is None and args.train_data_dir_vkitti is None:
    #     raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args

def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler") # DDPM
    noise_scheduler.register_to_config(prediction_type=args.prediction_type) # sample，即预测样本
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    ) # 文本分词器

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        ) # 文本编码器
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        ) # VAE
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision,
        class_embed_type="projection", projection_class_embeddings_input_dim=4,
        low_cpu_mem_usage=False, device_map=None,
    ) # U-Net
    
    # Replace the first layer to accept 8 in_channels. 
    # 改造U-Net输入使其适配8通道
    _weight = unet.conv_in.weight.clone()
    _bias = unet.conv_in.bias.clone()
    _weight = _weight.repeat(1, 2, 1, 1) 
    _weight *= 0.5
    # unet.config.in_channels *= 2
    config_dict = EasyDict(unet.config)
    config_dict.in_channels *= 2
    unet._internal_dict = config_dict

    # new conv_in channel
    _n_convin_out_channel = unet.conv_in.out_channels
    _new_conv_in =nn.Conv2d(
        8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    )
    _new_conv_in.weight = nn.Parameter(_weight)
    _new_conv_in.bias = nn.Parameter(_bias)
    unet.conv_in = _new_conv_in

    # Freeze vae and text_encoder and set unet to trainable
    # 冻结VAE、文本编码器，训练U-Net
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # 性能优化配置
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet", in_channels=8)
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # 梯度检查点
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets and dataloaders.
    # -------------------- Dataset1: Hypersim --------------------
    train_hypersim_dataset, preprocess_train_hypersim, collate_fn_hypersim = get_hypersim_dataset_depth_normal(
        args.train_data_dir_hypersim, args.resolution_hypersim, args.random_flip, 
        norm_type=args.norm_type, truncnorm_min=args.truncnorm_min, align_cam_normal=args.align_cam_normal
        ) # /vip_media/jinshan/Data/DIS5K 576 True trunc_disparity 0.02 False
    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            train_hypersim_dataset = train_hypersim_dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset_hypersim = train_hypersim_dataset.with_transform(preprocess_train_hypersim)

    train_dataloader_hypersim = torch.utils.data.DataLoader(
        train_dataset_hypersim,
        shuffle=True,
        collate_fn=collate_fn_hypersim,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True
    )
    # # -------------------- Dataset2: VKITTI --------------------
    # transform_vkitti = VKITTITransform(random_flip=args.random_flip)
    # train_dataset_vkitti = VKITTIDataset(args.train_data_dir_vkitti, transform_vkitti, args.norm_type, truncnorm_min=args.truncnorm_min)
    # train_dataloader_vkitti = torch.utils.data.DataLoader(
    #     train_dataset_vkitti, 
    #     shuffle=True,
    #     collate_fn=collate_fn_vkitti,
    #     batch_size=args.train_batch_size, 
    #     num_workers=args.dataloader_num_workers,
    #     pin_memory=True
    #     )
    
    # Lr_scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader_hypersim) / args.gradient_accumulation_steps)
    assert args.max_train_steps is not None or args.num_train_epochs is not None, "max_train_steps or num_train_epochs should be provided"
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    # 打包所有配置
    unet, optimizer, train_dataloader_hypersim, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader_hypersim, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader_hypersim) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("task_name")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples Hypersim = {len(train_dataset_hypersim)}")
    # logger.info(f"  Num examples VKITTI = {len(train_dataset_vkitti)}")
    logger.info(f"  Using mix datasets: {args.mix_dataset}")
    logger.info(f"  Dataset alternation probability of Hypersim = {args.prob_hypersim}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Unet timestep = {args.timestep}")
    logger.info(f"  Task name: {args.task_name}")
    logger.info(f"  Is Full Evaluation?: {args.FULL_EVALUATION}")
    logger.info(f"Output Workspace: {args.output_dir}")

    global_step = 0
    first_epoch = 0

    # 加载检查点信息
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    if args.validation_images is not None:
        log_validation(
            vae,
            text_encoder,
            tokenizer,
            unet,
            args,
            accelerator,
            weight_dtype,
            global_step,
        )
            
    for epoch in range(first_epoch, args.num_train_epochs):
        iter_hypersim = iter(train_dataloader_hypersim) # 两个数据集
        # iter_vkitti = iter(train_dataloader_vkitti)

        train_loss = 0.0
        log_ann_loss = 0.0
        log_rgb_loss = 0.0

        for _ in range(len(train_dataloader_hypersim)):
            batch = next(iter_hypersim)

            with accelerator.accumulate(unet):
                # Convert images to latent space # 编码彩色图
                rgb_latents = vae.encode(
                    torch.cat((batch["pixel_values"],batch["pixel_values"]), dim=0).to(weight_dtype)
                    ).latent_dist.sample()
                rgb_latents = rgb_latents * vae.config.scaling_factor
                # Convert target_annotations to latent space
                assert len(args.task_name) == 1
                if args.task_name[0] == "depth":
                    TAR_ANNO = "depth_values"
                elif args.task_name[0] == "normal":
                    TAR_ANNO = "normal_values"
                else:
                    raise ValueError(f"Do not support {args.task_name[0]} yet. ")
                target_latents = vae.encode(
                    torch.cat((batch[TAR_ANNO],batch["pixel_values"]), dim=0).to(weight_dtype)
                    ).latent_dist.sample() # 拼接了目标深度图和彩色图
                target_latents = target_latents * vae.config.scaling_factor
                
                bsz = target_latents.shape[0]
                bsz_per_task = int(bsz/2)

                # Get the valid mask for the latent space # 获取有效掩码
                valid_mask_for_latent = batch.get("valid_mask_values", None)
                if args.task_name[0] == "depth" and valid_mask_for_latent is not None:
                    sky_mask_for_latent = batch.get("sky_mask_values", None)
                    valid_mask_for_latent = valid_mask_for_latent + sky_mask_for_latent
                if valid_mask_for_latent is not None:
                    valid_mask_for_latent = valid_mask_for_latent.bool()
                    invalid_mask = ~valid_mask_for_latent
                    valid_mask_down_anno = ~torch.max_pool2d(invalid_mask.float(), 8, 8).bool()
                    valid_mask_down_anno = valid_mask_down_anno.repeat((1, 4, 1, 1))
                else:
                    valid_mask_down_anno = torch.ones_like(target_latents[:bsz_per_task]).to(target_latents.device).bool()
                
                valid_mask_down_rgb = torch.ones_like(target_latents[bsz_per_task:]).to(target_latents.device).bool()

                # Sample noise that we'll add to the latents # 获取噪声
                noise = torch.randn_like(target_latents)
                
                # 处理噪声
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (target_latents.shape[0], target_latents.shape[1], 1, 1), device=target_latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                
                # Set timestep
                timesteps = torch.tensor([args.timestep], device=target_latents.device).repeat(bsz)
                timesteps = timesteps.long()
                
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                # 加噪过程，注意，同时往彩色图和目标深度图加噪，即两个任务：生成噪声图、重建原始图
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(target_latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)

                # Concatenate rgb and depth
                unet_input = torch.cat(
                    [rgb_latents, noisy_latents], dim=1
                )

                # Get the empty text embedding for conditioning # 空的文本prompt，并编码
                prompt = ""
                text_inputs = tokenizer(
                    prompt,
                    padding="do_not_pad",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(target_latents.device)
                encoder_hidden_states = text_encoder(text_input_ids, return_dict=False)[0]
                encoder_hidden_states = encoder_hidden_states.repeat(bsz, 1, 1)

                # Get the target for loss
                target = target_latents

                # Get the task embedding
                task_emb_anno = torch.tensor([1, 0]).float().unsqueeze(0).to(accelerator.device)
                task_emb_anno = torch.cat([torch.sin(task_emb_anno), torch.cos(task_emb_anno)], dim=-1).repeat(bsz_per_task, 1)
                task_emb_rgb = torch.tensor([0, 1]).float().unsqueeze(0).to(accelerator.device)
                task_emb_rgb = torch.cat([torch.sin(task_emb_rgb), torch.cos(task_emb_rgb)], dim=-1).repeat(bsz_per_task, 1)
                task_emb = torch.cat((task_emb_anno, task_emb_rgb), dim=0)

                # Predict # 单步去噪
                model_pred = unet(unet_input, timesteps, encoder_hidden_states, return_dict=False,
                                class_labels=task_emb)[0]

                # Compute loss # 预测原图来计算损失而不是预测噪声
                anno_loss = F.mse_loss(model_pred[:bsz_per_task][valid_mask_down_anno].float(), target[:bsz_per_task][valid_mask_down_anno].float(), reduction="mean")
                rgb_loss = F.mse_loss(model_pred[bsz_per_task:][valid_mask_down_rgb].float(), target[bsz_per_task:][valid_mask_down_rgb].float(), reduction="mean")
                loss = anno_loss + rgb_loss

                # Gather loss
                avg_anno_loss = accelerator.gather(anno_loss.repeat(args.train_batch_size)).mean()
                log_ann_loss += avg_anno_loss.item() / args.gradient_accumulation_steps
                avg_rgb_loss = accelerator.gather(rgb_loss.repeat(args.train_batch_size)).mean()
                log_rgb_loss += avg_rgb_loss.item() / args.gradient_accumulation_steps
                train_loss = log_ann_loss + log_rgb_loss

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            logs = {"SL": loss.detach().item(), 
                    "SL_A": anno_loss.detach().item(), 
                    "SL_R": rgb_loss.detach().item(), 
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
        
            # Checks if the accelerator has performed an optimization step behind the scenes
            # 是否完成梯度积累
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss,
                                "anno_loss": log_ann_loss,
                                "rgb_loss": log_rgb_loss},
                                 step=global_step)
                train_loss = 0.0
                log_ann_loss = 0.0
                log_rgb_loss = 0.0

                checkpointing_steps = args.checkpointing_steps
                validation_steps = args.validation_steps
                
                if accelerator.is_main_process:
                    if global_step % checkpointing_steps == 0: # 保存
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                
                if global_step % validation_steps == 0:
                    log_validation(
                        vae,
                        text_encoder,
                        tokenizer,
                        unet,
                        args,
                        accelerator,
                        weight_dtype,
                        global_step,
                    )

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)

        pipeline = LotusGPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
