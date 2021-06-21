#!/usr/bin/env bash

sudo python3 train.py \
	--name celebahqedge_ot \
	--dataset_mode celebahqedge \
	--dataroot '/home/fangneng.zfn/datasets/cvpr2021/CelebAMask-HQ' \
	--checkpoints_dir '/data/vdd/fangneng.zfn/SFERT7/checkpoints' \
	--correspondence 'ot' \
	--niter 30 \
	--niter_decay 30 \
	--which_perceptual 4_2 \
	--weight_perceptual 0.001 \
	--use_attention \
	--maskmix \
	--PONO \
	--PONO_C \
	--vgg_normal_correct \
	--fm_ratio 1.0 \
	--warp_bilinear \
	--warp_cycle_w 1 \
	--batchSize 4 \
	--gpu_ids 1 \
	--ot_weight \
#	--continue_train \
#	--which_epoch 20