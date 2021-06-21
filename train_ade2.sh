#!/usr/bin/env bash

sudo python3 train.py \
	--name ade20k2 \
	--dataset_mode ade20k \
	--dataroot '/home/fangneng.zfn/datasets/cvpr2021/ADEChallengeData2016/images' \
	--checkpoints_dir '/data/vdd/fangneng.zfn/UNITE+/checkpoints' \
	--correspondence 'ot' \
	--display_freq 1000 \
	--niter 100 \
	--niter_decay 100 \
	--maskmix \
	--use_attention \
	--warp_mask_losstype direct \
	--weight_mask 100.0 \
	--PONO \
	--PONO_C \
	--adaptor_nonlocal \
	--batchSize 6 \
	--gpu_ids 2,3 \
#	--continue_train \
#	--which_epoch 10