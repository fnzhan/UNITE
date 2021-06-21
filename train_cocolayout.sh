#!/usr/bin/env bash

sudo python3 train.py \
	--name cocolayout \
	--dataset_mode cocolayout \
	--dataroot '/data/vdd/fangneng.zfn/datasets/COCO-Stuff' \
	--checkpoints_dir '/data/vdd/fangneng.zfn/SFERT8/checkpoints' \
	--correspondence 'ot' \
	--display_freq 1000 \
	--novgg_featpair 0.0 \
	--blur 0.01 \
	--niter 30 \
	--niter_decay 30 \
	--maskmix \
	--use_attention \
	--warp_mask_losstype direct \
	--weight_mask 100.0 \
	--PONO \
	--PONO_C \
	--batchSize 8 \
	--gpu_ids 1,3 \
	--continue_train \
#	--which_epoch 10