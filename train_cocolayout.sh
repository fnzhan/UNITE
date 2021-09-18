#!/usr/bin/env bash

sudo python3 train.py \
	--name cocolayout \
	--dataset_mode cocolayout \
	--dataroot '/data/vdd/fangneng.zfn/datasets/COCO-Stuff' \
	--checkpoints_dir '/data/vdd/fangneng.zfn/UNITE/checkpoints' \
	--correspondence 'ot' \
	--display_freq 1000 \
	--novgg_featpair 0.0 \
	--niter 30 \
	--niter_decay 30 \
	--maskmix \
	--use_attention \
	--warp_mask_losstype direct \
	--weight_mask 100.0 \
	--PONO \
	--PONO_C \
	--ctx_w 0.1
	--batchSize 8 \
	--gpu_ids 1,3 \