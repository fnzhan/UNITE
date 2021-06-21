#!/usr/bin/env bash

sudo python3 test.py \
	--name coco \
	--dataset_mode coco \
	--dataroot '/data/vdd/fangneng.zfn/datasets/COCO-Stuff' \
	--checkpoints_dir '/data/vdd/fangneng.zfn/SFERT8/checkpoints' \
	--correspondence 'ot' \
	--blur 0.01 \
	--gpu_ids 0 \
	--nThreads 0 \
	--batchSize 1 \
	--use_attention \
	--maskmix \
	--warp_mask_losstype direct \
	--PONO \
	--PONO_C \
#	--ot_weight \
#	--use_coordconv \
#	--how_many 10