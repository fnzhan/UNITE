#!/usr/bin/env bash

sudo python3 test.py \
	--name cocolayout \
	--dataset_mode cocolayout \
	--dataroot '/data/vdd/fangneng.zfn/datasets/COCO-Stuff' \
	--checkpoints_dir '/data/vdd/fangneng.zfn/cocosnet_coco/checkpoints' \
	--correspondence 'ot' \
	--gpu_ids 1 \
	--nThreads 0 \
	--batchSize 1 \
	--use_attention \
	--maskmix \
	--warp_mask_losstype direct \
	--PONO \
	--PONO_C \
	--how_many 10 \
#	--ot_weight \
#	--use_coordconv \
