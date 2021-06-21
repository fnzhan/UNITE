#!/usr/bin/env bash

sudo python3 test.py \
	--name celebahqedge \
	--dataset_mode celebahqedge \
	--dataroot '/home/fangneng.zfn/datasets/cvpr2021/CelebAMask-HQ' \
	--checkpoints_dir '/data/vdd/fangneng.zfn/SFERT7/checkpoints' \
	--correspondence 'ot' \
	--nThreads 0 \
	--use_attention \
	--maskmix \
	--PONO \
	--PONO_C \
	--warp_bilinear \
	--batchSize 2 \
	--gpu_ids 0 \
	--ot_weight \
#	--use_coordconv \
#	--how_many 10