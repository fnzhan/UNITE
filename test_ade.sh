#!/usr/bin/env bash

sudo python3 test.py \
	--name ade20k \
	--dataset_mode ade20k \
	--dataroot '/home/fangneng.zfn/datasets/cvpr2021/ADEChallengeData2016/images' \
	--checkpoints_dir '/data/vdd/fangneng.zfn/UNITE+/checkpoints' \
	--correspondence 'ot' \
	--nThreads 0 \
	--use_attention \
	--maskmix \
	--warp_mask_losstype direct \
	--PONO \
	--PONO_C \
	--use_coordconv \
	--adaptor_nonlocal \
	--batchSize 3 \
	--gpu_ids 4 \
#	--ot_weight \
#	--use_coordconv \
#	--how_many 10