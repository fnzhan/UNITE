#!/usr/bin/env bash

sudo python3 test.py \
	--name deepfashion \
	--dataset_mode deepfashion \
	--dataroot '/home/fangneng.zfn/datasets/cvpr2021/DeepFashion' \
	--checkpoints_dir '/data/vdd/fangneng.zfn/SFERT7/checkpoints' \
	--correspondence 'ot' \
	--nThreads 0 \
	--use_attention \
	--PONO \
	--PONO_C \
	--warp_bilinear \
	--no_flip \
	--video_like \
	--ot_weight \
	--batchSize 2 \
	--gpu_ids 0 \
#	--how_many 10