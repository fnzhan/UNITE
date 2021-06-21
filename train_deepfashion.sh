#!/usr/bin/env bash

sudo python3 train.py \
	--name deepfashion \
	--dataset_mode deepfashion \
	--dataroot '/home/fangneng.zfn/datasets/DeepFashion' \
	--checkpoints_dir '/data/vdd/fangneng.zfn/SFERT7/checkpoints' \
	--correspondence 'ot' \
	--niter 50 \
	--niter_decay 50 \
	--which_perceptual 4_2 \
	--weight_perceptual 0.001 \
	--use_attention \
	--real_reference_probability 0.0 \
	--PONO \
	--PONO_C \
	--vgg_normal_correct \
	--fm_ratio 1.0 \
	--warp_bilinear \
	--warp_self_w 100 \
	--no_flip \
	--video_like \
	--batchSize 2 \
	--gpu_ids 1 \
	--ot_weight \
	--continue_train \
#	--which_epoch 20
