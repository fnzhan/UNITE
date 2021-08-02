srun -p dsta --mpi=pmi2 \
--gres=gpu:1 -n1 \
--ntasks-per-node=1 \
--job-name=tip_ade20k \
--kill-on-bad-exit=1 \
-w SG-IDC1-10-51-2-37 \
python3 train.py \
	--name ade20k \
	--dataset_mode ade20k \
	--dataroot '/mnt/lustre/fnzhan/datasets/ICCV2021/ADEChallengeData2016/images' \
	--correspondence 'ot' \
	--display_freq 1000 \
	--niter 150 \
	--niter_decay 150 \
	--maskmix \
	--use_attention \
	--warp_mask_losstype direct \
	--weight_mask 100.0 \
	--PONO \
	--PONO_C \
	--use_coordconv \
	--adaptor_nonlocal \
	--ctx_w 1.0 \
	--batchSize 3 \
	--gpu_ids 0 \
	--continue_train
#	--which_epoch 10
#	--checkpoints_dir '/data/vdd/fangneng.zfn/UNITE+/checkpoints' \
#-o file.out -e file.err
