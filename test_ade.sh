srun -p dsta --mpi=pmi2 \
--gres=gpu:1 -n1 \
--ntasks-per-node=1 \
--job-name=unite_ade20k \
--kill-on-bad-exit=1 \
-w SG-IDC1-10-51-2-37 \
python3 test.py \
	--name ade20k \
	--dataset_mode ade20k \
	--dataroot '/mnt/lustre/fnzhan/datasets/ICCV2021/ADEChallengeData2016/images' \
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
	--gpu_ids 0 \
#	--ot_weight \
#	--use_coordconv \
#	--how_many 10