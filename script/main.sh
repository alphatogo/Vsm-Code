#! /bin/sh

GPU=$1
RUN=$2

run()
{

IMAGE_SIZE=$1
IN_CHANNELS=$2
MODEL=$3
DATASET=$4
ENCODER=$5
CONDITIONING=$6
POOLING=$7
CONTEXT=$8
PATCH_SIZE=$9
SAMPLE_SIZE=$10
BATCH_SIZE=$11
BATCH_SIZE_EVAL=$12


MODEL_FLAGS="--image_size ${IMAGE_SIZE} --in_channels ${IN_CHANNELS} --num_channels 64 
--context_channels 256 --dropout 0.2 --num_res_blocks 2 --model ${MODEL} --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size ${BATCH_SIZE} --batch_size_eval ${BATCH_SIZE_EVAL} --dataset ${DATASET} --ema_rate 0.995"
ENCODER_FLAGS="--patch_size ${PATCH_SIZE} --encoder_mode ${ENCODER} --sample_size ${SAMPLE_SIZE} 
--mode_conditioning ${CONDITIONING} --pool ${POOLING} --mode_context ${CONTEXT}"

CUDA_VISIBLE_DEVICES=$GPU \
python main.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $ENCODER_FLAGS

}

# Run config
case $RUN in
    vfsddpm_celeba_unet_film_mean_sigma)
      run 64 3 vfsddpm celeba unet film mean deterministic 16 5 10 200
      ;;
esac

