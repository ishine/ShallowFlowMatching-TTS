export CUDA_VISIBLE_DEVICES="0"
dist_backend="nccl"
num_workers=4
prefetch=20
train_engine=torch_ddp
model=flow

torchrun --nnodes=1 --nproc_per_node=1\
  train.py \
  --train_engine $train_engine \
  --config configs/cosyvoice.yaml \
  --train_data examples/libritts/cosyvoice/data/train.data.list \
  --cv_data examples/libritts/cosyvoice/data/dev.data.list \
  --model $model \
  --checkpoint $pretrained_model_dir/$model.pt \
  --model_dir `pwd`/exp \
  --tensorboard_dir `pwd`/tensorboard \
  --ddp.dist_backend $dist_backend \
  --num_workers ${num_workers} \
  --prefetch ${prefetch} \
  --pin_memory \
  --use_amp
