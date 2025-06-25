dist_backend="nccl"
num_workers=8
prefetch=20
train_engine=torch_ddp
model=flow

export MASTER_ADDR=$(hostname -I | awk '{print $2}')
export MASTER_PORT=12345

export PATH=/work/opt/local/aarch64/cores/cuda/12.6/bin:$PATH
export LD_LIBRARY_PATH=/work/opt/local/aarch64/cores/cuda/12.6/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/work/opt/local/aarch64/cores/cuda/12.6
unset OMPI_MCA_mca_base_env_list

mpirun -hostfile $PBS_NODEFILE \
  -np 8 \
  --npernode 1 \
  -bind-to none \
  -map-by node \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x PATH -x LD_LIBRARY_PATH -x CUDA_HOME \
  python train_multi_node.py \
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