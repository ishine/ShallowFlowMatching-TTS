# StableTTS-sfm

## Modifications
1. We use randomly initialized speaker embeddings for VCTK.
2. The initial 10% steps are warm-up steps.
3. We implement and use FP16 for training.

## Inference
For quick inference, you can download our pre-trained weights:
```bash
wget https://huggingface.co/ydqmkkx/SFM-models/resolve/main/StableTTS-sfm-vctk.pt
```
Then use the inference script `inference.ipynb`.

## Training
1. Change the paths in the txt files of the `data` folder.
2. Extract mels:
```bash
python preprocess.py
```
Then you can obtain the `VCTK` folder and `filelists/vctk.json`.

3. Start the training with single GPU:
```bash
python train.py
```
4. We use 2-node (1 GPU in each node) training with `mpirun`:
```bash
export MASTER_ADDR=your_MASTER_ADDR
export MASTER_PORT=12345

mpirun -hostfile your_hostfile \
  -np 2 \
  --npernode 1 \
  -bind-to none \
  -map-by node \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  python train_multi_node.py
```

## Configurations
Change `config.py`.

## SFM implementation
Our SFM implementations are mainly in
```python
models/text_encoder.py
models/flow_matching.py
models/model.py
```