# CosyVoice-DiT-sfm

## Preparations
The preprocessed LibriTTS data of CosyVoice is needed. \
You can follow the official instructions of CosyVoice or download the data by
```bash
wget https://huggingface.co/ydqmkkx/SFM-models/resolve/main/CosyVoice-libritts-data.zip
unzip CosyVoice-libritts-data.zip
```
We only keep the necessary files for the training and inference of CosyVoice-DiT-sfm.

## Inference
For quick inference, you can download our pre-trained weights:
```bash
wget https://huggingface.co/ydqmkkx/SFM-models/resolve/main/CosyVoice-DiT-sfm-libritts.pt
```
Then use the inference script `inference.ipynb`.
For
```python
for wav in wavs:
    wavs_dict[wav.split(" ")[0]] = wav.split(" ", 1)[1].strip("\n").replace("xxx", "your LibriTTS wav path")
```
You need to fill in the wav path of LibriTTS. You can see a `wav.scp` in a folder of `CosyVoice-libritts-data` for a better understanding.

## Training
1. Change the paths in `filelist_generator.py` line 11 and run
```bash
python filelist_generator.py
```
2. Extract mels:
```bash
python preprocess.py
```
Then you can obtain the `LibriTTS` folder and `filelists/filelist.json`.

3. Start the training with single GPU:
```bash
python train.py
```
4. We use 4-node (1 GPU in each node) training with `mpirun`:
```bash
export MASTER_ADDR=your_MASTER_ADDR
export MASTER_PORT=12345

mpirun -hostfile your_hostfile \
  -np 4 \
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