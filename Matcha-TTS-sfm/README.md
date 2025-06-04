# Matcha-TTS-sfm

## Modifications
1. We use Vocos but not HiFi-GAN as the vocoder. Thus, the `data_statistics` in `configs/data/ljspeech.yaml` and `configs/data/vctk.yaml` is not needed and set as default values.
2. To accelerate the training, in the first training loop, the extracted mel will be saved in the dataset folders:
`matcha/data/text_mel_datamodule.py`, line 199.
3. If the training freezes initially, you may need to set the `num_workers` as 0 in `configs/data/ljspeech.yaml` or `configs/data/vctk.yaml`. After the first training epoch (all mels are extracted and saved), you can use multiple `num_workers` and start a new training.
4. Although we follow the official implementation and do not use a learning rate scheduler, we provide a linear scheduler implementation: `matcha/models/baselightningmodule.py`, line 37.

## Inference
For quick inference, you can download our pre-trained weights:
```bash
cd logs
wget https://huggingface.co/ydqmkkx/SFM-models/resolve/main/MatchaTTS-sfm-ljspeech.ckpt
wget https://huggingface.co/ydqmkkx/SFM-models/resolve/main/MatchaTTS-sfm-vctk.ckpt
```
Then use the inference script `synthesis.ipynb`.

## Training
1. Change the paths in the txt files of the `data` folder.
2. Start the training with single GPU:
```bash
python matcha/train.py experiment=ljspeech
or
python matcha/train.py experiment=vctk
```

## Configurations
1. Change batch_size or num_workers: `configs/data`
2. Change model size or optimizer (learning rate): `configs/model`
3. For multi-GPU or multi-node training, change the `configs/train.yaml/defaults/trainer` (e.g., `default`->`ddp`), and then change and use `configs/trainer/ddp.yaml` or others.

## SFM implementation
Our SFM implementations are mainly in
```python
matcha/models/components/text_encoder.py
matcha/models/components/flow_matching.py
matcha/models/matcha_tts.py
```