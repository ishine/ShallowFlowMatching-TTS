# CosyVoice-sfm

## Environment
Please follow the official implementation of [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) to install the environment. \
The package of Matcha-TTS is needed, we think instead of using
```python
import sys
sys.path.append('third_party/Matcha-TTS')
```
in the [CosyVoice implementation](https://github.com/FunAudioLLM/CosyVoice?tab=readme-ov-file#basic-usage), installing the official Matcha-TTS package directly is more convenient:
```bash
pip install matcha-tts
```
Please note that this will update the version of Pytorch. You may need to install Pytorch again with the version that CosyVoice uses. \
Then,
```bash
pip install torchdiffeq
```

## Quick Inference
1. Download the official pre-trained model for its pre-trained tokenizer, LLM, and HiFi-GAN modules:
```bash
mkdir -p pretrained_models
cd pretrained_models
git clone https://huggingface.co/model-scope/CosyVoice-300M
```
2. Download our pre-trained flow module:
```bash
wget https://huggingface.co/ydqmkkx/SFM-models/resolve/main/CosyVoice-sfm-epoch_199_step_200201.pt
```
3. Download the preprocessed LibriTTS data, which only keeps the necessary files for the inference:
```bash
cd ..
wget https://huggingface.co/ydqmkkx/SFM-models/resolve/main/CosyVoice-libritts-data.zip
unzip CosyVoice-libritts-data.zip
```
4. Use the inference script `synthesis.ipynb`.
For
```python
for wav in wavs:
    wavs_dict[wav.split(" ")[0]] = wav.split(" ", 1)[1].strip("\n").replace("xxx", "your LibriTTS wav path")
```
You need to fill in the wav path of LibriTTS. You can see a `wav.scp` in a folder of `CosyVoice-libritts-data` for a better understanding.

## Training
1. Clone the [`examples`](https://github.com/FunAudioLLM/CosyVoice/tree/main/examples) folder and preprocess the LibriTTS dataset following `examples/libritts/cosyvoice/run.sh`. Then you can get the `examples/libritts/cosyvoice/data` folder.
2. Start the training with single GPU:
```bash
./run.sh
```
- We use a large dynamic batch size. If the GPU memory is limited, please reduce the `max_frames_in_batch` in `configs/cosyvoice.yaml`.
- We store the weights every 20 epochs, which can be changed in `cosyvoice/utils/executor.py` line 84.
3. We use 8-node (1 GPU in each node) training with `mpirun` and the training script is `run_multi_node.sh`.
- The official CosyVoice code is still updating now. For the code we have used, dataset loading can cause some problems when using DDP (multi-GPU) training, which can be found in an [issue](https://github.com/FunAudioLLM/CosyVoice/issues/297).
- Therefore, we add `cosyvoice/utils/executor.py` line 50-51 to solve this problem, which drops the last part of the data in each epoch. Therefore, for DDP training, please uncomment the two lines. The number 1000 is determind by the 24000 dynamic batch size and the size of LibriTTS dataset, so for other batch sizes and datasets, you need to calculate the total batches in one epoch and change the number.

## SFM implementation
Our SFM implementations are mainly in `cosyvoice/flow`.