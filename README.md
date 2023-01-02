# DiffSingerMiniEngine
A minimum inference engine for DiffSinger MIDI-less mode.

## Getting Started

1. Install `onnxruntime` following the [official guidance](https://onnxruntime.ai/).
2. Install other dependencies with `pip install PyYAML soundfile`.
3. Download ONNX version of the NSF-HiFiGAN vocoder from [here](https://github.com/openvpi/vocoders/releases/tag/nsf-hifigan-v1) and unzip it into `assets/vocoder` directory.
4. Put your ONNX acoustic models into `assets/acoustic` directory.
5. Edit `configs/default.yaml` or create another config file according to your preference and local environment.
6. Run server with `python server.py`.

## API Specification

TBD

## How to Obtain Acoustic Models

1. [Train with your own dataset](https://github.com/openvpi/DiffSinger/blob/refactor/pipelines/no_midi_preparation.ipynb) or download pretrained checkpoints from [here](https://github.com/openvpi/DiffSinger/releases/tag/v1.4.0).
2. Export PyTorch checkpoints to ONNX format. See instructions [here](https://github.com/openvpi/DiffSinger/blob/refactor/docs/README-SVS-onnx.md).
