# DiffSingerMiniEngine
A minimum inference engine for DiffSinger MIDI-less mode.

## Getting Started

1. Install `onnxruntime` following the [official guidance](https://onnxruntime.ai/). `pip install onnxruntime-gpu`
2. Install other dependencies with `pip install PyYAML soundfile`.
3. Download ONNX version of the NSF-HiFiGAN vocoder from [here](https://github.com/openvpi/vocoders/releases/tag/nsf-hifigan-v1) and unzip it into `assets/vocoder` directory.
4. Download an ONNX rhythm predictor from [here](https://github.com/openvpi/DiffSinger/releases/tag/v1.4.1) and put it into `assets/rhythmizer` directory.
5. Put your ONNX acoustic models into `assets/acoustic` directory.
6. Edit `configs/default.yaml` or create another config file according to your preference and local environment.
7. Run server with `python server.py` or `python server.py --config <YOUR_CONFIG>`.

## API Specification
* 版本信息

```
GET /version HTTP/1.1

HTTP/1.1 200 OK
{"version": "1.0.1", "date": "2023-01-08"}
```

* 模型列表
```
GET /models HTTP/1.1

HTTP/1.1 200 OK
Content-Type:application/json
{"models": ["1215_opencpop_ds1000_fix_label_nomidi"]}
```
* 生成节奏
```
POST /rhythm HTTP/1.1
Content-Type:application/json
{
    "notes":[
        {"key": 0,"duration": 0.5,"slur": false,"phonemes": ["SP"]},
        {"key": 69,"duration": 0.5,"slur": false,"phonemes": ["sh","a"]},
        {"key": 71,"duration": 1.0,"slur": true}
    ]
}

HTTP/1.1 200 OK
Content-Type:application/json
{"phonemes":[
    {"name": "SP", "duration": 0.235995352268219}, 
    {"name": "sh", "duration": 0.264004647731781}, 
    {"name": "a", "duration": 1.5}
]}
```

* 提交
```
POST /submit HTTP/1.1
Content-Type:application/json
{
    "model": "1215_opencpop_ds1000_fix_label_nomidi",
    "phonemes":[
        {"name": "SP", "duration": 0.235995352268219}, 
        {"name": "sh", "duration": 0.264004647731781}, 
        {"name": "a", "duration": 1.5}
    ],
    "f0":{
        "timestep": 0.01,
        "values": [440.0,440.0,440.0,440.0,440.0]
    },
    "speedup": 50
}

HTTP/1.1 200 OK
Content-Type:application/json
{
    "token": "afbc3057747f0cd98b67f01038855380",
    "status": "SUBMITTED",
    "code": "ae67"
}
```
* 查询
```
POST /query HTTP/1.1
Content-Type:application/json
{"token": "afbc3057747f0cd98b67f01038855380"}

HTTP/1.1 200 OK
Content-Type:application/json
{"status": "HIT_CACHE"}
```

* 取消任务
```
POST /cancel HTTP/1.1
Content-Type:application/json
{"token": "afbc3057747f0cd98b67f01038855380","code":"ae67"}
{"succeeded": false,"message": "Task result already in cache."}
```
* 下载文件
```
GET /download?token=afbc3057747f0cd98b67f01038855380 HTTP/1.1

HTTP/1.1 200 ok
content-type: audio/wav
```

## How to Obtain Acoustic Models

1. [Train with your own dataset](https://github.com/openvpi/DiffSinger/blob/refactor/pipelines/no_midi_preparation.ipynb) or download pretrained checkpoints from [here](https://github.com/openvpi/DiffSinger/releases/tag/v1.4.0).
2. Export PyTorch checkpoints to ONNX format. See instructions [here](https://github.com/openvpi/DiffSinger/blob/refactor/docs/README-SVS-onnx.md).
