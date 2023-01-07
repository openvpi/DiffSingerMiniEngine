import hashlib
import json
import logging
import os
import random

import onnxruntime as ort
import yaml

_dll_loaded = False


class ProviderError(Exception):
    def __init__(self, *args):
        super().__init__(*args)


def load_configs(path: str) -> dict:
    with open(path, 'r', encoding='utf8') as f:
        return yaml.safe_load(f)


def create_session(model_path: str, providers: list, force_on_cpu: bool = False) -> ort.InferenceSession:
    global _dll_loaded

    available_providers_selected = []
    if not force_on_cpu:
        for provider in providers:
            if provider['name'] in ort.get_available_providers():
                available_providers_selected.append(provider)
            elif len(available_providers_selected) == 0:
                logging.warning(f'{provider["name"]} is not available on this machine. Skipping.')
    else:
        available_providers_selected.append({
            'name': 'CPUExecutionProvider',
            'options': {}
        })

    if not available_providers_selected:
        raise ProviderError('None of the selected execution providers is available on this machine.')
    providers = [(provider['name'], provider['options']) for provider in available_providers_selected]

    # Create session options
    options = ort.SessionOptions()
    if available_providers_selected[0]['name'] == 'DmlExecutionProvider':
        # DirectML does not support memory pattern optimizations or parallel execution in onnxruntime. See
        # https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html#configuration-options
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    elif available_providers_selected[0]['name'] == 'CUDAExecutionProvider' and not _dll_loaded:
        # CUDA_PATH may break in virtual environments.
        # Add CUDA and cuDNN libraries to DLL directories manually.
        attr = available_providers_selected[0].get('attributes')
        if attr is not None:
            cuda_path = available_providers_selected[0]['attributes'].get('cuda_path')
            cudnn_path = available_providers_selected[0]['attributes'].get('cudnn_path')
            if cuda_path is not None and os.path.exists(cuda_path):
                os.add_dll_directory(cuda_path)
            if cudnn_path is not None and os.path.exists(cudnn_path):
                os.add_dll_directory(cudnn_path)
        _dll_loaded = True

    # Create inference session
    session = ort.InferenceSession(path_or_bytes=model_path, sess_options=options, providers=providers)

    return session


def load_dictionary(path: str) -> dict:
    with open(path, 'r', encoding='utf8') as f:
        rules = [ln.strip().split('\t') for ln in f.readlines()]
    return {r[0]: r[1].split() for r in rules}


def dictionary_to_phonemes(dictionary: dict, pad: int) -> list:
    phonemes = {'AP', 'SP'}
    [phonemes.update(seq) for seq in dictionary.values()]
    return [None for _ in range(pad)] + sorted(phonemes)


def request_to_token(request: dict) -> str:
    req_str = json.dumps(request, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(req_str.encode(encoding='utf-8')).hexdigest()


def random_string(length: int) -> str:
    chars = '0123456789abcdef'
    return ''.join(random.choice(chars) for _ in range(length))
