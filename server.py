import argparse
import glob
import json
import logging
import os.path
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from http.server import HTTPServer, BaseHTTPRequestHandler

import soundfile

import synthesis
import utils

VERSION = '1.0.1'
DATE = '2023-01-08'
SERVER_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_ROOT = os.path.join(SERVER_ROOT, 'configs')
ACOUSTIC_ROOT = os.path.join(SERVER_ROOT, 'assets', 'acoustic')

logging.basicConfig(level='DEBUG',
                    format="%(asctime)s - %(levelname)-7s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")


def version(request: BaseHTTPRequestHandler):
    v = {
        'version': VERSION,
        'date': DATE
    }
    request.send_response(200)
    request.send_header('Content-Type', 'application/json')
    request.end_headers()
    request.wfile.write(json.dumps(v).encode('utf8'))


def models(request: BaseHTTPRequestHandler):
    res = {
        'models': [os.path.basename(file)[:-5] for file in glob.glob(os.path.join(ACOUSTIC_ROOT, '*.onnx'))]
    }
    request.send_response(200)
    request.send_header('Content-Type', 'application/json')
    request.end_headers()
    request.wfile.write(json.dumps(res).encode('utf8'))


def rhythm(request: BaseHTTPRequestHandler):
    """
    Example:
        {
          "notes": [
            {
              "key": 0,
              "duration": 0.5,
              "slur": false,
              "phonemes": [
                "SP"
              ]
            },
            {
              "key": 69,
              "duration": 0.5,
              "slur": false,
              "phonemes": [
                "sh",
                "a"
              ]
            },
            {
              "key": 71,
              "duration": 1.0,
              "slur": true
            }
          ]
        }
    """
    request_body = json.loads(request.rfile.read(int(request.headers['Content-Length'])))
    ph_seq, ph_dur = synthesis.predict_rhythm(request_body['notes'], phoneme_list, vowels, config)
    res = {
        'phonemes': [
            {
                'name': name,
                'duration': duration
            }
            for name, duration in zip(ph_seq, ph_dur)
        ]
    }
    request.send_response(200)
    request.send_header('Content-Type', 'application/json')
    request.end_headers()
    request.wfile.write(json.dumps(res).encode('utf8'))


def submit(request: BaseHTTPRequestHandler):
    """
    Example:
        {
          "model": "1215_opencpop_ds1000_fix_label_nomidi",
          "phonemes": [
            {
              "name": "SP",
              "duration": 0.5
            },
            {
              "name": "SP",
              "duration": 0.5
            }
          ],
          "f0": {
            "timestep": 0.01,
            "values": [
              440.0,
              440.0,
              440.0,
              440.0,
              440.0
            ]
          },
          "speedup": 50
        }
    """
    request_body = json.loads(request.rfile.read(int(request.headers['Content-Length'])))
    if 'speedup' not in request_body:
        request_body['speedup'] = config['acoustic']['speedup']
    token = utils.request_to_token(request_body)
    cache_file = os.path.join(cache, f'{token}.wav')
    if os.path.exists(cache_file):
        res = {
            'token': token,
            'status': 'HIT_CACHE'
        }
    else:
        mutex.acquire()
        code = utils.random_string(4)
        if token in tasks:
            piles[token].append(code)
        else:
            if token in failures:
                failures.pop(token)
            tasks[token] = pool.submit(_execute, request_body, cache_file, token)
            piles[token] = [code]
        mutex.release()
        res = {
            'token': token,
            'status': 'SUBMITTED',
            'code': code
        }
    request.send_response(200)
    request.send_header('Content-Type', 'application/json')
    request.end_headers()
    request.wfile.write(json.dumps(res).encode('utf8'))


def query(request: BaseHTTPRequestHandler):
    request_body = json.loads(request.rfile.read(int(request.headers['Content-Length'])))
    token = request_body['token']
    cache_file = os.path.join(cache, f'{token}.wav')
    if os.path.exists(cache_file):
        res = {
            'status': 'HIT_CACHE'
        }
        request.send_response(200)
        request.send_header('Content-Type', 'application/json')
        request.end_headers()
        request.wfile.write(json.dumps(res).encode('utf8'))
    else:
        mutex.acquire()
        if token in tasks:
            task = tasks[token]
            res = {}
            if task.cancelled():
                res['status'] = 'CANCELLED'
            elif task.done():
                if token in failures:
                    res['status'] = 'FAILED'
                    res['message'] = failures[token]
                else:
                    res['status'] = 'FINISHED'
            elif task.running():
                res['status'] = 'RUNNING'
            else:
                res['status'] = 'QUEUED'
            request.send_response(200)
            request.send_header('Content-Type', 'application/json')
            request.end_headers()
            request.wfile.write(json.dumps(res).encode('utf8'))
        elif token in failures:
            res = {
                'status': 'FAILED',
                'message': failures[token]
            }
            request.send_response(200)
            request.send_header('Content-Type', 'application/json')
            request.end_headers()
            request.wfile.write(json.dumps(res).encode('utf8'))
        else:
            request.send_error(404)
        mutex.release()


def cancel(request: BaseHTTPRequestHandler):
    request_body = json.loads(request.rfile.read(int(request.headers['Content-Length'])))
    token = request_body['token']
    code = request_body['code']
    mutex.acquire()
    if os.path.exists(os.path.join(cache, f'{token}.wav')):
        res = {
            'succeeded': False,
            'message': 'Task result already in cache.'
        }
    elif token not in tasks or code not in piles[token]:
        res = {
            'succeeded': False,
            'message': 'Invalid token or code.'
        }
    else:
        piles[token].remove(code)
        if len(piles[token]) == 0:
            tasks[token].cancel()
            tasks.pop(token)
            piles.pop(token)
        res = {
            'succeeded': True
        }
    mutex.release()
    request.send_response(200)
    request.send_header('Content-Type', 'application/json')
    request.end_headers()
    request.wfile.write(json.dumps(res).encode('utf8'))


def download(request: BaseHTTPRequestHandler):
    params = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(request.path).query))
    token = params['token']
    cache_file = os.path.join(cache, f'{token}.wav')
    if os.path.exists(cache_file):
        request.send_response(200)
        request.send_header('Content-Type', 'audio/wav')
        request.end_headers()
        with open(cache_file, 'rb') as f:
            request.wfile.write(f.read())
    else:
        request.send_response(404)


def _execute(request: dict, cache_file: str, token: str):
    logging.info(f'Task \'{token}\' begins')
    try:
        wav = synthesis.run_synthesis(
            request, phoneme_list,
            os.path.join(ACOUSTIC_ROOT, f'{request["model"]}.onnx'),
            config
        )
        os.makedirs(cache, exist_ok=True)
        soundfile.write(cache_file, wav, config['vocoder']['sample_rate'])
        logging.info(f'Task \'{token}\' finished')
    except Exception as e:
        failures[token] = str(e)
        logging.error(f'Task \'{token}\' failed')
        logging.error(str(e))
        raise e
    finally:
        mutex.acquire()
        tasks.pop(token)
        piles.pop(token)
        mutex.release()


config = {}
dictionary = {}
dict_pad = -1
phoneme_list = []
vowels = set()
vocoder_path = ''
cache = ''
pool: ThreadPoolExecutor
tasks = {}
piles = {}
failures = {}

apis = {
    '/version': (version, ['GET']),
    '/models': (models, ['GET']),
    '/rhythm': (rhythm, ['POST']),
    '/submit': (submit, ['POST']),
    '/query': (query, ['POST']),
    '/cancel': (cancel, ['POST']),
    '/download': (download, ['GET'])
}
mutex = threading.Lock()


class Request(BaseHTTPRequestHandler):
    def do_GET(self):
        url_split = urllib.parse.urlsplit(self.path)
        url_path = url_split.path.rstrip('/')
        if url_path not in apis:
            self.send_error(404)
        elif 'GET' not in apis[url_path][1]:
            self.send_error(405)
        else:
            apis[url_path][0](self)

    def do_POST(self):
        url_split = urllib.parse.urlsplit(self.path)
        url_path = url_split.path.rstrip('/')
        if url_path not in apis:
            self.send_error(404)
        elif 'POST' not in apis[url_path][1]:
            self.send_error(405)
        else:
            apis[url_path][0](self)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start DiffSinger inference server')
    parser.add_argument('--config', type=str, required=False, default='default')
    args = parser.parse_args()

    cfg_path = os.path.join(CONFIG_ROOT, f'{args.config}.yaml')
    config.update(utils.load_configs(cfg_path))
    logging.info(f'Using config from \'{cfg_path}\'')

    dict_path = config['dictionary']['filename']
    if not os.path.isabs(dict_path):
        dict_path = os.path.join(SERVER_ROOT, dict_path)
    dict_pad = config['dictionary']['reserved_tokens']
    dictionary.update(utils.load_dictionary(dict_path))
    vowels.update(utils.dictionary_to_vowels(dictionary))
    logging.info(f'Loaded dictionary from \'{dict_path}\'')

    phoneme_list.extend(utils.dictionary_to_phonemes(dictionary, dict_pad))
    vocoder_path = config['vocoder']['filename']
    if not os.path.isabs(vocoder_path):
        vocoder_path = os.path.join(SERVER_ROOT, vocoder_path)
    assert os.path.exists(vocoder_path), 'Vocoder model not found. Please check your configuration.'
    logging.info(f'Found vocoder at \'{vocoder_path}\'')

    cache = config['server']['cache_dir']
    if not os.path.isabs(cache):
        cache = os.path.join(SERVER_ROOT, cache)
    logging.info(f'Cache will be saved in \'{cache}\'')

    pool = ThreadPoolExecutor(max_workers=config['server']['max_threads'])

    host = ('127.0.0.1', config['server']['port'])
    with HTTPServer(host, Request) as server:
        logging.info('Server starting at %s:%s' % host)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        except InterruptedError:
            pass
