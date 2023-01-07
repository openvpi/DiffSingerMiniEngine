import numpy as np

import utils


def acoustic_preprocess(name2token: list,
                        phonemes: list,
                        durations: list,
                        f0: list,
                        frame_length: float,
                        f0_timestep: float):
    tokens = [name2token.index(ph) for ph in phonemes]
    tokens = np.array(tokens, dtype=np.int64)

    ph_dur = np.array(durations)
    ph_acc = np.around(np.add.accumulate(ph_dur) / frame_length + 0.5).astype(np.int64)
    ph_dur = np.diff(ph_acc, prepend=0)

    t_max = (len(f0) - 1) * f0_timestep
    f0_seq = np.interp(
        np.arange(0, t_max, frame_length, dtype=np.float32),
        f0_timestep * np.arange(len(f0), dtype=np.float32),
        np.array(f0, dtype=np.float32)
    ).astype(np.float32)
    required_length = ph_dur.sum()
    actual_length = f0_seq.shape[0]
    if actual_length > required_length:
        f0_seq = f0_seq[:required_length]
    elif actual_length < required_length:
        f0_seq = np.concatenate((f0_seq, np.full((required_length - actual_length,), fill_value=f0_seq[-1])))

    return tokens[None], ph_dur[None], f0_seq[None]


def acoustic_infer(model: str, providers: list, *, tokens, durations, f0, speedup):
    session = utils.create_session(model, providers)
    mel = session.run(['mel'], {'tokens': tokens, 'durations': durations, 'f0': f0, 'speedup': speedup})[0]
    return mel


def vocoder_infer(model: str, providers: list, mel, f0, force_on_cpu=True):
    session = utils.create_session(model, providers, force_on_cpu=force_on_cpu)
    waveform = session.run(['waveform'], {'mel': mel, 'f0': f0})[0]
    return waveform


def run_synthesis(request: dict, name2token: list, acoustic: str, vocoder: str, configs: dict):
    tokens, durations, f0 = acoustic_preprocess(
        name2token=name2token,
        phonemes=[ph['name'] for ph in request['phonemes']],
        durations=[ph['duration'] for ph in request['phonemes']],
        f0=request['f0']['values'],
        frame_length=configs['vocoder']['hop_size'] / configs['vocoder']['sample_rate'],
        f0_timestep=request['f0']['timestep']
    )
    speedup = np.array(request['speedup'], dtype=np.int64)
    mel = acoustic_infer(
        model=acoustic, providers=configs['providers'],
        tokens=tokens, durations=durations, f0=f0, speedup=speedup
    )
    waveform = vocoder_infer(
        model=vocoder, providers=configs['providers'], mel=mel, f0=f0,
        force_on_cpu=configs['vocoder']['force_on_cpu']
    )
    return waveform[0]
