import numpy as np

import utils


def rhythm_preprocess(name2token: list,
                      words: list,
                      midi: list,
                      midi_dur: list,
                      is_slur: list):
    tokens = []
    midi_seq = []
    midi_dur_seq = []
    is_slur_seq = []
    last_phoneme = 'SP'
    for i in range(len(words)):
        if not is_slur[i]:
            for ph in words[i]:
                tokens.append(name2token.index(ph))
                midi_seq.append(midi[i])
                midi_dur_seq.append(midi_dur[i])
                is_slur_seq.append(False)
            last_phoneme = words[i][-1]
        else:
            tokens.append(name2token.index(last_phoneme))
            midi_seq.append(midi[i])
            midi_dur_seq.append(midi_dur[i])
            is_slur_seq.append(True)
    tokens = np.array(tokens, dtype=np.int64)
    midi_seq = np.array(midi_seq, dtype=np.int64)
    midi_dur_seq = np.array(midi_dur_seq, dtype=np.float32)
    is_slur_seq = np.array(is_slur_seq, dtype=np.bool_)
    return tokens[None], midi_seq[None], midi_dur_seq[None], is_slur_seq[None]


def rhythm_infer(model: str, providers: list, tokens, midi, midi_dur, is_slur):
    session = utils.create_session(model, providers)
    ph_dur = session.run(['ph_dur'], {'tokens': tokens, 'midi': midi, 'midi_dur': midi_dur, 'is_slur': is_slur})[0]
    return ph_dur


def rhythm_postprocess(ph_seq, midi_dur, ph_dur, all_vowels):
    for i in range(len(ph_dur)):
        if ph_seq[i] in all_vowels:
            if i < len(ph_dur) - 1 and ph_seq[i + 1] not in all_vowels:
                ph_dur[i] = midi_dur[i] - ph_dur[i + 1]
                if ph_dur[i] < 0:
                    ph_dur[i] = 0
                    ph_dur[i + 1] = midi_dur[i]
            else:
                ph_dur[i] = midi_dur[i]


def predict_rhythm(notes: list, name2token: list, all_vowels: set, configs: dict):
    tokens, midi_seq, midi_dur_seq, is_slur_seq = rhythm_preprocess(
        name2token=name2token,
        words=[note.get('phonemes') for note in notes],
        midi=[note['key'] for note in notes],
        midi_dur=[note['duration'] for note in notes],
        is_slur=[note['slur'] for note in notes]
    )
    ph_dur = rhythm_infer(
        model=configs['rhythmizer']['filename'], providers=configs['providers'],
        tokens=tokens, midi=midi_seq, midi_dur=midi_dur_seq, is_slur=is_slur_seq
    )
    ph_seq = [name2token[tok] for tok in tokens[0].tolist()]
    midi_seq = midi_seq[0].tolist()
    midi_dur_seq = midi_dur_seq[0].tolist()
    is_slur_seq = is_slur_seq[0].tolist()
    ph_dur = ph_dur[0].tolist()
    rhythm_postprocess(
        ph_seq=ph_seq, midi_dur=midi_dur_seq,
        ph_dur=ph_dur, all_vowels=all_vowels
    )
    i = 0
    while i < len(ph_seq):
        if is_slur_seq[i]:
            ph_dur[i - 1] += ph_dur[i]
            ph_seq.pop(i)
            midi_seq.pop(i)
            midi_dur_seq.pop(i)
            is_slur_seq.pop(i)
            ph_dur.pop(i)
        else:
            i += 1
    return ph_seq, ph_dur


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


def acoustic_infer(model: str, providers: list, tokens, durations, f0, speedup):
    session = utils.create_session(model, providers)
    mel = session.run(['mel'], {'tokens': tokens, 'durations': durations, 'f0': f0, 'speedup': speedup})[0]
    return mel


def vocoder_infer(model: str, providers: list, mel, f0, force_on_cpu=True):
    session = utils.create_session(model, providers, force_on_cpu=force_on_cpu)
    waveform = session.run(['waveform'], {'mel': mel, 'f0': f0})[0]
    return waveform


def run_synthesis(request: dict, name2token: list, acoustic: str, configs: dict):
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
        model=configs['vocoder']['filename'], providers=configs['providers'], mel=mel, f0=f0,
        force_on_cpu=configs['vocoder']['force_on_cpu']
    )
    return waveform[0]
