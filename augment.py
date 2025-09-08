import os
from copy import deepcopy

import numpy as np
import librosa
from scipy.io import wavfile
from scipy import signal
import soundfile

from mydatasets import read_wav

np.random.seed(607)


def augment_with_channel(data_path, musan_path, rirs_path, lines_path,
                         clip_signal=True, amplitude_factor=0.2, preserve_utt_vol=True):
    """Performs channel augmentation, based on pre-generated blueprints.

    :param data_path: path to folder with dataset that is to be augmented
    :param musan_path: absolute path to main MUSAN folder
    :param rirs_path: absolute path to `real_rirs_isotropic_noises` folder from RIR dataset
    :param lines_path: relative path to a lines file from this project
    :param clip_signal: whether to keep the length of the utterance the same, defaults to True
    :param amplitude_factor: factor to decrease the signal's amplitude with before convolving
        with RIR, defaults to 0.2
    :param preserve_utt_vol: whether to keep the amplitude of the input utterance the same
        regardless of whether noise is added, defaults to True

    """
    print(f"Augmenting utterances in {lines_path}...")
    with open(os.path.join(data_path, lines_path), 'r') as file:
        lines = file.readlines()
    for line in lines:
        utt_path = line.split()[1]

        orig_utt_path, noise_path, rir_path, processing = utt_path.rsplit(
            '.wav', 1)[0].rsplit('+', maxsplit=3)  # maxsplit to work with SCC utts as well

        # Perform augmentation
        clean_signal = read_wav(os.path.join(data_path, "unaugmented",
                                             orig_utt_path.split("/", maxsplit=1)[1]))
        # Step 1: add noise
        if noise_path == 'quiet':
            signal = deepcopy(clean_signal)
        else:  # Parse name to get right folder and amplification level
            if noise_path.startswith("speech"):
                noise_dir = "speech/us-gov"
                level = 0.5
            elif noise_path.startswith("noise-free"):
                noise_dir = "noise/free-sound"
                level = 0.3
            elif noise_path.startswith("noise-sound"):
                noise_dir = "noise/sound-bible"
                level = 0.3
            elif noise_path.startswith("music-fma-wa"):
                noise_dir = "music/fma-western-art"
                level = 0.5
            elif noise_path.startswith("music-fma"):
                noise_dir = "music/fma"
                level = 0.3
            elif noise_path.startswith("music-jamendo"):
                noise_dir = "music/jamendo"
                level = 0.3
            elif noise_path.startswith("music-rfm"):
                noise_dir = "music/rfm"
                level = 0.3
            elif noise_path.startswith("music-hd"):
                noise_dir = "music/hd-classical"
                level = 0.5
            noise_wav = read_wav(os.path.join(musan_path, noise_dir, noise_path))
            signal = add_noise(clean_signal, noise_wav, noise_level=level,
                               preserve_utt_vol=preserve_utt_vol)

        # Step 2: convolve with room impulse response
        signal = signal * amplitude_factor  # Decrease clipping caused by RIR convolution
        rir_wav = read_wav(os.path.join(rirs_path, rir_path))
        signal = convolve_rir(signal, rir_wav, clip_signal=clip_signal)

        # Step 3: optionally filter signal
        if processing == 'telephone':
            signal = bandpass(signal)
        elif processing == 'highpass':
            signal = highpass(signal)
        elif processing == 'lowpass':
            signal = lowpass(signal)

        file_dir = os.path.join(data_path, utt_path).rsplit('/', 1)[0]
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        soundfile.write(os.path.join(data_path, utt_path), signal, samplerate=16000)

    print("Augmentation complete.")


def lowpass(utt):
    """Applies lowpass filter.

    :param utt: array to be lowpassed
    :return: filtered array
    """
    filter = signal.butter(5, 3000, 'lowpass', output='sos', fs=16000)
    filtered = signal.sosfilt(filter, utt)

    return filtered


def highpass(utt):
    """Applies highpass filter.

    :param utt: array to be highpassed
    :return: filtered array
    """
    filter = signal.butter(5, 392, 'highpass', output='sos', fs=16000)
    filtered = signal.sosfilt(filter, utt)

    return filtered


def bandpass(utt):
    """Applies bandpass filter.

    :param utt: array to be bandpassed
    :return: filtered array
    """
    filter = signal.butter(10, [300, 3400], 'bandpass', output='sos', fs=16000)
    filtered = signal.sosfilt(filter, utt)

    return filtered


def add_noise(utt, noise, noise_level=0.5, preserve_utt_vol=True):
    """Adds noise array to utterance.

    :param utt: array to add noise to
    :param noise: array to add
    :param noise_level: multiplier for noise array, defaults to 0.5
    :param preserve_utt_vol: whether to keep the amplitude of the input utterance the same
        regardless of noise level, defaults to True
    :return: augmented array
    """
    if len(noise) < len(utt):  # Noise should overlap the entire signal
        times = int(np.ceil(len(utt) / len(noise)))
        noise = np.tile(noise, times)
    # Get random part of noise
    if len(noise) == len(utt):  # Catch rare case which would cause error in randint
        start_time = 0
    else:
        start_time = np.random.randint(0, len(noise) - len(utt))
    noise = noise[start_time:start_time + len(utt)]
    noise *= noise_level

    mixed = utt + noise
    if not preserve_utt_vol:
        mixed /= 2

    return mixed


def convolve_rir(utt, rir, clip_signal=True):
    """Convolve audio array with Room Impulse Response.

    :param utt: array to convolve
    :param rir: array to convolve with
    :param clip_signal: whether to keep the length of the utterance the same, defaults to True
    :return: convolved array
    """
    rir = np.mean(rir, axis=1)  # RIRs used are stereo, should be mono

    orig_len = len(utt)

    if orig_len > len(rir):  # To convolve, our arrays need to have the same length
        rir = pad(rir, len(utt))
    else:
        utt = pad(utt, len(rir))

    convolved_signal = signal.convolve(utt, rir)

    if clip_signal:
        return convolved_signal[:orig_len]
    else:
        return convolved_signal


def pad(array, length):
    """Pad array with zeros

    :param array: array to pad
    :param length: length to pad to
    :return: padded array
    """
    new_array = np.zeros(length)
    new_array[:len(array)] = array

    return new_array


def select_utts(data_path, wavs_path, lines_path, min_utt_length=3):
    """Selects utterances from LJ Speech to use in augmented LJ Speech, and writes to
    `selected-sentences.txt'.

    :param data_path: path to main LJ Speech directory
    :param wavs_path: path to LJ Speech ``wavs`` directory, relative to data_path
    :param lines_path: path to LJ Speech' ``metadata.csv``, relative to data_path 
    :param min_utt_length: minimum duration of utterances to be selected, defaults to 3 (same 
        minimum duration as VoxCeleb)
    """
    with open(os.path.join(data_path, lines_path), 'r', encoding='utf-8') as file:
        lines = file.readlines()

    sections = {}
    for line in lines:
        filepath = f'{line.split('|')[0]}.wav'
        section_id = filepath.split('-')[0]
        if section_id not in sections:
            sections[section_id] = []
        sections[section_id].append(filepath)

    sents = []
    for utt_list in sections.values():
        duration = 0
        while duration < min_utt_length:  # We don't want to include very short sentences
            id = np.random.randint(len(utt_list))
            path = utt_list[id]
            duration = librosa.get_duration(path=os.path.join(data_path, wavs_path, path))
        sents.append(path + '\n')  # For writelines below
    with open(os.path.join(data_path, 'selected-sentences.txt'), 'w') as file:
        file.writelines(sents)
