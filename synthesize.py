import os

import torch
from TTS.api import TTS
import librosa
import numpy as np
import soundfile

from mydatasets import read_wav


def synthesize_utts(data_path, lines_path):
    """Synthesizes utterances using xTTS-v2, based on pregenerated blueprints.

    :param data_path: path to dataset for which to synthesize utterances
    :param lines_path: relative path to a lines file from this project
    """
    print(f"Synthesizing utterances in {lines_path}...")
    np.random.seed(607)

    with open(os.path.join(data_path, lines_path), 'r') as file:
        lines = file.readlines()
    # Populate dict going from filename to sentence
    with open(os.path.join(data_path, "content-lines.txt"), 'r') as file:
        content_lines = file.readlines()
    content_texts = {}
    for line in content_lines:
        filepath, content = line.split('|')
        filename = filepath.split('/')[-1]
        content_texts[filename] = content

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    for line in lines:
        utt_path = line.split()[1]
        speaker_path, content_path = utt_path.split('+')[:2]
        target_path = os.path.join(data_path, "unaugmented",
                                   '+'.join([speaker_path.split('/', maxsplit=1)[-1], content_path]))
        if os.path.exists(target_path):
            continue  # The utts for most lines will already have been synthesized for another line
        target_dir = target_path.rsplit('/', maxsplit=1)[0]
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        source_speaker_path = os.path.join(data_path, 'speaker-wav',
                                           speaker_path.split('/', maxsplit=1)[-1])
        text = content_texts[content_path.split('/')[-1]].strip()
        utt = synthesize_utt(tts, source_speaker_path, text)
        soundfile.write(target_path, utt, samplerate=16000)

    print("Synthesis complete.")


def synthesize_utt(tts, speaker_path, text):
    """Synthesizes a single utterances.

    :param tts: instantiated text-to-speech model
    :param speaker_path: path to an utterances to clone the speaker of
    :param text: text for synthesized utterances
    :return: generated audio array, in 16 kHz
    """
    array = tts.tts(text=text, speaker_wav=speaker_path, language='en')
    array = librosa.resample(np.array(array), orig_sr=24000, target_sr=16000)

    return array


def select_speaker_utts_2(SCC_path, VOiCES_path):
    """Selects utterances from VOiCES to clone the speakers of for SCC, and writes to
    `speaker-lines.txt`.

    :param SCC_path: path to SCC dataset
    :param VCTK_path: path to VOiCES' `train`, relative to `SCC_path`.
    """
    speakers = []
    dirs = os.listdir(os.path.join(SCC_path, VOiCES_path))
    for speaker in dirs:
        utts = os.listdir(os.path.join(SCC_path, VOiCES_path, speaker))
        utt_name = utts[np.random.randint(len(utts))]
        speakers.append('/'.join(["speaker_wav", speaker, utt_name]) + '\n')
    with open(os.path.join(SCC_path, "speaker-lines.txt"), 'w') as file:
        file.writelines(speakers)


def select_speaker_utts(SCC_path, VCTK_path):
    """Selects utterances from VCTK to clone the speakers of for SCC, and writes to
    `speaker-lines.txt`.

    :param SCC_path: path to SCC dataset
    :param VCTK_path: path to VCTK's extracted `wav48_silence_trimmed`, relative to `SCC_path`.
    """
    speakers = []
    dirs = os.listdir(os.path.join(SCC_path, VCTK_path))
    for speaker in dirs:
        if speaker.endswith('.txt'):
            continue
        utt_path = f"{speaker}/{speaker}_023_mic1.wav"
        if not os.path.exists(os.path.join(SCC_path, VCTK_path,
                                           utt_path.replace(".wav", ".flac"))):
            continue  # In case speaker has not recorded sentence 23
        speakers.append('/'.join(["speaker_wav", utt_path]) + '\n')
        if len(speakers) == 100:
            break
    with open(os.path.join(SCC_path, "speaker-lines.txt"), 'w') as file:
        file.writelines(speakers)


def select_content_utts(SCC_path, LJSpeech_path, min_utt_length=3, n_per_section=4):
    """Selects utterances from LJSpeech to get the text from for SCC synthesis, and writes to
    `content-lines.txt`.

    :param SCC_path: path to SCC dataset
    :param LJSpeech_path: path to LJSpeech's extracted `LJSpeech-1.1`, relative to `SCC_path`. 
    :param min_utt_length: the minimum duration in seconds of the original utterance allowed, 
        defaults to 3, same as used for VoxCeleb
    :param n_per_section: the number of utterances to select for each of the 50 sections included
        in LJ Speech, defaults to 4
    """
    with open(os.path.join(SCC_path, LJSpeech_path, 'metadata.csv'),
              'r', encoding='utf-8') as file:
        lines = file.readlines()

    sections = {}
    for line in lines:
        filepath = f'wavs/{line.split('|')[0]}.wav'
        transcription = line.split('|')[2]
        section_id = filepath.split('-')[0]
        if section_id not in sections:
            sections[section_id] = []
        sections[section_id].append((filepath, transcription))

    sents = []
    for utt_list in sections.values():
        for i in range(n_per_section):
            duration = 0
            while duration < min_utt_length:  # We don't want to include very short sentences
                id = np.random.randint(len(utt_list))
                path = utt_list[id][0]
                duration = librosa.get_duration(
                    path=os.path.join(SCC_path, LJSpeech_path, path))
            sents.append('/'.join([LJSpeech_path.replace(os.sep, '/'), path]) +
                         '|' + utt_list[id][1])  # Write path as well as transcription
            utt_list.pop(id)
    with open(os.path.join(SCC_path, "content-lines.txt"), 'w') as file:
        file.writelines(sents)
