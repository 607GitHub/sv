# Computes spectrogram, Fbank and MFCC with specified parameters

import librosa
import numpy as np


def spectrogram(utt):
    """Calculates spectrogram for 16kHz audio array, with 25ms window and 10ms hop.

    :param utt: audio array
    :return: spectrogram array
    """
    return np.abs(librosa.stft(utt, n_fft=400, hop_length=160))


def fbank(spectrogram, n_mels=80):
    """Calculates Mel filter bank for spectrogram.

    :param spectrogram: spectrogram array
    :param n_mels: number of mels to use, defaults to 80
    :return: Mel filter bank array
    """
    return librosa.feature.melspectrogram(S=spectrogram**2, sr=16000, n_mels=n_mels)


def mfcc(fbank):
    """Calculates first 13 MFCC coefficients for Mel filter bank.
    Includes derivatives and double derivatives.

    :param fbank: Mel filter bank array
    :return: concatenated array of MFCCs, derivatives and double derivatives
    """
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(fbank), n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    deltadelta = librosa.feature.delta(mfcc, order=2)
    return np.concatenate((mfcc, delta, deltadelta))
