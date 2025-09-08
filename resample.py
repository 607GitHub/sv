import os

import librosa
import soundfile


def resample_data(source_dir, target_dir, file_paths, source_rate, target_rate=16000, flac=False):
    """Resamples a set of audio files. The old files are not deleted.

    :param source_dir: path to the directory from where to read the relative file paths
    :param target_dir: path to the directory in which to put the relative new file paths
    :param file_paths: list or set of file paths of files to resample, relative to
        `source_dir`. The same paths will be created in `target_dir`.
    :param source_rate: source rate of to be resampled data
    :param target_rate: rate to resample to, defaults to 16000
    :param flac: 'hack' to work with VCTK, which originally has FLACs, defaults to False
    """
    print("Resampling data...")
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    for file_path in file_paths:
        if flac:
            real_file_path = file_path[:-4] + '.flac'
        else:
            real_file_path = file_path
        old = librosa.load(os.path.join(source_dir, real_file_path), sr=source_rate)[0]
        new = librosa.resample(old, orig_sr=source_rate, target_sr=target_rate)
        file_dir = os.path.join(target_dir, real_file_path).rsplit('/', 1)[0]
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        soundfile.write(os.path.join(target_dir, file_path), new, target_rate)

    print("Data resampled.")
