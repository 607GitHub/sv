import os
import argparse
import sys
import shutil

from mydatasets import BritishIsles, VOiCES, LJSpeech, VCTK, SCC
import resample
import augment
# import synthesize Done below because it is slow and only needed for SCC


def setup_british_isles(data_path, generate_splits):
    """Sets up British Isles dataset, extracting archives and resampling required utterances.
    Removes extracted archives when finished. 

    :param data_path: path to `british-isles` data folder, including the zipped archives and 
    `line_index_all.csv`
    :param generate_splits: whether to generate new splits or not (bool)
    """
    print("Setting up British Isles...")
    if generate_splits:
        print("Generating new splits.")
        BritishIsles(data_path, "line_index_all.csv", mode="generate")
    else:
        print("Using existing splits.")

    unzip_archives(data_path, "temp")

    wav_paths = extract_wav_paths(data_path, ["speaker-train-lines.txt",
                                              "speaker-dev-lines.txt",
                                              "speaker-test-lines.txt"])

    resample.resample_data(os.path.join(data_path, "temp"), os.path.join(data_path, "wav"),
                           wav_paths, 48000)

    shutil.rmtree(os.path.join(data_path, "temp"))

    print("British Isles set up.")


def setup_VOiCES(data_path, generate_splits):
    """Sets up VOiCES dataset, extracting archive.

    :param data_path: path to `VOiCES` data folder with the archived dataset
    :param generate_splits: whether to generate new splits or not (bool)
    """
    print("Setting up VOiCES...")
    unzip_archives(data_path)

    if generate_splits:
        print("Generating new splits.")
        VOiCES(data_path, "references/train_index.csv",
               "references/test_index.csv", mode="generate")
    else:
        print("Using existing splits.")

    print("VOiCES set up.")


def setup_LJSpeech(data_path, generate_splits, clip_signal, amplitude_factor):
    """Sets up augmented LJ Speech, extracting archives, resampling required utterances and
    augmenting with channel settings. Removes extracted LJ Speech archive when finished,
    but keeps unaugmented resampled utterances.

    :param data_path: path to folder with LJ Speech archive; MUSAN and RIR, either archived
    or unzipped, should be one directory above
    :param generate_splits: whether to generate new splits or not (bool)
    :param clip_signal: whether to keep the length of the utterance the same
    :param amplitude_factor: factor to decrease the signal's amplitude with before convolving with
        RIR
    """
    print("Setting up LJ Speech...")
    # Check if datasets for augmentation (MUSAN and RIR) have been extracted yet
    if not os.path.exists(os.path.join(data_path, "..", "RIRS_NOISES")):
        unzip_archives(os.path.join(data_path, ".."))
    unzip_archives(data_path, "temp")

    musan_path = os.path.join(data_path, "..", "musan")
    rirs_path = os.path.join(data_path, "..", "RIRS_NOISES", "real_rirs_isotropic_noises")
    if generate_splits:
        print("Generating new splits.")
        augment.select_utts(data_path,
                            os.path.join("temp", "LJSpeech-1.1", "wavs"),
                            os.path.join("temp", "LJSpeech-1.1", "metadata.csv"))
        LJSpeech(data_path, "selected-sentences.txt", musan_path, rirs_path, mode="generate")
    else:
        print("Using existing splits.")

    wav_paths = extract_wav_paths(data_path, ["channel-train-lines.txt",
                                              "channel-dev-lines.txt",
                                              "channel-test-lines.txt"],
                                  remove_subdirs=2)  # No subdirs for different utts yet
    resample.resample_data(os.path.join(data_path, "temp", "LJSpeech-1.1", "wavs"),
                           os.path.join(data_path, "unaugmented"), wav_paths, 22050)

    os.mkdir(os.path.join(data_path, "wav"))
    for lines_path in ["channel-train-lines.txt", "channel-dev-lines.txt",
                       "channel-test-lines.txt"]:
        augment.augment_with_channel(data_path, musan_path, rirs_path, lines_path,
                                     clip_signal=clip_signal, amplitude_factor=amplitude_factor)

    shutil.rmtree(os.path.join(data_path, "temp"))
    print("Augmented LJ Speech set up.")


def setup_VCTK(data_path, generate_splits):
    """Sets up VCTK dataset, extracting archive and resampling required utterances.
    Removes extracted archive when finished. 

    :param data_path: path to `VCTK` data folder with the zipped archive
    :param generate_splits: whether to generate new splits or not (bool)
    """
    print("Setting up VCTK...")
    unzip_archives(data_path, "temp")
    unzip_archives(os.path.join(data_path, "temp"))  # VCTK has a zip in a zip
    if generate_splits:
        print("Generating new splits.")
        VCTK(data_path, os.path.join("temp", "wav48_silence_trimmed"), mode="generate")
    else:
        print("Using existing splits.")

    wav_paths = extract_wav_paths(data_path, ["train-lines.txt", "dev-lines.txt",
                                              "test-lines.txt"])
    resample.resample_data(os.path.join(data_path, "temp", "wav48_silence_trimmed"),
                           os.path.join(data_path, "wav"), wav_paths, 48000, flac=True)

    shutil.rmtree(os.path.join(data_path, "temp"))
    print("VCTK set up.")


def setup_SCC(data_path, generate_splits, version, clip_signal, amplitude_factor,
              preserve_utt_vol):
    """Sets up SCC, extracting archives, resampling required utterances, synthesizing utterances
    and augmenting with channel settings. Removes extracted archives when finished,
    but keeps resampled utterances and unaugmented synthesized utterances.

    :param data_path: path to empty directory; MUSAN and RIR, either archived
    or unzipped, should be one directory above
    :param generate_splits: whether to generate new splits or not (bool)
    :param version: whether to generate version 1 or 2 of SCC (both will be placed in the same directory)
    :param clip_signal: whether to keep the length of the utterance the same
    :param amplitude_factor: factor to decrease the signal's amplitude with before convolving with
        RIR
    :param preserve_utt_vol: whether to keep the amplitude of the input utterance the same
        regardless of whether noise is added
    """
    import synthesize
    print("Setting up SCC, version", version)

    # Check if datasets for augmentation (MUSAN and RIR) have been extracted yet
    if not os.path.exists(os.path.join(data_path, "..", "RIRS_NOISES")):
        unzip_archives(os.path.join(data_path, ".."))
    musan_path = os.path.join(data_path, "..", "musan")
    rirs_path = os.path.join(data_path, "..", "RIRS_NOISES", "real_rirs_isotropic_noises")
    # We need to unzip the corpora that are used to base synthesis on
    unzip_archives(os.path.join(data_path, "..", "LJSpeech"), "../SCC/temp-LJSpeech")
    if version == 1:
        unzip_archives(os.path.join(data_path, "..", "VCTK"), "../SCC/temp-VCTK")
        unzip_archives(os.path.join(data_path, "temp-VCTK"))  # VCTK has a zip in a zip
    elif version == 2:
        unzip_archives(os.path.join(data_path, "..", "VOiCES"), "../SCC/temp-VOiCES")

    if generate_splits:
        print("Generating new splits.")
        # Select speaker and content utts
        if version == 1:
            synthesize.select_speaker_utts(data_path,
                                           os.path.join("temp-VCTK", "wav48_silence_trimmed"))
            synthesize.select_content_utts(data_path,
                                           os.path.join("temp-LJSpeech", "LJSpeech-1.1"),
                                           n_per_section=2)
        elif version == 2:
            synthesize.select_speaker_utts_2(data_path,
                                             os.path.join("temp-VOiCES", "VOiCES_devkit", "source-16k", "train"))
            synthesize.select_content_utts(data_path,
                                           os.path.join("temp-LJSpeech", "LJSpeech-1.1"),
                                           n_per_section=4)

        # Generate blueprints for utts to be synthesized and augmented
        SCC(data_path, "content-lines.txt", version, "speaker-lines.txt",
            musan_path, rirs_path, mode='generate')
    else:
        print("Using existing splits.")

    if version == 1:
        wav_paths = extract_wav_paths(data_path, ["speaker-lines.txt"], no_labels=True)
        resample.resample_data(os.path.join(data_path, "temp-VCTK", "wav48_silence_trimmed"),
                               os.path.join(data_path, "speaker-wav"), wav_paths,
                               48000, flac=True)
    elif version == 2:  # No resampling required for VOiCES, but data should be in right dir
        shutil.move(os.path.join(data_path, "temp-VOiCES",
                    "VOiCES_devkit", "source-16k", "train"), data_path)
        os.rename(os.path.join(data_path, "train"), os.path.join(data_path, "speaker-wav"))

    os.mkdir(os.path.join(data_path, "unaugmented"))
    os.mkdir(os.path.join(data_path, "wav"))
    for use in ['probing', 'disentanglement']:
        for task in ['speaker', 'content', 'channel']:
            for split in ['train', 'dev', 'test']:
                lines_path = "-".join([use, task, split, "lines"]) + ".txt"
                synthesize.synthesize_utts(data_path, lines_path)
                augment.augment_with_channel(data_path, musan_path, rirs_path, lines_path,
                                             clip_signal, amplitude_factor, preserve_utt_vol)

    shutil.rmtree(os.path.join(data_path, "temp-LJSpeech"))
    if version == 1:
        shutil.rmtree(os.path.join(data_path, "temp-VCTK"))
    elif version == 2:
        shutil.rmtree(os.path.join(data_path, "temp-VOiCES"))
    print("SCC set up.")


def unzip_archives(data_path, target_dir="."):
    """Extracts all archives in directory with .zip, .tar.gz or .tar.bz2 extension.

    :param data_path: path to directory with archives to be extracted
    :param target_dir: path where files should be extracted to, defaults to "."
    """
    print(f"Unzipping archive(s) in {data_path}...")
    errors = False
    target_path = os.path.join(data_path, target_dir)
    if not os.path.exists(target_path):
        os.mkdir(os.path.join(target_path))
    contents = os.listdir(data_path)
    for item in contents:
        if item.endswith(".zip") or item.endswith(".tar.gz") or item.endswith(".tar.bz2"):
            shutil.unpack_archive(os.path.join(data_path, item), target_path)
    if not errors:
        print("Archive(s) unzipped.")


def extract_wav_paths(data_path, splits_paths, remove_subdirs=1, no_labels=False):
    """Reads in a list of lines files, and returns a set of all the paths referred to in the files.
    For augmented datasets, returns only the paths to the base utterances.

    :param data_path: path to dataset under consideration
    :param splits_paths: list of relative paths to splits files from which to extract wav paths 
    :param remove_subdirs: how many directories to remove from the path, defaults to 1;
        used to not return unwanted (sub)-directories
    :param no_labels: set to True for files with only paths and no labels, defaults to False
    :return: set of unique wav paths referred to in all the lines from `splits_paths`
    """
    wav_paths = set()  # The same file might be referred to multiple times
    for splits_path in splits_paths:
        with open(os.path.join(data_path, splits_path), "r") as file:
            lines = file.readlines()
        for line in lines:
            if no_labels:
                wav_path = line
            else:
                wav_path = line.split()[1]
            wav_path = wav_path.split("+")[0]  # In case of augmented datasets
            # Remove subdirectories that are not in original dataset
            wav_path = wav_path.split("/", maxsplit=remove_subdirs)[-1]
            wav_paths.add(wav_path.strip())

    return wav_paths


def main(args):
    """Calls setup method for each dataset included in `args.datasets`.

    :param args: commandline arguments
    """
    for dataset in args.datasets:
        match dataset:
            case 'british-isles':
                setup_british_isles(os.path.join(
                    args.data_path, 'british-isles'), args.generate_splits)
            case 'VOiCES':
                setup_VOiCES(os.path.join(args.data_path, 'VOiCES'), args.generate_splits)
            case 'LJSpeech':
                setup_LJSpeech(os.path.join(args.data_path, 'LJSpeech'), args.generate_splits,
                               clip_signal=not args.no_signal_clip,
                               amplitude_factor=args.amplitude_factor,
                               preserve_utt_vol=not args.noise_decreases_utt_vol)
            case 'VCTK':
                setup_VCTK(os.path.join(args.data_path, 'VCTK'), args.generate_splits)
            case 'SCC':
                setup_SCC(os.path.join(args.data_path, 'SCC'), args.generate_splits,
                          args.SCC_version, clip_signal=not args.no_signal_clip,
                          amplitude_factor=args.amplitude_factor,
                          preserve_utt_vol=not args.noise_decreases_utt_vol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',
                        help="Path to main data directory",
                        type=str,
                        default='data')
    parser.add_argument('--datasets',
                        help="Which datasets to set-up",
                        type=str,
                        nargs='+',
                        choices=['british-isles', 'VOiCES', 'LJSpeech', 'VCTK', 'SCC'],
                        default=['british-isles, VOiCES'])
    parser.add_argument('--generate_splits',
                        help="Whether to generate new train/dev/test splits",
                        action="store_true")
    parser.add_argument('--no_signal_clip',
                        help="Include to allow RIRs to increase utterance length (replicating original dataset)",
                        action='store_true')
    parser.add_argument('--amplitude_factor',
                        help="To reduce clipping before convolving with RIR",
                        type=float,
                        default=0.2)
    parser.add_argument('--noise_decreases_utt_vol',
                        help="Include to let noise augmentation decrease volume of the original utterance (replicating original dataset)",
                        action='store_true')
    parser.add_argument('--SCC_version',
                        help="Which version of SCC to generate",
                        type=int,
                        choices=[1, 2],
                        default=2)
    args = parser.parse_args()

    sys.exit(main(args))
