# Contains datasets for classification (probing + disentanglement) and speaker verification
import os
import random
from copy import deepcopy

from torch.utils.data import Dataset
import numpy as np
from scipy.io import wavfile


class ClassificationDataset(Dataset):
    """Implements the general form of dataset used for classification tasks,
    mainly taking care of data loading.

    :param data_path: path to root of dataset
    :param return_type: what type of data to return when indexed. Can be:
        - tuple of label and audio array (`array`)
        - tuple of label and filepath (`path`)
        - tuple of label, audio array and filepath (`both`)
    """

    def __init__(self, data_path, return_type='array'):
        self.wav_paths = []
        self.labels = []
        self.data_path = os.path.normpath(data_path)
        self.return_type = return_type

    def __getitem__(self, index):
        """Fetches item from dataset.

        :param index: index of item to fetch (int)
        :return: depends on `self.return_type`
        """
        label = self.labels[index]
        if (self.return_type == 'array'):
            utt = read_wav(os.path.join(self.data_path, self.wav_paths[index]))
        if (self.return_type == 'path'):
            utt = os.path.join(self.data_path, self.wav_paths[index])
        if (self.return_type == 'both'):
            audio_array = read_wav(os.path.join(self.data_path, self.wav_paths[index]))
            filepath = os.path.join(self.data_path, self.wav_paths[index])
            return label, audio_array, filepath

        return label, utt

    def __len__(self):
        """Returns the number of examples in the dataset.

        :return: number of examples in the dataset (int)
        """
        return len(self.labels)

    def n_classes(self):
        """Returns the number of classes in the dataset.

        :return: number of classes in the dataset (int)
        """
        return len(set(self.labels))

    def load_data(self, lines_path):
        """Reads in text file with lines consisting of a label and a file path, seperated by a
        space, and stores both labels and file paths.

        :param lines_path: path to lines file, relative to `self.data_path`
        """
        with open(os.path.join(self.data_path, lines_path), 'r') as file:
            examples = file.readlines()
        for line in examples:
            example = line.split()
            self.labels.append(int(example[0]))
            path = example[1]
            self.wav_paths.append(os.path.normpath(path))


class BritishIsles(ClassificationDataset):
    """Implements British Isles dataset. Can be used to generate splits or to load existing splits.
    `setup_datasets.py` is used to generate the dataset.

    :param data_path: path to `british-isles` data folder
    :param lines_path: 
        - if mode='generate': relative path to `line_index_all.csv`
        - if mode='read': relative path to a lines file from this project
    :param return_type: see parent class, defaults to 'array'
    :param mode: whether to generate splits ('generate') or load dataset ('read'), defaults to 
        'read'
    """

    def __init__(self, data_path, lines_path, return_type='array', mode='read'):
        super().__init__(data_path, return_type)

        if (mode == 'generate'):
            self.generate_splits(lines_path)
        elif (mode == 'read'):
            self.load_data(lines_path)

    def generate_splits(self, all_lines_path):
        """Generates train, dev and test splits for speaker and content classification tasks.

        :param all_lines_path: path to `line_index_all.csv`, relative to `self.data_path`
        """
        random.seed(607)

        with open(os.path.join(self.data_path, all_lines_path), 'r') as file:
            lines = file.readlines()

        sentences = {}
        speakers = {}

        for line in lines:
            example = line.split(', ')
            sentid = example[0]
            file_id = example[1]
            speaker = file_id[:9]
            if (sentid[:2] != 'EN'):
                continue
            sentid = int(sentid[2:])
            if (sentid > 50):
                continue
            path = f'wav/{file_id}.wav'
            if sentid not in sentences:
                sentences[sentid] = []
            sentences[sentid].append(path)
            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append(path)

        train_split = .8
        dev_split = .1

        for data_dict, task in [(sentences, 'content'), (speakers, 'speaker')]:
            train_ex = []
            dev_ex = []
            test_ex = []

            for class_idx, (class_id, utt_list) in enumerate(data_dict.items()):
                ids = [i for i in range(len(utt_list))]
                train_n = int(len(ids) * train_split)
                dev_n = int(len(ids) * dev_split)

                train_ids = random.sample(ids, train_n)
                train_ex.extend([(class_idx, utt_list[train_id]) for train_id in train_ids])

                ids = list(set(ids) - set(train_ids))
                dev_ids = random.sample(ids, dev_n)
                dev_ex.extend([(class_idx, utt_list[dev_id]) for dev_id in dev_ids])

                test_ids = list(set(ids) - set(dev_ids))
                test_ex.extend([(class_idx, utt_list[test_id]) for test_id in test_ids])

            for split in [('train', train_ex), ('dev', dev_ex), ('test', test_ex)]:
                lines = []
                for ex in split[1]:
                    lines.append(str(ex[0]) + ' ' + ex[1] + '\n')
                with open(os.path.join(self.data_path, "-".join(
                        [task, split[0], "lines"]) + ".txt"), 'w') as file:
                    file.writelines(lines)


class VOiCES(ClassificationDataset):
    """Implements VOiCES dataset. Can be used to generate splits or to load existing splits.

    :param data_path: path to `VOiCES` data folder
    :param lines_path: 
        - if mode='generate': relative path to `train_index.csv`
        - if mode='read': relative path to a lines file from this project
    :param test_lines_path: relative path to `test_index.csv`, only to be supplied when
        mode='generate'
    :param return_type: see parent class, defaults to 'array'
    :param mode: whether to generate splits ('generate') or load dataset ('read'), defaults to 
        'read'
    """

    def __init__(self, data_path, lines_path, test_lines_path=None, return_type='array',
                 mode='read'):
        super().__init__(data_path, return_type)

        if (mode == 'generate'):
            self.generate_splits(lines_path, test_lines_path)
        elif (mode == 'read'):
            self.load_data(lines_path)

    def generate_splits(self, train_dev_lines_path, test_lines_path):
        """Generates train, dev and test splits for channel classification task.

        :param train_dev_lines_path: path to `train_index.csv`, relative to `self.data_path`
        :param test_lines_path: path to `test_index.csv`, relative to `self.data_path`
        """
        random.seed(607)

        channels = {}
        # Both splits have the channel settings in different orders,
        # we need to be able to map to the same labels
        channel_id_to_idx = {}

        for split, lines_path in [('train_dev', train_dev_lines_path), ('test', test_lines_path)]:
            with open(os.path.join(self.data_path, lines_path), 'r') as file:
                lines = file.readlines()

            channels[split] = {}

            for line in lines[1:]:
                example = line.split(',')
                distractor = example[3]
                path = example[4]
                mic = example[6]
                room = example[8]
                if (path[:4] != 'dist'):  # Header or empty line
                    continue
                channel_id = distractor + mic + room
                if channel_id not in channels[split]:
                    channels[split][channel_id] = []
                if channel_id not in channel_id_to_idx:
                    channel_id_to_idx[channel_id] = len(channel_id_to_idx.keys())
                channels[split][channel_id].append(path)

        train_split = .9

        train_ex = []
        dev_ex = []

        for chann_id, chann_list in channels['train_dev'].items():
            chann_idx = channel_id_to_idx[chann_id]
            ids = [i for i in range(len(chann_list))]
            train_n = int(len(ids) * train_split)

            train_ids = random.sample(ids, train_n)
            train_ex.extend([(chann_idx, chann_list[train_id]) for train_id in train_ids])

            dev_ids = list(set(ids) - set(train_ids))
            dev_ex.extend([(chann_idx, chann_list[dev_id]) for dev_id in dev_ids])

        test_ex = []

        for chann_id, chann_list in channels['test'].items():
            chann_idx = channel_id_to_idx[chann_id]
            test_ex.extend([(chann_idx, path) for path in chann_list])

        for split in [('train', train_ex), ('dev', dev_ex), ('test', test_ex)]:
            lines = []
            for ex in split[1]:
                lines.append(str(ex[0]) + ' ' + ex[1] + '\n')
            with open(os.path.join(self.data_path, "-".join(
                    [split[0], "lines"]) + ".txt"), 'w') as file:
                file.writelines(lines)


class LJSpeech(ClassificationDataset):
    """Implements augmented LJSpeech dataset. Can be used to generate splits or to load existing
    splits. Augmented LJSpeech is generated using `setup_datasets.py`, which also uses 
    `augment.py`.

    :param data_path: path to `LJSpeech` data folder
    :param lines_path: 
        - if mode='generate': relative path to a file containing paths of utterances selected
            for augmentation
        - if mode='read': relative path to a lines file from this project
    :param musan_path: absolute path to main MUSAN folder
    :param rirs_path: absolute path to `real_rirs_isotropic_noises` folder from RIR dataset
    :param return_type: see parent class, defaults to 'array'
    :param mode: whether to generate splits ('generate') or load dataset ('read'), defaults to 
        'read'
    """

    def __init__(self, ljspeech_data_path, lines_path, musan_path=None, rirs_path=None, return_type='array', mode='read'):
        super().__init__(ljspeech_data_path, return_type)

        if (mode == 'generate'):
            self.generate_splits(lines_path, musan_path, rirs_path)
        elif (mode == 'read'):
            self.load_data(lines_path)

    def generate_splits(self, selected_utts_path, musan_path, rirs_path):
        """Generates train, dev and test splits for content and channel classification tasks.
        A blueprint is created for augmentation, but actual augmentation is done in augment.py.

        :param selected_utts_path: path to a file containing paths of utterances selected for
            augmentation, relative to `self.data_path`
        :param musan_path: absolute path to main MUSAN folder
        :param rirs_path: absolute path to `real_rirs_isotropic_noises` folder from RIR dataset
        """
        random.seed(607)

        with open(os.path.join(self.data_path, selected_utts_path), 'r') as file:
            utts = file.readlines()  # The utterances to be augmented
        utts = [os.path.normpath(utt.rstrip()) for utt in utts]

        sents = {}  # Sentence classes for content classification
        channels = {}  # Channel classes for channel classification

        # Get lists of paths for different augmentation settings
        wav_paths = {}
        for noise, dirs in [('pop', [os.path.join('music', 'fma'), os.path.join('music', 'rfm'),
                                     os.path.join('music', 'jamendo')]),
                            ('classical', [os.path.join('music', 'fma-western-art'),
                                           os.path.join('music', 'hd-classical')]),
                            ('babble', [os.path.join('speech', 'us-gov')]),
                            ('misc', [os.path.join('noise', 'free-sound'),
                                      os.path.join('noise', 'sound-bible')])]:

            wav_paths[noise] = []
            for dir in dirs:
                wav_paths[noise].extend([os.path.join(dir, filename)
                                         for filename in os.listdir(os.path.join(
                                             musan_path, dir)) if filename[-4:] == '.wav'])

        rir_files = os.listdir(rirs_path)
        for room in ['stairway', 'office', 'meeting', 'booth', 'aula']:
            wav_paths[room] = [filename for filename in rir_files if filename.startswith(
                room, 23)]

        for utt in utts:
            sents[utt] = []

            for noise in ['quiet', 'pop', 'classical', 'babble', 'misc']:
                for room in ['stairway', 'office', 'meeting', 'booth', 'aula']:
                    for bandpass in ['normal', 'telephone']:
                        # Step 1: add noise
                        if noise == 'quiet':
                            noise_path = 'quiet'
                        else:
                            noise_path = wav_paths[noise][np.random.randint(
                                len(wav_paths[noise]))]  # Random noise file

                        # Step 2: convolve with room impulse response
                        rir_path = wav_paths[room][np.random.randint(
                            len(wav_paths[room]))]  # Random RIR

                        # We only include the filename in the noise_path, to not have / in
                        # filename. augment.augment_with_channel will infer full path.
                        filename = "+".join([utt, noise_path.split(os.sep)[-1],
                                             rir_path, bandpass]) + ".wav"

                        chann_id = noise + room + bandpass
                        if chann_id not in channels:
                            channels[chann_id] = []
                        sents[utt].append('/'.join(['wav', utt, filename]))
                        channels[chann_id].append('/'.join(['wav', utt, filename]))

        # Generate splits (copied from BritishIsles, could possibly be generalised)
        train_split = .8
        dev_split = .1

        for data_dict, task in [(sents, 'content'), (channels, 'channel')]:
            train_ex = []
            dev_ex = []
            test_ex = []

            for class_idx, (class_id, utt_list) in enumerate(data_dict.items()):
                ids = [i for i in range(len(utt_list))]
                train_n = int(len(ids) * train_split)
                dev_n = int(len(ids) * dev_split)

                train_ids = random.sample(ids, train_n)
                train_ex.extend([(class_idx, utt_list[train_id]) for train_id in train_ids])

                ids = list(set(ids) - set(train_ids))
                dev_ids = random.sample(ids, dev_n)
                dev_ex.extend([(class_idx, utt_list[dev_id]) for dev_id in dev_ids])

                test_ids = list(set(ids) - set(dev_ids))
                test_ex.extend([(class_idx, utt_list[test_id]) for test_id in test_ids])

            for split in [('train', train_ex), ('dev', dev_ex), ('test', test_ex)]:
                lines = []
                for ex in split[1]:
                    lines.append(str(ex[0]) + ' ' + ex[1] + '\n')
                with open(os.path.join(self.data_path, "-".join(
                        [task, split[0], "lines"]) + ".txt"), 'w') as file:
                    file.writelines(lines)


class VCTK(ClassificationDataset):
    """Implements VCTK dataset. Can be used to generate splits or to load existing splits.
    `setup_datasets.py` is used to generate the dataset.

    :param data_path: path to `VCTK` data folder
    :param lines_path: 
        - if mode='generate': relative path to `wav48_silence_trimmed` folder
        - if mode='read': relative path to a lines file from this project
    :param return_type: see parent class, defaults to 'array'
    :param mode: whether to generate splits ('generate') or load dataset ('read'), defaults to 
        'read'
    """

    def __init__(self, data_path, lines_path, return_type='array', mode='read'):
        super().__init__(data_path, return_type)

        if (mode == 'generate'):
            self.generate_splits(lines_path)
        elif (mode == 'read'):
            self.load_data(lines_path)

    def generate_splits(self, wav_dir):
        """Generates train, dev and test splits for speaker classification task.

        :param wav_dir: path to `wav48_silence_trimmed` folder, relative to `self.data_path`
        """
        random.seed(607)

        speakers = {}

        dirs = os.listdir(os.path.join(self.data_path, wav_dir))
        for speaker in dirs:
            if speaker.endswith('.txt'):
                continue
            speakers[speaker] = []
            utts = os.listdir(os.path.join(self.data_path, wav_dir, speaker))
            for utt in utts:
                _, sentence, micwav = utt.split("_")
                if not micwav.startswith("mic1"):
                    continue
                if int(sentence) < 26:  # We don't use the sentences that are also in British Isles
                    continue
                speakers[speaker].append(f"wav/{speaker}/{utt.replace(".flac", ".wav")}")
                if len(speakers[speaker]) == 50:  # We select 50 sentences per speaker
                    break

        train_split = .8
        dev_split = .1

        train_ex = []
        dev_ex = []
        test_ex = []

        for speaker_idx, (speaker_id, speaker_list) in enumerate(speakers.items()):
            ids = [i for i in range(len(speaker_list))]
            train_n = int(len(ids) * train_split)
            dev_n = int(len(ids) * dev_split)

            train_ids = random.sample(ids, train_n)
            train_ex.extend([(speaker_idx, speaker_list[train_id]) for train_id in train_ids])

            ids = list(set(ids) - set(train_ids))
            dev_ids = random.sample(ids, dev_n)
            dev_ex.extend([(speaker_idx, speaker_list[dev_id]) for dev_id in dev_ids])

            test_ids = list(set(ids) - set(dev_ids))
            test_ex.extend([(speaker_idx, speaker_list[test_id]) for test_id in test_ids])

        for split in [('train', train_ex), ('dev', dev_ex), ('test', test_ex)]:
            lines = []
            for ex in split[1]:
                lines.append(str(ex[0]) + ' ' + ex[1] + '\n')
            with open(os.path.join(self.data_path, "-".join(
                    [split[0], "lines"]) + ".txt"), 'w') as file:
                file.writelines(lines)


class SCC(ClassificationDataset):
    """Implements SCC dataset. Can be used to generate splits or to load existing splits.
    SCC is generated using `setup_datasets.py`, which also uses `synthesize.py` and `augment.py`.

    :param data_path: path to `SCC` data folder
    :param lines_path: 
        - if mode='generate': relative path to a file containing paths and scripts of utterances
            selected as content for synthesis
        - if mode='read': relative path to a lines file from this project
    :param version: version of SCC to generate, 1 or 2;
        1 requires 100 content lines and speaker lines, 2 requires 200 content lines and
        speaker lines
    :param lines_path_2: relative path to a file containing paths of utterances of which to clone
        the speaker in synthesis, only supplied when mode='generate', defaults to None
    :param musan_path: absolute path to main MUSAN folder, defaults to None
    :param rirs_path: absolute path to `real_rirs_isotropic_noises` folder from RIR dataset, defaults to None
    :param return_type: see parent class, defaults to 'array'
    :param mode: whether to generate splits ('generate') or load dataset ('read'), defaults to 
        'read'
    """

    def __init__(self, data_path, lines_path, version=2, lines_path_2=None,  musan_path=None,
                 rirs_path=None, return_type='array', mode='read'):
        super().__init__(data_path, return_type)

        if (mode == 'generate'):
            self.generate_splits(version, lines_path, lines_path_2, musan_path, rirs_path)
        elif (mode == 'read'):
            self.load_data(lines_path)

    def generate_splits(self, version, content_lines_path, speaker_lines_path, musan_path, rirs_path):
        """Generates train, dev and test splits for speaker, content and channel classification
        tasks. A blueprint is created for synthesis and augmentation, but actual synthesis and 
        augmentation are done in synthesis.py and augment.py.

        :param version: version of SCC to generate, 1 or 2;
            1 requires 100 content lines and speaker lines, 2 requires 200 content lines and
            speaker lines
        :param content_lines_path: path to a file containing paths and scripts of utterances
            selected as content for synthesis, relative to `self.data_path`
        :param speaker_lines_path: path to a file containing paths of utterances of which to clone
        the speaker in synthesis, relative to `self.data_path`
        :param musan_path: absolute path to main MUSAN folder
        :param rirs_path: absolute path to `real_rirs_isotropic_noises` folder from RIR dataset
        """
        random.seed(607)

        # The lines to be synthesized (paths and transcription)
        with open(os.path.join(self.data_path, content_lines_path), 'r') as file:
            content_lines = file.readlines()
        content_lines = [path.rstrip() for path in content_lines]
        # The voices to clone (paths only)
        with open(os.path.join(self.data_path, speaker_lines_path), 'r') as file:
            speaker_lines = file.readlines()
        speaker_lines = [path.rstrip() for path in speaker_lines]

       # Get lists of paths for different augmentation settings
        wav_paths = {}
        for noise, dirs in [('pop', [os.path.join('music', 'fma'), os.path.join('music', 'rfm'),
                                     os.path.join('music', 'jamendo')]),
                            ('classical', [os.path.join('music', 'fma-western-art'),
                             os.path.join('music', 'hd-classical')]),
                            ('babble', [os.path.join('speech', 'us-gov')]),
                            ('misc', [os.path.join('noise', 'free-sound'),
                                      os.path.join('noise', 'sound-bible')])]:

            wav_paths[noise] = []
            for dir in dirs:
                wav_paths[noise].extend([os.path.join(dir, filename)
                                         for filename in os.listdir(os.path.join(
                                             musan_path, dir)) if filename[-4:] == '.wav'])

        rir_files = os.listdir(rirs_path)
        for room in ['stairway', 'office', 'meeting', 'booth', 'aula', 'lecture']:
            wav_paths[room] = [filename for filename in rir_files if filename.startswith(
                f'binaural_{room}', 14)]
        for room in ['smallroom1', 'mediumroom1', 'largeroom1', 'largeroom2']:
            wav_paths[room] = [filename for filename in rir_files if filename.startswith(
                room, 18)]

        all_noises = ['quiet', 'pop', 'classical', 'babble', 'misc']
        if version == 1:
            all_rooms = ['stairway', 'office', 'meeting', 'booth', 'aula',
                         'lecture', 'smallroom1', 'mediumroom1', 'largeroom1', 'largeroom2']
        elif version == 2:  # To get same domain for both probing and disentanglement split
            all_rooms = ['stairway', 'smallroom1', 'office', 'mediumroom1', 'meeting',
                         'largeroom1', 'booth', 'largeroom2', 'aula', 'lecture']

        # Select indices of settings to use for different properties in different datasets
        speaker_ids, content_ids, channel_ids = {}, {}, {}
        remaining_ids = {}
        for property in ['speaker', 'content', 'channel']:
            # Remaining ids for the other two properties, to not reuse exact examples between tasks
            remaining_ids[property] = [i for i in range(100)]
        for dictionary, var_property in [(speaker_ids, 'speaker'), (content_ids, 'content'),
                                         (channel_ids, 'channel')]:
            for target_property in ['speaker', 'content', 'channel']:
                dictionary[target_property] = {}
                if var_property == target_property:  # We use all available settings for the
                    # attribute trained on
                    dictionary[target_property]['disentanglement'] = [i for i in range(100)]
                    dictionary[target_property]['probing'] = [i for i in range(100)]
                else:
                    # Select ids for disentanglement and probing for this varied property for
                    # target task
                    selected_ids = random.sample(remaining_ids[var_property], 30)
                    dictionary[target_property]['disentanglement'] = selected_ids[:15]
                    dictionary[target_property]['probing'] = selected_ids[15:]
                    remaining_ids[var_property] = [
                        id for id in remaining_ids[var_property] if id not in selected_ids]
                if version == 2:
                    dictionary[target_property]['probing'] = [idx + 100 for idx in
                                                              dictionary[target_property]['probing']]
                    # For SCC 1, there are 100 classes for each attribute, which are used in both
                    # probing and disentanglement splits. For SCC 2, there are 200 classes for
                    # each attribute, of which # the first 100 get used in probing, and the latter
                    # 100 get used in disentanglement.

        speakers = {'disentanglement': {}, 'probing': {}}
        sents = {'disentanglement': {}, 'probing': {}}
        channels = {'disentanglement': {}, 'probing': {}}
        for data_dictionary, target_property in [(sents, 'content'), (channels, 'channel'),
                                                 (speakers, 'speaker')]:
            for dataset in ['probing', 'disentanglement']:
                for speaker_id in speaker_ids[target_property][dataset]:
                    for content_id in content_ids[target_property][dataset]:
                        speaker_path = speaker_lines[speaker_id]
                        content_path, _ = content_lines[content_id].split('|')

                        for channel_id in channel_ids[target_property][dataset]:
                            if version == 1:  # 100 classes
                                room_id, remainder = divmod(channel_id, 10)  # 10 rooms
                                # 5 noise sources, 2 filter settings
                                noise_id, filter = divmod(remainder, 2)
                            if version == 2:  # 200 classes
                                room_id, remainder = divmod(channel_id, 20)  # 10 rooms
                                # 5 noise sources, 4 filter settings
                                noise_id, filter = divmod(remainder, 4)

                            # Augmentation (adapted from LJSpeech)
                            # Step 1: add noise
                            if all_noises[noise_id] == 'quiet':
                                noise_path = 'quiet'
                            else:
                                noise_path = wav_paths[all_noises[noise_id]][np.random.randint(
                                    len(wav_paths[all_noises[noise_id]]))]  # Random noise file

                            # Step 2: convolve with room impulse response
                            rir_path = wav_paths[all_rooms[room_id]][np.random.randint(
                                len(wav_paths[all_rooms[room_id]]))]  # Random RIR

                            # Step 3: add filter
                            if filter == 0:
                                filter = "normal"
                            elif filter == 1:
                                filter = "telephone"
                            elif filter == 2:
                                filter = "lowpass"
                            elif filter == 3:
                                filter = "highpass"

                            speaker_filename = speaker_path.split('/')[-1]
                            filename = "+".join([
                                speaker_filename, content_path.split('/')[-1],
                                noise_path.split(os.sep)[-1], rir_path, filter]) + ".wav"

                            match target_property:
                                case 'speaker':
                                    class_id = speaker_id
                                case 'content':
                                    class_id = content_id
                                case 'channel':
                                    class_id = channel_id
                            if class_id not in data_dictionary[dataset]:
                                data_dictionary[dataset][class_id] = []
                            data_dictionary[dataset][class_id].append(
                                '/'.join(["wav", speaker_path.split('/')[-2], filename]))

        # Generate splits (copied from BritishIsles, could possibly be generalised)
        train_split = .8
        dev_split = .1

        for data_dict, task in [(speakers, 'speaker'), (sents, 'content'), (channels, 'channel')]:
            for dataset in ['disentanglement', 'probing']:
                train_ex = []
                dev_ex = []
                test_ex = []

                for class_idx, (class_id, utt_list) in enumerate(data_dict[dataset].items()):
                    ids = [i for i in range(len(utt_list))]
                    train_n = int(len(ids) * train_split)
                    dev_n = int(len(ids) * dev_split)

                    train_ids = random.sample(ids, train_n)
                    train_ex.extend([(class_idx, utt_list[train_id]) for train_id in train_ids])

                    ids = list(set(ids) - set(train_ids))
                    dev_ids = random.sample(ids, dev_n)
                    dev_ex.extend([(class_idx, utt_list[dev_id]) for dev_id in dev_ids])

                    test_ids = list(set(ids) - set(dev_ids))
                    test_ex.extend([(class_idx, utt_list[test_id]) for test_id in test_ids])

                for split in [('train', train_ex), ('dev', dev_ex), ('test', test_ex)]:
                    lines = []
                    for ex in split[1]:
                        lines.append(str(ex[0]) + ' ' + ex[1] + '\n')
                    with open(os.path.join(self.data_path,
                                           "-".join([dataset, task,
                                                     split[0]]) + f"-lines.txt"), 'w') as file:
                        file.writelines(lines)


class VoxCeleb(Dataset):
    """Implements VoxCeleb dataset.

    :param data_path: path to `VoxCeleb1-test` data folder
    :param trial_pairs_path: path to `veri_test2.txt`
    :param return_type: what type of data to return when indexed. Can be:
        - tuple of label, audio array of first utt, audio array of second utt (`array`)
        - tuple of label, path to first utt, path to second utt (`path`)
    """

    def __init__(self, data_path, trial_pairs_path, return_type='array'):
        self.wav_paths_1 = []
        self.wav_paths_2 = []
        self.labels = []
        self.data_path = os.path.normpath(data_path)
        self.return_type = return_type

        self.load_data(trial_pairs_path)

    def __getitem__(self, index):
        """Fetches item from dataset.

        :param index: index of item to fetch (int)
        :return: depends on `self.return_type`
        """
        label = self.labels[index]
        if (self.return_type == 'array'):
            utt_1 = read_wav(os.path.join(self.data_path, self.wav_paths_1[index]))
            utt_2 = read_wav(os.path.join(self.data_path, self.wav_paths_2[index]))
        if (self.return_type == 'path'):
            utt_1 = os.path.join(self.data_path, self.wav_paths_1[index])
            utt_2 = os.path.join(self.data_path, self.wav_paths_2[index])

        return label, utt_1, utt_2

    def __len__(self):
        """Returns the number of examples in the dataset.

        :return: number of examples in the dataset (int)
        """
        return len(self.labels)

    def load_data(self, trial_pairs_path):
        """Reads in text file with lines consisting of a label and paths to two utterances, and
        stores both labels and file paths.

        :param trial_pairs_path: path to lines file, relative to `self.data_path`
        """
        with open(os.path.join(self.data_path, trial_pairs_path), 'r') as file:
            trial_pairs = file.readlines()
        for line in trial_pairs:
            trial_pair = line.split()
            self.labels.append(int(trial_pair[0]))
            self.wav_paths_1.append(os.path.normpath(trial_pair[1]))
            self.wav_paths_2.append(os.path.normpath(trial_pair[2]))


def read_wav(wav_path):
    """Reads audio array and returns in the expected format for Speechbrain models.

    :param wav_path: absolute path to a 16000Hz wav file
    :return: audio array, in float64
    """
    sample_rate, array = wavfile.read(wav_path)
    assert sample_rate == 16000

    return array.astype(np.float64) / np.iinfo(np.int16).max
