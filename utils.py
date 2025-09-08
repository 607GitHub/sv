# Stores classes and methods that are used by multiple modules
# Imports that are commented out are imported elsewhere in the module, to save time when they are not required
from collections import namedtuple
import os
import pickle
import random

# from transformers import Wav2Vec2FeatureExtractor, AutoConfig, WavLMModel, WavLMForXVector, UniSpeechSatModel, UniSpeechSatForXVector
# from torch.utils.data import Subset
# from torch.utils.data import DataLoader
# import torch
# import torchaudio
import numpy as np

# from mydatasets import BritishIsles, VOiCES, LJSpeech, VCTK, SCC
# from speechbrain.inference.speaker import EncoderClassifier
# import nemo.collections.asr as nemo_asr
import features


class ModelInfo:
    """Stores info about supported models and implements methods to get back information.
    """

    def __init__(self):
        Model = namedtuple('Model', ['origin', 'hf_name', 'clean_name',
                           'layer_count', 'baseline', 'input_type'])
        # Models are indexed by their huggingface name
        self.models = {'wavlm-base-plus-sv':
                       Model('microsoft', 'wavlm-base-plus-sv', 'WavLM (SV)', 14, 2, 'array'),
                       'wavlm-base-plus':
                       Model('microsoft', 'wavlm-base-plus', 'WavLM (general)', 13, 2, 'array'),
                       'unispeech-sat-base-plus-sv':
                       Model('microsoft', 'unispeech-sat-base-plus-sv', 'UniSpeech-SAT (SV)',
                             14, 2, 'array'),
                       'unispeech-sat-base-plus':
                       Model('microsoft', 'unispeech-sat-base-plus', 'UniSpeech-SAT (general)',
                             13, 2, 'array'),
                       'spkrec-ecapa-voxceleb':
                       Model('speechbrain', 'spkrec-ecapa-voxceleb', 'ECAPA-TDNN', 7, 1, 'path'),
                       'spkrec-xvect-voxceleb':
                       Model('speechbrain', 'spkrec-xvect-voxceleb', 'x-vector', 17, 3, 'path'),
                       'spkrec-resnet-voxceleb':
                       Model('speechbrain', 'spkrec-resnet-voxceleb', 'ResNet', 6, 1, 'path'),
                       'speakerverification_en_titanet_large':
                       Model('nvidia', 'speakerverification_en_titanet_large', 'TitaNet',
                             6, 1, 'path'),
                       'features': Model('x', 'features', 'Feature baseline', 4, None, 'array')
                       }

    #
    # Also used internally to get right key for models dictionary
    def hf_name(self, model, include_origin=False):
        """Takes a huggingface path with or without origin, and returns with or without origin, as
        requested.

        :param model: a path/name of a huggingface model, with or without origin
        :param include_origin: whether to include the origin in the output, defaults to False
        :raises ValueError: when `model` is not recognised as a supported model
        :return: huggingface path of the model, with or without origin following `include_origin`
        """
        if model in self.models.keys():
            if include_origin:
                return '/'.join([self.models[model].origin, model])
            else:
                return model
        try:
            if model.split('/')[1] in self.models.keys():
                if include_origin:
                    return model
                else:
                    return model.split('/')[1]
            else:
                raise ValueError(model, 'is not a model known by utils.ModelInfo.')
        except:
            raise ValueError(model, 'is not a model known by utils.ModelInfo.')

    def n_layers(self, model):
        """Returns the amount of layers in a model.

        :param model: huggingface path of the model, with or without origin
        :return: amount of layers in the model (int)
        """
        return self.models[self.hf_name(model)].layer_count

    def layer_idx(self, model, idx):
        """Returns the absolute index of a layer within a model (eg. to convert from `-1`).

        :param model: huggingface path of the model, with or without origin
        :param idx: layer index (relative or absolute) (int or string)
        :return: absolute layer index (int)
        """
        return range(self.n_layers(model))[int(idx)]

    def feature_baseline_layer(self, model):
        """Returns the layer index of the feature baseline for a model, within `x/features`.

        :param model: huggingface path of the model, with or without origin
        :return: layer index of the feature baseline for the model
        """
        return self.models[self.hf_name(model)].baseline

    def clean_name(self, model):
        """Returns nice-looking name for model, to be used in visualizations.

        :param model: huggingface path of the model, with or without origin
        :return: nice-looking name for model
        """
        return self.models[self.hf_name(model)].clean_name

    def origin(self, model):
        """Returns huggingface origin of model

        :param model: huggingface path of the model, with or without origin
        :return: origin of the model
        """
        return self.models[self.hf_name(model)].origin

    def input_type(self, model):
        """Returns the required input type for a model (`path`) or (`array`)

        :param model: huggingface path of the model, with or without origin
        :return: required input type for the model
        """
        return self.models[self.hf_name(model)].input_type


def load_model(model_path, device, random=False):
    """Loads one of the supported models and returns it instantiated.

    :param model_path: huggingface path of the model, with or without origin
    :param device: device to put the model on
    :param random: whether to load the random baseline instead of the trained checkpoint, 
        defaults to False
    :raises ValueError: if the origin is not recognised
    :return: the instantiated model, and its feature extractor (which may be `None`)
    """
    modelinfo = ModelInfo()
    model_path = modelinfo.hf_name(model_path, include_origin=True)
    model_origin = modelinfo.origin(model_path)

    match model_origin:
        case 'microsoft':
            from transformers import Wav2Vec2FeatureExtractor, AutoConfig, WavLMModel, WavLMForXVector, UniSpeechSatModel, UniSpeechSatForXVector

            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
            model_architecture = modelinfo.hf_name(model_path).split('-')[0]
            if (model_path[-2:] == 'sv'):
                if (model_architecture == 'wavlm'):
                    model_class = WavLMForXVector
                elif (model_architecture == 'unispeech'):
                    model_class = UniSpeechSatForXVector
            else:
                if (model_architecture == 'wavlm'):
                    model_class = WavLMModel
                elif (model_architecture == 'unispeech'):
                    model_class = UniSpeechSatModel
            if (random):
                config = AutoConfig.from_pretrained(model_path)
                model = model_class(config)
            else:
                model = model_class.from_pretrained(model_path)

        case 'speechbrain':
            from speechbrain.inference.speaker import EncoderClassifier

            model = EncoderClassifier.from_hparams(source=model_path, run_opts={"device": device})
            if (random):
                # eg. spkrec-xvect-voxceleb.pkl
                with open(os.path.join('speechbrain_files', modelinfo.hf_name(model_path) + '.pkl'), 'rb') as file:
                    module_dict = pickle.load(file)
                model.mods = module_dict
                model.hparams.modules = module_dict
            feature_extractor = None

        case 'nvidia':
            import nemo.collections.asr as nemo_asr

            model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_path)
            feature_extractor = None

        case 'x':
            if model_path.split('/')[1] == 'features':
                model = None
                feature_extractor = None

        case _:
            raise ValueError("Model path incorrect or not supported")

    return model, feature_extractor


def get_datasets_for_task(data_path, model_path, task, eval_split, tiny_test=None, datasets_set=1):
    """Loads train and eval datasets.

    :param data_path: path to main data directory
    :param model_path: huggingface path of the model, with or without origin
    :param task: attribute for classification (`speaker`, `content` or `channel`)
    :param eval_split: what evaluation split to get (`dev` or `test`)
    :param tiny_test: if set to an integer, loads Subset of the full dataset, defaults to None
    :param datasets_set: what set of datasets to use, defaults to 1, meaning:
        - 1: British Isles and VOiCES
        - 2: VCTK and augmented LJ Speech
        - 3: SCC, disentanglement
        - 4: SCC, probing
    :raises ValueError: if the task is not recognised
    :raises ValueError: if the datasets set is not recognised
    :return: a train dataset, an evaluation dataset (both Subsets if `tiny_test` is not `None`),
        and the amount of classes in the dataset
    """
    from mydatasets import BritishIsles, VOiCES, LJSpeech, VCTK, SCC
    from torch.utils.data import Subset

    modelinfo = ModelInfo()
    match datasets_set:
        case 1:
            match task:
                case 'content':
                    train_dataset = BritishIsles(
                        os.path.join(data_path, 'british-isles'),
                        'content-train-lines.txt', return_type=modelinfo.input_type(model_path),
                        mode='read')
                    eval_dataset = BritishIsles(
                        os.path.join(data_path, 'british-isles'),
                        f'content-{eval_split}-lines.txt',
                        return_type=modelinfo.input_type(model_path), mode='read')
                case 'channel':
                    train_dataset = VOiCES(
                        os.path.join(data_path, 'VOiCES'), 'train-lines.txt',
                        return_type=modelinfo.input_type(model_path), mode='read')
                    eval_dataset = VOiCES(
                        os.path.join(data_path, 'VOiCES'), f'{eval_split}-lines.txt',
                        return_type=modelinfo.input_type(model_path), mode='read')
                case 'speaker':
                    train_dataset = BritishIsles(
                        os.path.join(data_path, 'british-isles'),
                        'speaker-train-lines.txt', return_type=modelinfo.input_type(model_path),
                        mode='read')
                    eval_dataset = BritishIsles(
                        os.path.join(data_path, 'british-isles'),
                        f'speaker-{eval_split}-lines.txt',
                        return_type=modelinfo.input_type(model_path), mode='read')
                case _:
                    raise ValueError("Task not implemented")
        case 2:
            match task:
                case 'content':
                    train_dataset = LJSpeech(
                        os.path.join(data_path, 'LJSpeech'), 'content-train-lines.txt',
                        return_type=modelinfo.input_type(model_path), mode='read')
                    eval_dataset = LJSpeech(
                        os.path.join(data_path, 'LJSpeech'), f'content-{eval_split}-lines.txt',
                        return_type=modelinfo.input_type(model_path), mode='read')
                case 'channel':
                    train_dataset = LJSpeech(
                        os.path.join(data_path, 'LJSpeech'), 'channel-train-lines.txt',
                        return_type=modelinfo.input_type(model_path), mode='read')
                    eval_dataset = LJSpeech(
                        os.path.join(data_path, 'LJSpeech'), f'channel-{eval_split}-lines.txt',
                        return_type=modelinfo.input_type(model_path), mode='read')
                case 'speaker':
                    train_dataset = VCTK(
                        os.path.join(data_path, 'VCTK'), 'train-lines.txt',
                        return_type=modelinfo.input_type(model_path), mode='read')
                    eval_dataset = VCTK(
                        os.path.join(data_path, 'VCTK'), f'{eval_split}-lines.txt',
                        return_type=modelinfo.input_type(model_path), mode='read')
                case _:
                    raise ValueError("Task not implemented")
        case 3 | 4:
            if datasets_set == 3:
                use = 'disentanglement'
            elif datasets_set == 4:
                use = 'probing'
            match task:
                case 'content':
                    train_dataset = SCC(
                        os.path.join(data_path, 'SCC'),
                        f'{use}-content-train-lines.txt',
                        return_type=modelinfo.input_type(model_path), mode='read')
                    eval_dataset = SCC(
                        os.path.join(data_path, 'SCC'),
                        f'{use}-content-{eval_split}-lines.txt',
                        return_type=modelinfo.input_type(model_path), mode='read')
                case 'channel':
                    train_dataset = SCC(
                        os.path.join(data_path, 'SCC'),
                        f'{use}-channel-train-lines.txt',
                        return_type=modelinfo.input_type(model_path), mode='read')
                    eval_dataset = SCC(
                        os.path.join(data_path, 'SCC'),
                        f'{use}-channel-{eval_split}-lines.txt',
                        return_type=modelinfo.input_type(model_path), mode='read')
                case 'speaker':
                    train_dataset = SCC(
                        os.path.join(data_path, 'SCC'),
                        f'{use}-speaker-train-lines.txt',
                        return_type=modelinfo.input_type(model_path), mode='read')
                    eval_dataset = SCC(
                        os.path.join(data_path, 'SCC'),
                        f'{use}-speaker-{eval_split}-lines.txt',
                        return_type=modelinfo.input_type(model_path), mode='read')
                case _:
                    raise ValueError("Task not implemented")
        case _:
            raise ValueError("Datasets set unrecognised")
    n_classes = train_dataset.n_classes()
    if (tiny_test is not None):
        train_dataset = Subset(train_dataset, random.sample(
            list(np.arange(len(train_dataset))), tiny_test))
        eval_dataset = Subset(eval_dataset, random.sample(
            list(np.arange(len(eval_dataset))), tiny_test))

    return train_dataset, eval_dataset, n_classes


def get_probe_data(model, feature_extractor, dataset, reps_path, layer, device, model_path,
                   overwrite_reps):
    """Gets dataset to use for probing, with tuples of model activations and label.

    :param model: instantiated model 
    :param feature_extractor: feature extractor for model, or `None`
    :param dataset: instantiated dataset for certain split
    :param reps_path: path where the model representations should be stored
    :param layer: what layer of the model to get the activations of
    :param device: c if generating new representations
    :param model_path: huggingface path of the model, with or without origin
    :param overwrite_reps: whether to compute new representations even if a pickle can be restored
    :return: a list of tuples of probe input and label
    """
    if not overwrite_reps:
        try:
            with open(reps_path, 'rb') as file:
                all_representations = pickle.load(file)
            print("Representations loaded from", reps_path)
        except FileNotFoundError:
            print("No pickle with representations found.")

            overwrite_reps = True

    if overwrite_reps:
        all_representations = obtain_representations(
            model, feature_extractor, model_path, dataset, device)
        if not os.path.exists("representations"):
            os.mkdir("representations")
        with open(reps_path, 'wb') as file:
            pickle.dump(all_representations, file)

    probe_data = []

    for representations, label in all_representations:
        probe_input = representations[layer]
        probe_data.append((probe_input, label))

    return probe_data


def obtain_representations(model, feature_extractor, model_path, dataset, device):
    """Obtains activations of all layers for a model and dataset.

    :param model: instantiated model 
    :param feature_extractor: feature extractor for model, or `None`
    :param model_path: huggingface path of the model, with or without origin
    :param dataset: instantiated dataset containing utterances and labels
    :param device: what device to put the model and inputs on
    :return: list of tuples of complete model representations and labels
    """
    import torch
    import torchaudio
    from torch.utils.data import DataLoader

    if model is not None:  # For feature baseline
        model.to(device)
    data = []

    modelinfo = ModelInfo()
    origin = modelinfo.origin(model_path)

    dataloader = DataLoader(dataset, batch_size=None, shuffle=False)
    print("Obtaining representations...")
    if (origin == 'microsoft'):
        for label, utt in dataloader:
            input = feature_extractor(utt, padding=False, return_tensors="pt", sampling_rate=16000)
            input = input.to(device)
            with torch.no_grad():
                representations = model(**input, output_hidden_states=True)
            pooled_representations = [torch.mean(hidden_state.squeeze(), dim=0).cpu()
                                      for hidden_state in representations.hidden_states]
            if (model_path[-2:] == 'sv'):  # Add speaker embedding
                pooled_representations.append(representations.embeddings.squeeze().cpu())
            data.append((pooled_representations, label))

    elif (origin == 'speechbrain'):
        model_architecture = model_path.split('-')[1]
        model.eval()
        for label, utt in dataloader:
            signal, _ = torchaudio.load(utt)
            signal = signal.to(device)
            with torch.no_grad():
                embedding, hidden_states = model.encode_batch(signal, output_hidden_states=True)
            pooled_representations = []
            for rep in hidden_states[:-1]:
                if (model_architecture == 'ecapa'):
                    pooled_representations.append(torch.mean(rep.squeeze(), dim=-1).cpu())
                elif (model_architecture == 'xvect'):
                    pooled_representations.append(torch.mean(rep.squeeze(), dim=0).cpu())
                elif (model_architecture == 'resnet'):
                    pooled_representations.append(torch.mean(
                        rep.squeeze(), dim=1).view(-1).cpu())  # From 2D to 1D
            pooled_representations.append(hidden_states[-1].squeeze().cpu())  # Already pooled
            pooled_representations.append(embedding.squeeze().squeeze().cpu())
            data.append((pooled_representations, label))

    elif (origin == 'nvidia'):
        for label, utt in dataloader:
            with torch.no_grad():
                embedding, hidden_states = model.get_embedding(utt, output_hidden_states=True)
            pooled_representations = [torch.mean(
                hidden_state[0].squeeze(), dim=-1).cpu() for hidden_state in hidden_states]
            pooled_representations.append(embedding.squeeze())
            data.append((pooled_representations, label))

    elif (origin == 'x'):
        for label, utt in dataloader:
            pooled_features = []
            utt = utt.numpy().astype(np.float32)
            spectrogram = features.spectrogram(utt)
            pooled_features.append(np.mean(spectrogram, axis=1))
            fbank = features.fbank(spectrogram)
            pooled_features.append(np.mean(fbank, axis=1))
            mfccs = features.mfcc(fbank)
            pooled_mfccs = np.mean(mfccs, axis=1)
            pooled_features.append(pooled_mfccs)
            xvector_fbank = features.fbank(spectrogram, n_mels=24)
            pooled_features.append(np.mean(xvector_fbank, axis=1))
            data.append((pooled_features, label))

    return data


def pass_through_vib(dataset, vib):
    """Passes representations of probing dataset through VIB and returns new probing dataset.

    :param dataset: list consisting of tuples of model activations and labels
    :param vib: instantiated VIB
    :return: list consisting of tuples of VIB activations and labels
    """
    import torch

    print("Passing representations through VIB...")
    vib.eval()
    probe_data = []
    for (rep, label) in dataset:
        with torch.no_grad():
            _, mu, _ = vib(rep, encoder_only=True)
        probe_data.append((mu, label))

    return probe_data
