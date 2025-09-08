import argparse
import os
import sys
import itertools
import pickle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import torchaudio

from efficient_CKA import MinibatchCKA
import utils
import features
from mydatasets import BritishIsles, VOiCES, LJSpeech


def calculate_lincka(data_path, models, feature_extractors, model_paths, device, batch_size=64):
    """Commputes pairwise layer similarity analysis using minibatch LinCKA for a list of models.

    :param data_path: path to main data directory
    :param models: list of instantiated models
    :param feature_extractors: list of feature extractors or `None` for models which don't need one
    :param model_paths: huggingface model paths without the origin
    :param device: device to put models on, and perform calculations on
    :param batch_size: batch size to use for LinCKA, defaults to 64
    :return: a two-dimensional NumPy array representing the layer similarities
    """
    dataloaders = get_dataloaders(data_path)
    modelinfo = utils.ModelInfo()
    n_layers = sum([modelinfo.n_layers(model) for model in model_paths])
    cka = MinibatchCKA(num_layers=n_layers)
    # We keep minibatches per model, to later combine after transposing and padding
    minibatches = [[] for _ in range(len(models))]
    for dataloader in dataloaders:
        for i in tqdm(range(len(dataloader)), mininterval=60, maxinterval=180):
            label, utt_array, utt_path = next(iter(dataloader))
            for i in range(len(models)):
                minibatches[i].append(obtain_activations(utt_array, utt_path,
                                      models[i], feature_extractors[i], model_paths[i], device))

            if len(minibatches[0]) == batch_size:
                for i in range(len(minibatches)):
                    minibatches[i] = transpose_and_pad_minibatch(minibatches[i])
                full_batch = list(itertools.chain(*minibatches))
                cka.update_state(full_batch)
                minibatches = [[] for i in range(len(models))]

        # We throw away remaining data, because if the remainder batch is too small,
        # it causes NaN values
    return cka.result().numpy()


def obtain_activations(utt_array, utt_path, model, feature_extractor, model_path, device):
    """Gets activations of all layers for one utterance and one model.

    :param utt_array: audio array of utterance
    :param utt_path: path to utterance
    :param model: instantiated model
    :param feature_extractor: feature extractor, or `None` for model which doesn't need one
    :param model_path: path to model
    :param device: device to put model and audio array on
    :return: list of representations per model layer for the utterance
    """
    modelinfo = utils.ModelInfo()
    model_origin = modelinfo.origin(model_path)

    match model_origin:
        case 'microsoft':
            input = feature_extractor(utt_array, padding=False,
                                      return_tensors="pt", sampling_rate=16000)
            input = input.to(device)
            model.to(device)
            with torch.no_grad():
                out = model(**input, output_hidden_states=True)
            representations = [tf.convert_to_tensor(torch.transpose(
                hidden_state, 1, 2).squeeze().cpu()) for hidden_state in out.hidden_states]
            if modelinfo.hf_name(model_path)[-2:] == 'sv':
                representations.append(tf.convert_to_tensor(
                    out.embeddings.squeeze().unsqueeze(dim=1).cpu()))
        case 'speechbrain':
            signal, _ = torchaudio.load(utt_path)
            signal = signal.to(device)
            with torch.no_grad():
                embedding, hidden_states = model.encode_batch(signal, output_hidden_states=True)
            representations = []
            model_architecture = model_path.split('-')[1]
            for rep in hidden_states[:-1]:
                if (model_architecture == 'ecapa'):
                    representations.append(tf.convert_to_tensor(rep.squeeze().cpu()))
                elif (model_architecture == 'xvect'):
                    representations.append(tf.convert_to_tensor(
                        torch.transpose(rep, 1, 2).squeeze().cpu()))
                elif (model_architecture == 'resnet'):
                    rep = torch.transpose(rep, 2, 3).squeeze()
                    # From 2D to 1D to be compatible with later code
                    rep = rep.reshape(rep.shape[0] * rep.shape[1], rep.shape[2]).cpu()
                    representations.append(tf.convert_to_tensor(rep))
            representations.append(tf.convert_to_tensor(
                hidden_states[-1].squeeze().unsqueeze(dim=1).cpu()))
            representations.append(tf.convert_to_tensor(
                embedding.squeeze().unsqueeze(dim=1).cpu()))
        case 'nvidia':
            with torch.no_grad():
                embedding, hidden_states = model.get_embedding(utt_path, output_hidden_states=True)
            representations = [hidden_state[0].squeeze() for hidden_state in hidden_states]
            representations.append(embedding.squeeze().unsqueeze(dim=1))
        case 'x':
            utt = utt_array.numpy().astype(np.float32)
            spectrogram = features.spectrogram(utt)
            fbank = features.fbank(spectrogram)
            mfccs = features.mfcc(fbank)
            representations = [spectrogram, fbank, mfccs]

    return representations


def transpose_and_pad_minibatch(minibatch):
    """ Transposes minibatch from example x layer x feature x frame to 
    layer x example x feature x frame, and applies padding for frame lengths.

    :param minibatch: a minibatch of model activations of all investigated models, structured
        as above
    :return: a transposed and padded minibatch, structured as above
    """
    minibatch_T = list(zip(*minibatch))
    padded_minibatch = []
    for layer in minibatch_T:
        padded_layer = []
        max_length = max([len(ex[0]) for ex in layer])
        repdim = len(layer[0])
        for ex in layer:
            # As we use numpy arrays below we can get an
            # issue when the representations are on the gpu
            try:
                ex = ex.cpu()
            except:
                ex = ex
            padded_ex = np.zeros((repdim, max_length))
            padded_ex[:, :len(ex[0])] = ex
            padded_layer.append(tf.convert_to_tensor(padded_ex))
        padded_minibatch.append(padded_layer)

    return padded_minibatch


def get_dataloaders(data_path):
    """Instantiates DataLoaders with examples to put through models for LinCKA. Specifically uses
    British Isles, VOiCES and augmented LJ Speech.

    :param data_path: path to main data directory
    :return: lists of DataLoaders
    """
    datasets = []
    datasets.append(BritishIsles(os.path.join(data_path, 'british-isles'),
                    'content-train-lines.txt', return_type='both', mode='read'))
    datasets.append(VOiCES(os.path.join(data_path, 'VOiCES'),
                    'train-lines.txt', return_type='both', mode='read'))
    datasets.append(LJSpeech(os.path.join(data_path, 'LJSpeech'),
                    'channel-train-lines.txt', return_type='both', mode='read'))

    return [DataLoader(dataset, batch_size=None, shuffle=True) for dataset in datasets]


def main(args):
    """Processes commandline arguments and calls `calculate_lincka()`, then pickles result.

    :param args: commandline args
    """
    print("Doing similarity analysis for", args.models, flush=True)
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    np.random.seed(607)
    torch.manual_seed(607)

    models = []
    feature_extractors = []
    model_names = []
    modelinfo = utils.ModelInfo()
    for model_path in args.models:
        model_path = modelinfo.hf_name(model_path)
        model, feature_extractor = utils.load_model(model_path, device)
        models.append(model)
        feature_extractors.append(feature_extractor)
        model_names.append(model_path)

    heatmap = calculate_lincka(args.data_path, models, feature_extractors,
                               args.models, device, args.batch_size)
    filename = f"lincka_{'+'.join(model_names)}.pkl"
    result = {'heatmap': heatmap, 'batch_size': args.batch_size}

    with open(os.path.join('results', filename), 'wb') as file:
        pickle.dump(result, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',
                        help="Path to main data directory",
                        type=str,
                        default='data')
    parser.add_argument('--models',
                        help="Paths to models to do across model similarity analysis on",
                        type=str,
                        nargs='+',
                        default=['microsoft/wavlm-base-plus-sv', 'microsoft/unispeech-sat-base-plus-sv'])
    parser.add_argument('--batch_size',
                        help="Batch size to use for batch LinCKA calculation",
                        type=int,
                        default=64)

    args = parser.parse_args()

    sys.exit(main(args))
