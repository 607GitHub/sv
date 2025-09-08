import sys
import os
import argparse
import pickle

import torch
from torch.utils.data import DataLoader
import torchaudio

from mydatasets import VoxCeleb
import measures
from vib import VIB, VIBConfig
import utils


def evaluate_sv_model(model, feature_extractor, vib, model_path, dataset,
                      overwrite_representations, device='cpu'):
    """Evaluates a speaker verification model on speaker verification using VoxCeleb1-O. Also
        allows evaluating a VIB trained on top of a speaker verification model.

    :param model: instantiated model that examples are passed through
    :param feature_extractor: feature extractor for models that require it (otherwise ``None``)
    :param vib: instantiated VIB when probing a VIB (otherwise ``None``)
    :param model_path: path to model on HuggingFace 
    :param dataset: VoxCeleb dataset
    :param overwrite_representations: bool whether to regenerate representations (``True``) or 
        attempt to load them from a pickle (``False``)
    :param device: device to do probing on (cpu or gpu), defaults to 'cpu'
    :return: returns EER and the score threshold at which the EER was reached
    """

    print("Evaluating", model_path)
    modelinfo = utils.ModelInfo()

    reps_path = "representations/" + "_".join([modelinfo.hf_name(model_path), "voxceleb"])
    dataset = get_eval_data(model, feature_extractor, model_path,
                            dataset, reps_path, device, overwrite_representations)

    if vib is not None:
        dataset = pass_through_vib(dataset, vib)

    dataloader = DataLoader(dataset, batch_size=None, shuffle=False)

    all_scores = []
    all_labels = []

    cosine_sim = torch.nn.CosineSimilarity(dim=-1)

    for label, emb_1, emb_2 in dataloader:
        score = cosine_sim(emb_1, emb_2)
        all_scores.append(score.cpu().squeeze())
        all_labels.append(label)

    eer, threshold = measures.eer(all_labels, all_scores)

    return eer, threshold


def get_eval_data(model, feature_extractor, model_path, dataset, reps_path, device, overwrite_reps):
    """Loads data to use to obtain EER: pairs of speaker embeddings and their ground truth.

    :param model: instantiated model that examples are passed through
    :param feature_extractor: feature extractor for models that require it (otherwise ``None``)
    :param model_path: path to model on HuggingFace 
    :param dataset: VoxCeleb dataset
    :param reps_path: path where pickle with embeddings should be loaded from / stored to
    :param device: device to put model on (cpu or gpu)
    :param overwrite_reps: bool whether to regenerate representations (``True``) or 
        attempt to load them from a pickle (``False``)
    :return: list of tuples consisting of target label and two speaker embeddings
    """
    if not overwrite_reps:
        try:
            with open(reps_path, 'rb') as file:
                eval_data = pickle.load(file)
            print("Embeddings loaded from", reps_path)
        except:
            overwrite_reps = True

    if overwrite_reps:
        eval_data = pass_through_model(model, feature_extractor, model_path, dataset, device)
        with open(reps_path, 'wb') as file:
            pickle.dump(eval_data, file)

    return eval_data


def pass_through_model(model, feature_extractor, model_path, dataset, device):
    """Generates speaker embeddings for each example in dataset.

    :param model: instantiated model that examples are passed through
    :param feature_extractor: feature extractor for models that require it (otherwise ``None``)
    :param model_path: path to model on HuggingFace 
    :param dataset: VoxCeleb dataset
    :param device: device to put model on (cpu or gpu)
    :return: list of tuples consisting of target label and two speaker embeddings
    """
    model.to(device)
    data = []
    modelinfo = utils.ModelInfo()
    origin = modelinfo.origin(model_path)

    dataloader = DataLoader(dataset, batch_size=None, shuffle=False)
    print("Obtaining embeddings...")
    for label, utt_1, utt_2 in dataloader:
        utt_embs = []
        for utt in utt_1, utt_2:
            if (origin == 'microsoft'):
                input = feature_extractor(
                    utt, padding=False, return_tensors="pt", sampling_rate=16000)
                input = input.to(device)
                with torch.no_grad():
                    embedding = model(**input).embeddings.cpu()

            elif (origin == 'speechbrain'):
                signal, _ = torchaudio.load(utt)
                signal = signal.to(device)
                with torch.no_grad():
                    embedding = model.encode_batch(signal).cpu()

            elif (origin == 'nvidia'):
                with torch.no_grad():
                    embedding = model.get_embedding(utt).cpu()

            utt_embs.append(embedding)
        data.append((label, utt_embs[0], utt_embs[1]))

    return data


def pass_through_vib(dataset, vib):
    """Passes speaker embeddings from each examples through VIB encoder.

    :param dataset: Dataset containing (label, embedding_1, embedding_2) tuples
    :param vib: VIB model
    :return: list of tuples consisting of target label and two VIB encodings
    """
    print("Passing representations through VIB...")
    vib.eval()
    eval_data = []
    for label, emb_1, emb_2 in dataset:
        with torch.no_grad():
            _, mu_1, _ = vib(emb_1, encoder_only=True)
            _, mu_2, _ = vib(emb_2, encoder_only=True)
        eval_data.append((label, mu_1, mu_2))

    return eval_data


def main(args):
    """Instantiates speaker verification model, VoxCeleb and optionally VIB, and calls
        ``evaluate_sv_model`` with required arguments.

    :param args: commandline arguments
    :return: nothing
    """
    print(args)
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

    modelinfo = utils.ModelInfo()
    model_path = args.model

    dataset = VoxCeleb(os.path.join(args.data_path, 'VoxCeleb1-test'),
                       'veri_test2.txt', return_type=modelinfo.input_type(model_path))

    model, feature_extractor = utils.load_model(args.model, device)

    if args.vib_stage is not None:
        vib_name = "_".join([args.vib_stage, args.vib_task, str(args.vib_datasets_set),
                             modelinfo.hf_name(model_path)])
        with open(os.path.join('vib', "_".join([vib_name, "0", "test"]) + ".pth"), 'rb') as file:
            state_dict = torch.load(file, map_location=torch.device('cpu'))
        with open(os.path.join('vib', "_".join([vib_name, "cfg"]) + ".pkl"), 'rb') as file:
            vib_config = pickle.load(file)

        vib = VIB(vib_config)
        vib.load_state_dict(state_dict)
    else:
        vib = None

    print(evaluate_sv_model(model, feature_extractor, vib, model_path,
          dataset, args.overwrite_representations, device))

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',
                        help="Path to main data directory",
                        type=str,
                        default='data')
    parser.add_argument('--model',
                        help="Model to evaluate: huggingface path",
                        type=str,
                        default='microsoft/wavlm-base-plus-sv')
    parser.add_argument('--vib_stage',
                        help="VIB stage to use",
                        type=str,
                        choices=['1', '2'],
                        default=None)
    parser.add_argument('--vib_task',
                        help="VIB task to use",
                        type=str,
                        choices=['speaker', 'content', 'channel'],
                        default='speaker')
    parser.add_argument('--vib_datasets_set',
                        help="What set of datasets was used for training to be evaluated vibs",
                        type=int, default=3)
    parser.add_argument('--overwrite_representations',
                        help="Put data through model even if pickle exists",
                        action="store_true")

    args = parser.parse_args()

    sys.exit(main(args))
