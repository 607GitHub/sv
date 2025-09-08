import os
import sys
from copy import deepcopy
import argparse
import pickle

import torch
from torch.utils.data import DataLoader, Subset
from torch import optim
import numpy as np

import probes
from vib import VIB, VIBConfig
import utils


def probe_model(model, feature_extractor, vib, model_path, random, task, layer,
                datasets_set, eval_split, data_path, overwrite_representations, probe,
                hidden_size, no_mdl, tiny_test, device, batch_size, epsilon=0.1):
    """Applies probing to model. By default uses Minimum Description Length probing using online
        code. Optionally uses standard probing. Also works for VIBs trained on top of other model.

    :param model: instantiated model that examples are passed through
    :param feature_extractor: feature extractor for models that require it (otherwise ``None``)
    :param vib: instantiated VIB when probing a VIB (otherwise ``None``)
    :param model_path: path to model on HuggingFace 
        (eg. 'spkrec-ecapa-voxceleb' or 'speechbrain/spkrec-ecapa-voxceleb')
    :param random: bool signifying whether to use random baseline
    :param task: attribute to train and evaluate probe on (eg. 'content')
    :param layer: layer of model to probe (integer)
    :param datasets_set: datasets set to use, see :py:func:utils.get_datasets_for_task
    :param eval_split: whether to use 'dev' or 'test' split for evaluation
    :param data_path: path to main data folder
    :param overwrite_representations: bool whether to regenerate representations (``True``) or 
        attempt to load them from a pickle (``False``)
    :param probe: string indicating probe architecture to use
    :param hidden_size: hidden size of hidden layer when MLP probe is used (integer)
    :param no_mdl: bool set to ``True`` when requiring traditional probing
    :param tiny_test: ``None`` by default—set to use a subset of the datasets for testing (integer)
    :param device: device to do probing on (cpu or gpu)
    :param batch_size: batch size to use in probing
    :param epsilon: minimum loss eval improvement required to keep training, defaults to 0.1
    :return: dictionary containing probing metrics for each seed used
    """
    modelinfo = utils.ModelInfo()

    reps_path = "_".join(["representations/" + modelinfo.hf_name(model_path),
                          "random" if random else "trained", task,
                          str(datasets_set) +
                          (f"_tiny_{tiny_test}" if tiny_test is not None else "")])

    train_dataset, eval_dataset, n_classes = utils.get_datasets_for_task(
        data_path, model_path, task, eval_split, tiny_test, datasets_set)

    if __name__ == '__main__':
        print("Training probe on representations from",
              model.__class__.__name__, "on", train_dataset.__class__.__name__)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    if model is not None:
        model.to(device)
    train_probe_data = utils.get_probe_data(model, feature_extractor, train_dataset,
                                            f"{reps_path}_train.pkl", layer, device,
                                            model_path, overwrite_representations)
    eval_probe_data = utils.get_probe_data(model, feature_extractor, eval_dataset,
                                           f"{reps_path}_{eval_split}.pkl", layer, device,
                                           model_path, overwrite_representations)

    if vib is not None:
        train_probe_data = utils.pass_through_vib(train_probe_data, vib)
        eval_probe_data = utils.pass_through_vib(eval_probe_data, vib)

    embedding_size = int(train_probe_data[0][0].shape[0])
    hidden_size = hidden_size
    probe = create_probe(probe, embedding_size, hidden_size, n_classes)

    if no_mdl:
        train_sets = [train_probe_data]
        mdl_eval_sets = None
    else:
        train_sets, mdl_eval_sets = get_subsets_for_MDL_probing(train_probe_data)
    real_eval_dataloader = DataLoader(eval_probe_data, batch_size=batch_size, shuffle=False)

    mdl_eval_losses = []  # Stores for each portion the loss on the mdl eval set
    epoch_train_losses = []  # Stores for each portion the train loss per epoch
    epoch_eval_losses = []  # Stores for each portion the eval loss on the real eval set per epoch

    probe.to(device)
    first_i = 0
    for i in range(len(train_sets)):
        # In case of a dataset that is too small to have all MDL subsets
        if len(train_sets[i]) < 1:
            first_i += 1
            continue

        train_dataloader = DataLoader(train_sets[i], batch_size=batch_size, shuffle=True)

        probe.reset()  # New probe
        optimizer = optim.AdamW(probe.parameters(), lr=0.001)
        probe.train()

        no_improvement = 0
        best_eval_loss = float('inf')
        best_probe = None

        train_losses = []
        eval_losses = []

        epoch = 0
        # Train probe until convergence
        while (no_improvement < 5):
            epoch_loss = 0

            for embs, labels in train_dataloader:
                embs = embs.to(device)
                logits = probe(embs)
                loss = loss_fn(logits.cpu(), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss

            # At end of epoch, do evaluation on actual eval set to check convergence
            probe.eval()
            eval_loss = 0
            for embs, labels in real_eval_dataloader:
                embs = embs.to(device)
                with torch.no_grad():
                    logits = probe(embs)
                    loss = loss_fn(logits.cpu(), labels)
                    eval_loss += loss
            if (eval_loss < best_eval_loss):
                # We update the best loss and probe on any improvement,
                # but stop training if it stays too small
                if ((best_eval_loss - eval_loss) > epsilon):
                    no_improvement = 0
                else:
                    no_improvement += 1
                best_eval_loss = eval_loss
                best_probe = deepcopy(probe)
            else:
                no_improvement += 1
            train_losses.append(float(epoch_loss))
            eval_losses.append(float(eval_loss))
            epoch += 1

        # Probe has converged
        epoch_train_losses.append(train_losses)
        epoch_eval_losses.append(eval_losses)

        if (mdl_eval_sets is None) or (i == len(mdl_eval_sets)):
            # All the data has been transmitted, we'll calculate accuracy
            # First condition for when no_mdl is True
            best_probe.eval()
            results = []
            for embs, labels in real_eval_dataloader:
                embs = embs.to(device)
                with torch.no_grad():
                    logits = probe(embs)
                preds = list(torch.argmax(logits, dim=-1).cpu())
                results.extend([preds[i] == labels[i] for i in range(len(preds))])
            eval_acc = float(sum(results) / len(results))
            if __name__ == '__main__':
                print("Best eval accuracy on entire dataset is", eval_acc)
            # No need to do MDL evaluation (and cannot because all the data has been transmitted)
            break

        # Calculate loss on next data portion
        mdl_eval_dataloader = DataLoader(mdl_eval_sets[i], batch_size=batch_size, shuffle=False)

        best_probe.eval()
        eval_loss = 0
        for embs, labels in mdl_eval_dataloader:
            embs = embs.to(device)
            with torch.no_grad():
                logits = best_probe(embs)
                loss = loss_fn(logits.cpu(), labels)
                eval_loss += loss

        if __name__ == '__main__':
            print("Eval loss on portion", i, "is", float(eval_loss), "after", epoch, "epochs.")
        mdl_eval_losses.append(float(eval_loss))

    if not no_mdl:
        codelength = len(train_sets[first_i]) * np.log(n_classes) + sum(mdl_eval_losses)
        uniform_codelength = len(train_dataset) * np.log(n_classes)
        compression = uniform_codelength / codelength
        if __name__ == '__main__':
            print("Compression is", compression)

        results = {'epoch_train_losses': epoch_train_losses,
                   'epoch_eval_losses': epoch_eval_losses, 'mdl_eval_losses': mdl_eval_losses,
                   'codelength': codelength, 'eval_acc': eval_acc,
                   'uniform_codelength': uniform_codelength, 'compression': compression}
    else:
        results = {'epoch_train_losses': epoch_train_losses,
                   'epoch_eval_losses': epoch_eval_losses, 'eval_acc': eval_acc}

    return results  # For when we are calling this module from another script


def get_subsets_for_MDL_probing(dataset, portions=[0.001, 0.002, 0.004, 0.008, 0.016, 0.032,
                                                   0.0625, 0.125, 0.25, 0.5, 1]):
    """Generates subsets for online MDL probing. 
        Inspiration was taken from https://github.com/lena-voita/description-length-probing.

    :param dataset: Dataset to generate subsets from
    :param portions: portion sizes, defaults to 
        [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.0625, 0.125, 0.25, 0.5, 1],
        same as in original set-up by Voita and Titov.
    :return: list of train_sets and list of eval_sets, both containing Subsets of dataset
        for MDL probing
    """
    dataset = Subset(dataset, indices=np.random.permutation(len(dataset)))  # Shuffle
    train_sets = []
    eval_sets = []
    for i in range(len(portions)):
        train_sets.append(Subset(dataset, range(
            0, int(portions[i] * len(dataset)))))  # All the data transmitted so far
        if (i != len(portions) - 1):  # Don't need to transmit data at final training part
            # Next data to be transmitted
            eval_sets.append(Subset(dataset, range(
                int(portions[i] * len(dataset)), int(portions[i + 1] * len(dataset)))))

    return train_sets, eval_sets


def create_probe(probe_type, embedding_size, hidden_size, n_classes):
    """Generates new probe to be trained.

    :param probe_type: probe architecture to use ('MLP' or 'linear')
    :param embedding_size: dimensionality of input
    :param hidden_size: dimensionality of hidden layer in case of 'MLP'
    :param n_classes: amount of classes to map to
    :raises ValueError: when ``probe_type`` is not recognised
    :return: an instantiated probe
    """
    match probe_type:
        case 'MLP':
            probe = probes.MultiLayerPerceptron(embedding_size, hidden_size, n_classes)
        case 'linear':
            probe = probes.LinearClassifier(embedding_size, n_classes)
        case _:
            raise ValueError("Probe architecture not supported")

    return probe


def main(args):
    """Loads required model and optionally VIB and calls ``probe_model`` with required arguments.

    :param args: commandline arguments
    :return: results dictionary returned by ``probe_model``
    """
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

    print(f"Training and evaluating {args.probe} probe on {f'stage {args.vib_stage} {args.vib_task} VIB encoding of ' if args.vib_stage is not None else ''}layer {args.layer} of {args.model} for {
          args.task} on datasets set {args.datasets_set} {"with randomly initialised weights" if args.random else "with trained weights"} with seed {args.seed}, batch size {args.batch_size}.")

    modelinfo = utils.ModelInfo()

    # If run from run_probing_experiments.py, the seeds will already have been set
    if __name__ == '__main__':
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    model, feature_extractor = utils.load_model(args.model, device, random=args.random)

    if args.vib_stage is not None:
        layer = modelinfo.layer_idx(args.model, args.layer)
        vib_name = "_".join([args.vib_stage, args.vib_task, str(args.vib_datasets_set),
                             modelinfo.hf_name(args.model), str(layer)])

        with open(
            os.path.join(
                'vib', "_".join([vib_name, str(args.seed) +
                                 (f'_tiny_{args.tiny_test}' if args.tiny_test is not None else ''),
                                 args.eval_split]) + ".pth"), 'rb') as file:
            state_dict = torch.load(file, map_location=torch.device('cpu'))
        with open(os.path.join('vib', f'{vib_name}_cfg.pkl'), 'rb') as file:
            vib_config = pickle.load(file)
        vib = VIB(vib_config)
        vib.load_state_dict(state_dict)
    else:
        vib = None

    results = probe_model(model, feature_extractor, vib, modelinfo.hf_name(args.model),
                          args.random, args.task, args.layer, args.datasets_set, args.eval_split,
                          args.data_path, args.overwrite_representations, args.probe,
                          args.hidden_size, args.no_mdl, args.tiny_test,
                          device, args.batch_size)

    if __name__ == '__main__':
        if args.vib_stage is not None:
            results_filename = '_'.join(
                (args.vib_stage, args.vib_task, modelinfo.hf_name(args.model), args.task))
            if args.probe == 'linear':
                results_filename += '_linear'
            with open('results/' + results_filename + '.pkl', 'wb') as file:
                pickle.dump(results, file)
        return 0
    else:
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',
                        help="Path to main data directory",
                        type=str,
                        default='data')
    parser.add_argument('--task',
                        help="Task to train on",
                        type=str,
                        default='content',
                        choices=['content', 'channel', 'speaker'])
    parser.add_argument('--model',
                        help="Model to probe: see utils.ModelInfo for supported models",
                        type=str,
                        default='microsoft/wavlm-base-plus-sv')
    parser.add_argument('--vib_stage',
                        help="VIB stage to use",
                        type=str,
                        choices=['1', '2'],
                        default=None)
    parser.add_argument('--vib_task', help="VIB task to use",
                        type=str,
                        choices=['content', 'channel', 'speaker'],
                        default='content')
    parser.add_argument('--layer',
                        help="Which layer of the model to train the probe on",
                        type=int,
                        default=-1)
    parser.add_argument('--probe', help="Architecture of probe",
                        type=str,
                        choices=['MLP', 'linear'],
                        default='MLP')
    parser.add_argument('--hidden_size',
                        help="Size of hidden layer in MLP",
                        type=int,
                        default=500)
    parser.add_argument('--random',
                        help="Use randomly initialised weights",
                        action='store_true')
    parser.add_argument('--no_mdl',
                        help="Do regular probing, only reporting accuracy",
                        action="store_true")
    parser.add_argument('--overwrite_representations',
                        help="Put data through model even if pickle exists",
                        action="store_true")
    parser.add_argument('--eval_split',
                        help="Whether to evaluate on the dev or the test set",
                        type=str,
                        choices=['dev', 'test'],
                        default='dev')
    parser.add_argument('--seed',
                        help="Set seed to use for pytorch and numpy",
                        type=int,
                        default=0)
    parser.add_argument('--datasets_set',
                        help="What set of datasets to use",
                        type=int,
                        default=1)
    parser.add_argument('--vib_datasets_set',
                        help="What set of datasets was used for training to be probed vibs",
                        type=int,
                        default=3)
    parser.add_argument('--batch_size',
                        help="Batch size to use",
                        type=int,
                        default=128)
    parser.add_argument('--tiny_test',
                        help="Use subset of n random data examples",
                        type=int,
                        default=None)

    args = parser.parse_args()

    sys.exit(main(args))
