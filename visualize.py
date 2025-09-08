import pickle
import argparse
import sys
import os
import datetime
from math import ceil

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import librosa
# import torch (done below because it takes long and is only necessary for a few plot types)

from mydatasets import read_wav
import features
import utils
from vib import VIB, VIBConfig


def plot_layer_results(results_path, models, tasks, eval_split, metric, multitask,
                       regenerate_legend, ymax, datasets_set):
    """Plot compression, accuracy or codelength for all layers of a model.

    :param results_path: path to directory saving experiment results
    :param models: names of models to save results from
    :param tasks: attributes to show results for
    :param eval_split: evaluation split to show results for
    :param metric: metric to show
    :param multitask: whether to plot all tasks in one plot or in separate plots
    :param regenerate_legend: whether to save the legend in case of a multitask plot
    :param ymax: the ymax for compression and codelength (may be `None`)
    :param datasets_set: datasets set for which to show results
    """
    uniform_codelength_error = "Uniform codelengths of features and model don't \
        correspond, were different datasets used?"
    modelinfo = utils.ModelInfo()

    for model in models:
        data = {}
        for task in tasks:
            data[task] = {'trained': [], 'random': [], 'feature': []}
            for layer in range(modelinfo.n_layers(model)):
                for weights in ['trained', 'random']:
                    path = os.path.join(
                        results_path,
                        "_".join([model, weights, task, str(layer), str(datasets_set),
                                  eval_split]) + ".pkl")
                    try:
                        with open(path, 'rb') as file:
                            results = pickle.load(file)
                    except:
                        print(path, "can not be found, skipping.")
                        continue
                    if (metric == 'compression'):
                        data[task][weights].append(calculate_mean_and_std(results, 'compression'))
                    elif (metric == 'accuracy'):
                        data[task][weights].append(calculate_mean_and_std(results, 'eval_acc'))
                    elif (metric == 'codelength'):
                        data[task][weights].append(calculate_mean_and_std(results, 'codelength'))

            baseline_path = os.path.join(
                results_path,
                "_".join(["features", "trained", task,
                          str(modelinfo.feature_baseline_layer(model)), str(datasets_set),
                          eval_split]) + ".pkl")
            with open(baseline_path, 'rb') as file:
                feature_results = pickle.load(file)
            if (metric == 'compression'):
                mean, std = calculate_mean_and_std(feature_results, 'compression')
                feature_uniform_codelength = list(
                    feature_results.values())[0]['uniform_codelength']
                uniform_codelength = list(results.values())[0]['uniform_codelength']
                assert np.isclose(
                    uniform_codelength, feature_uniform_codelength), uniform_codelength_error
                # Layer number is not applicable
                data[task]['feature'] = [(mean, std)] * modelinfo.n_layers(model)
            elif (metric == 'accuracy'):
                data[task]['feature'] = [calculate_mean_and_std(
                    feature_results, 'eval_acc')] * modelinfo.n_layers(model)
            elif (metric == 'codelength'):
                data[task]['feature'] = [calculate_mean_and_std(
                    feature_results, 'codelength')] * modelinfo.n_layers(model)

        if not multitask:
            for task in tasks:
                fig, ax = plt.subplots()
                for setting in ['trained', 'random', 'feature']:
                    layers = np.arange(len(data[task][setting]))
                    means = [x[0] for x in data[task][setting]]
                    stds = [x[1] for x in data[task][setting]]
                    ax.errorbar(layers, means, stds, label=setting)
                ax.legend()

                if (metric == 'codelength'):
                    metric_name = 'codelength'
                elif (metric == 'accuracy'):
                    metric_name = 'acc'
                elif (metric == 'compression'):
                    metric_name = 'MDL'
                ax.set_xlabel('Layer')
                if (metric == 'compression'):
                    ax.set_ylabel('Compression')
                    ax.set_ylim(ymin=0)
                elif (metric == 'codelength'):
                    ax.set_ylabel('Codelength (bits)')
                elif (metric == 'accuracy'):
                    ax.set_ylabel('Evaluation accuracy')
                    ax.set_ylim(0, 1)
                image_path = f"plots/{metric_name}_per_layer_{model}_{task}_{datasets_set}_{eval_split}.pdf"
                plt.savefig(image_path,  bbox_inches='tight')
        else:
            fig, ax = plt.subplots()
            # Colours from https://www.nki.nl/about-us/responsible-research/guidelines-color-blind-friendly-figures/
            colors = {'speaker': '#4477AA', 'content': '#228833', 'channel': '#AA3377'}
            for task in tasks:
                color = colors[task]
                for setting, linestyle in [('trained', '-'), ('random', '--'), ('feature', ':')]:
                    layers = np.arange(len(data[task][setting])) + 1
                    means = [x[0] for x in data[task][setting]]
                    stds = [x[1] for x in data[task][setting]]
                    ax.errorbar(
                        layers, means, stds, label=f"{task.capitalize()} ({setting})",
                        color=color, linestyle=linestyle)
            if regenerate_legend:
                plt.rcParams["legend.frameon"] = False
                legend = plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.savefig(f"plots/per_layer_legend_{datasets_set}.pdf",
                            bbox_inches=legend.get_window_extent().transformed(
                                fig.dpi_scale_trans.inverted()))
                legend.remove()
            ax.set_xlabel('Layer')
            if (metric == 'compression'):
                ax.set_ylabel('Compression')
                ax.set_ylim(ymin=0)
                if ymax is not None:
                    ax.set_ylim(ymax=ymax)
            elif (metric == 'codelength'):
                ax.set_ylabel('Codelength (bits)')
            elif (metric == 'accuracy'):
                ax.set_ylabel('Evaluation accuracy')
                ax.set_ylim(0, 1)
            if (metric == 'codelength'):
                metric_name = 'codelength'
            elif (metric == 'accuracy'):
                metric_name = 'acc'
            elif (metric == 'compression'):
                metric_name = 'MDL'
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            image_path = os.path.join("plots",
                                      "_".join([metric_name, "per", "layer", model,
                                                str(datasets_set), eval_split]) + ".pdf")
            plt.savefig(image_path,  bbox_inches='tight')

# Per given task, shows results for all model_settings


def plot_portion_results(results_path, tasks, model_settings, eval_split, metric, datasets_set):
    """Plots MDL results for the different portions of online code, for a list of model settings.

    :param results_path: ath to directory saving experiment results
    :param tasks: attributes to show results for
    :param model_settings: list of model settings to compare. For example: 
        [spkrec-ecapa-voxceleb/trained/-1,
        spkrec-ecapa-voxceleb/trained/-2,
        spkrec-ecapa-voxceleb/random/-1
        ]
    :param eval_split: evaluation split to show results for
    :param metric: metric to show (`mdl_loss`, `eval_loss` or `train_loss`)
    :param datasets_set: datasets set for which to show results
    """
    plt.figure(figsize=(8, 4.8))
    modelinfo = utils.ModelInfo()
    for task in tasks:
        fig, ax = plt.subplots()
        ax.set_xlabel('Train set size')
        portions = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.0625, 0.125, 0.25, 0.5]
        if metric == 'eval_loss' or metric == 'train_loss':
            portions.append(1)  # MDL loss is not calculated over final portion
        ax.set_xticks(np.arange(len(portions)))
        ax.set_xticklabels(portions)
        if metric == 'mdl_loss':
            ax.set_ylabel('Loss on next portion')
        elif metric == 'eval_loss':
            ax.set_ylabel('Best ' + eval_split + ' loss')
        elif metric == 'train_loss':
            ax.set_ylabel('Best train loss')

        for model_setting in model_settings:
            model, weights, layer = model_setting.split('/')
            layer = modelinfo.layer_idx(model, layer)
            path = os.path.join(
                results_path,
                "_".join([model, weights, task, str(layer), str(datasets_set),
                          eval_split]) + ".pkl")
            with open(path, 'rb') as file:
                results = pickle.load(file)
            if metric == 'mdl_loss':
                means, stds = calculate_mean_and_std(results, 'mdl_eval_losses')
            elif metric == 'eval_loss':
                means, stds = calculate_mean_and_std(
                    results, 'epoch_eval_losses', select_min=True)
            elif metric == 'train_loss':
                means, stds = calculate_mean_and_std(
                    results, 'epoch_train_losses', select_min=True)
            ax.errorbar(np.arange(len(portions)), means, stds, label=model_setting)
        ax.legend()
        timestamp = datetime.datetime.now().strftime(r'%Y-%m-%d_%H.%M.%S')
        image_path = os.path.join(
            "plots",
            "_".join([metric, "per", "portion", task, str(datasets_set), timestamp]) + ".pdf")
        plt.savefig(image_path,  bbox_inches='tight')
        plt.close()


def plot_layer_similarity(results_path, models):
    """Plots heatmap for pairwise layer similarity computed using LinCKA.

    :param results_path: path to directory saving experiment results
    :param models: list of sets of models to save results from, for example:
        [unispeech-sat-base-plus+unispeech-sat-base-plus-sv,
        spkrec-ecapa-voxceleb+spkrec-resnet-voxceleb
        ]
    """
    for comparison in models:
        plt.figure()
        path = os.path.join(results_path, "_".join(["lincka", comparison]) + ".pkl")
        models = comparison.split('+')  # File might contain multiple models
        modelinfo = utils.ModelInfo()

        with open(path, 'rb') as file:
            heatmap = pickle.load(file)['heatmap']

        xlabels = []
        ylabels = []
        for model in models:
            name_written = False
            for i in range(1, modelinfo.n_layers(model) + 1):
                if (ceil(modelinfo.n_layers(model) / i) == 2) and (name_written == False):
                    xlabels.append(str(i) + '\n' + modelinfo.clean_name(model))
                    ylabels.append(modelinfo.clean_name(model) + ' ' + str(i))
                    name_written = True
                else:
                    xlabels.append(str(i))
                    ylabels.append(str(i))

        plt.xticks(np.arange(heatmap.shape[0]), labels=xlabels)
        plt.yticks(np.arange(heatmap.shape[1]), labels=ylabels)
        plt.imshow(heatmap, cmap='inferno', vmin=0, vmax=1)
        plt.colorbar()
        image_path = os.path.join('plots', f"lincka_{comparison}.pdf")
        plt.savefig(image_path, bbox_inches='tight')


def plot_vib_losses(results_path, models, tasks, eval_split, stage, seed,
                    ymax, datasets_set, n_mean=100):
    """Plots training graph of information loss and task loss for each model-task pair.

    :param results_path: path to directory saving experiment results
    :param models: names of models to save results from
    :param tasks: attributes to show results for
    :param eval_split: evaluation split to show results for
    :param stage: VIB stage to show results for
    :param seed: VIB seed to show results for
    :param ymax: the ymax for compression and codelength (may be `None`)
    :param datasets_set: datasets set for which to show results
    :param n_mean: of how many training steps to take the mean for the loss, defaults to 100
    """
    for model in models:
        for task in tasks:
            with open(os.path.join(
                results_path,
                "_".join(["vib", str(stage), task, str(datasets_set), model, str(seed),
                          eval_split]) + ".pkl"), 'rb') as file:
                results = pickle.load(file)
            losses = results['train_losses']
            averages = {}
            for key in losses.keys():
                # Drop last because it might not have enough points
                averages[key] = [sum(losses[key][i:i+n_mean]) /
                                 n_mean for i in range(0, len(losses[key]), n_mean)][:-1]
            fig, ax = plt.subplots()
            for loss, colour in [('Task', '#4477AA'), ('Info', '#228833'), ('Total', '#AA3377')]:
                ax.plot(range(0, len(losses[loss]), n_mean)[:-1],
                        averages[loss], color=colour, label=loss)
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_ylim(ymin=0)
            if ymax is not None:
                ax.set_ylim(ymax=ymax)
            image_path = os.path.join(
                'plots',
                "_".join(["vib", "losses", str(stage), task, str(datasets_set), model, str(seed),
                          eval_split]) + ".pdf")
            plt.legend()
            plt.savefig(image_path, bbox_inches='tight')


def plot_vib_weights(vib_path, models, tasks,  eval_split, stage, layer, seed, datasets_set, extreme=1.2):
    """Plots a heatmap of the weights of a VIB decoder.

    :param vib_path: Path to VIB directory
    :param models: names of models to save results from
    :param tasks: attributes to show results for
    :param eval_split: evaluation split to show results for
    :param stage: VIB stage to show results for
    :param layer: model layer to show results for
    :param seed: VIB seed to show results for
    :param datasets_set: datasets set for which to show results
    :param extreme: positive extreme value for heatmap, defaults to 1.2
    """
    import torch
    modelinfo = utils.ModelInfo()
    for model in models:
        for task in tasks:
            fig, ax = plt.subplots()

            vib_name = "_".join([str(stage), task, str(datasets_set),
                                 modelinfo.hf_name(model), str(modelinfo.layer_idx(model, layer))])
            with open(os.path.join(
                    vib_path, "_".join([vib_name, str(seed), eval_split]) + ".pth"), 'rb') as file:
                state_dict = torch.load(file, map_location=torch.device('cpu'))
            with open(os.path.join(vib_path, "_".join([vib_name, "cfg"]) + ".pkl"), 'rb') as file:
                vib_config = pickle.load(file)
            latent_dim = vib_config.latent_dim
            data = torch.cat((state_dict['decoder.clf.weight'],
                              state_dict['decoder.clf.bias'].unsqueeze(dim=1)), dim=1)
            if torch.min(data) < -extreme or torch.max(data) > extreme:
                print("Warning: weights exceed imshow range")
            plt.xlabel('Embedding')
            xticks = [latent_dim *
                      i for i in range(int(data.shape[1] / latent_dim))] + [data.shape[1] - 1]
            if vib_config.stage_1_tasks is not None:
                xlabels = [
                    f'1-{stage_1_task.capitalize()}'
                    for stage_1_task in vib_config.stage_1_tasks] + [f'2-{task.capitalize()}']
            else:
                xlabels = [f'1-{task.capitalize()}']
            xlabel_positions = [pos + 0.5 * latent_dim for pos in xticks[:-1]]
            plt.xticks(xticks, minor=True)
            plt.xticks(xlabel_positions, xlabels, minor=False)
            yticks = [0, 24, 49, 74, 99]
            plt.yticks(yticks, [tick + 1 for tick in yticks])
            plt.tick_params(axis='x', which="major", length=0)
            plt.ylabel('Classes')
            plt.imshow(data, vmin=-extreme, vmax=extreme, interpolation='nearest', cmap='viridis')
            plt.colorbar(shrink=0.6)
            image_path = os.path.join(
                'plots', "_".join([
                    "vib", "weights", str(stage), task, str(datasets_set), model,
                    str(modelinfo.layer_idx(model, layer)), str(seed), eval_split]) + ".pdf")
            plt.savefig(image_path, bbox_inches='tight', dpi=300)


def plot_vib_samples(vib_path, data_path, models, tasks, eval_split, stage, arg_layer, seed,
                     datasets_set, datasets_set_2, extreme=0.7, n_samples=10):
    """Plots samples of encodings of a VIB: n samples from the same (randomly selected) class, n
        samples from different (randomly selected) classes.

    :param vib_path: Path to VIB directory
    :param data_path: Path to data directory
    :param models: names of models to save results from
    :param tasks: attributes to show results for
    :param eval_split: evaluation split to show results for
    :param stage: VIB stage to show results for
    :param arg_layer: model layer to show results for
    :param seed: VIB seed to show results for
    :param datasets_set: datasets set from which to get samples
    :param datasets_set_2: datasets set for which to get the trained VIB
    :param extreme: positive extreme value for heatmap, defaults to 0.7
    :param n_samples: max number of samples to show per VIB
    """
    import torch
    from torch.utils.data import Subset
    modelinfo = utils.ModelInfo()
    plt.rcParams['figure.figsize'] = [6.4, 5.5]
    for model_name in models:
        layer = modelinfo.layer_idx(model_name, arg_layer)
        for task in tasks:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            vib_name = "_".join([str(stage), task, str(datasets_set_2),
                                 modelinfo.hf_name(model_name), str(layer)])
            with open(os.path.join(vib_path,
                                   "_".join([vib_name, str(seed), eval_split]) + ".pth"),
                      'rb') as file:
                state_dict = torch.load(file, map_location=torch.device('cpu'))
            with open(os.path.join(vib_path, "_".join([vib_name, "cfg"]) + ".pkl"), 'rb') as file:
                vib_config = pickle.load(file)
            vib = VIB(vib_config)
            vib.load_state_dict(state_dict)

            model, feature_extractor = utils.load_model(model_name, 'cpu')
            dataset = utils.get_datasets_for_task(
                data_path, model_name, task, eval_split, datasets_set=datasets_set)[1]
            # Get samples for same and different label
            target_class = np.random.randint(dataset.n_classes())
            same_class_idx = []
            different_classes = []
            different_class_idx = []
            for i in range(len(dataset.labels)):
                label = dataset.labels[i]
                if label == target_class:
                    same_class_idx.append(i)
                else:
                    if label not in different_classes:
                        different_classes.append(label)
                        different_class_idx.append(i)
            same_class_samples = Subset(dataset, indices=np.random.choice(
                same_class_idx, size=(min(n_samples, len(same_class_idx))), replace=False))
            different_class_samples = Subset(dataset, indices=np.random.choice(
                different_class_idx, size=(n_samples), replace=False))

            encodings = {}
            for examples, which in [(same_class_samples, 'same_class'),
                                    (different_class_samples, 'different_class')]:
                model_reps = utils.obtain_representations(
                    model, feature_extractor, model_name, examples, 'cpu')
                model_reps = [(reps[layer], label) for (reps, label) in model_reps]
                representations = utils.pass_through_vib(model_reps, vib)
                encodings[which] = [x[0] for x in representations]  # Remove labels
            mappable = ax1.imshow(encodings['same_class'], vmin=-extreme, vmax=extreme)
            ax2.imshow(encodings['different_class'], vmin=-extreme, vmax=extreme)
            ax1.set_title('Within class')
            ax2.set_title('Across class')
            for ax in [ax1, ax2]:
                ax.set_ylabel('Sample')
                ax.set_yticks([])
                ax.set_xlabel('Latent dimension')
            plt.colorbar(mappable, ax=[ax1, ax2])
            image_path = os.path.join(
                'plots',
                "_".join(["vib", "samples", str(stage), task, str(datasets_set),
                         model_name, str(layer), str(seed), eval_split]) + ".pdf")
            plt.savefig(image_path, bbox_inches='tight', dpi=300)


def plot_utt(utt_path, feature):
    """Plots waveform, spectrogram, F-bank or MFCCs of utterance.

    :param utt_path: path to utterance
    :param feature: what to plot (`wave`, `spectogram`, `fbank` or `mfccs`)
    """
    utt = read_wav(utt_path)

    plt.figure(figsize=(8, 4))

    if (feature == 'wave'):
        librosa.display.waveshow(y=utt, sr=16000)
    else:
        spectrogram = features.spectrogram(utt)
        fbank = features.fbank(spectrogram)
        mfccs = features.mfcc(fbank)
        if (feature == 'spectrogram'):
            librosa.display.specshow(librosa.amplitude_to_db(
                np.abs(spectrogram), ref=np.max), sr=16000)
        elif (feature == 'fbank'):
            librosa.display.specshow(librosa.power_to_db(np.abs(fbank), ref=np.max), sr=16000)
        elif (feature == 'mfccs'):
            librosa.display.specshow(mfccs, sr=16000)

    plt.axis('off')
    utt_name = utt_path.split(os.sep)[-1]
    plt.savefig(f'images/{utt_name}-{feature}.pdf',  bbox_inches='tight', pad_inches=0)
    plt.close()


def calculate_mean_and_std(all_results, key, select_min=False):
    """Returns the mean and standard deviation for all seeds for a given key in a results dict.

    :param all_results: the results dict, as pickled in `run_probing_experiments.py`.
    :param key: key to get mean and standard deviation of
    :param select_min: use to get the mean of the best loss per epoch, defaults to False
    :return: mean and standard deviation (usually ints, but may be a numpy array for certain keys)
    """
    results_per_seed = []
    for seed in all_results.keys():
        results = all_results[seed][key]
        if select_min == True:
            results = [min(result) for result in results]
        results_per_seed.append(results)
    mean = np.mean(results_per_seed, axis=0)
    std = np.std(results_per_seed, axis=0)

    return mean, std


def main(args):
    """Sets font size and calls correct visualization with required arguments.

    :param args: commandline arguments
    """
    plt.rcParams.update({'font.size': args.font_size})
    modelinfo = utils.ModelInfo()
    # To allow model arguments with or without origin
    if args.visualization != 'layer_similarity':
        models = [modelinfo.hf_name(model) for model in args.models]

    match args.visualization:
        case 'codelength' | 'compression' | 'accuracy':
            plot_layer_results(args.results_path, models, args.tasks, args.eval_split,
                               metric=args.visualization, multitask=not args.single_task,
                               regenerate_legend=args.regenerate_legend, ymax=args.ymax,
                               datasets_set=args.datasets_set)
        case 'wave' | 'spectrogram' | 'fbank' | 'mfccs':
            plot_utt(args.utt_path, args.visualization)
        case 'mdl_loss_portion' | 'eval_loss_portion' | 'train_loss_portion':
            plot_portion_results(args.results_path, args.tasks, args.model_settings,
                                 args.eval_split, metric=args.visualization[:-8],
                                 datasets_set=args.datasets_set)
        case 'layer_similarity':
            plot_layer_similarity(args.results_path, args.models)
        case 'vib_losses':
            plot_vib_losses(args.results_path, models, args.tasks, args.eval_split,
                            args.stage, args.seed, args.ymax, args.datasets_set)
        case 'vib_weights':
            plot_vib_weights(args.vib_path, models, args.tasks, args.eval_split,
                             args.stage, args.layer, args.seed, args.datasets_set)
        case 'vib_samples':
            plot_vib_samples(args.vib_path, args.data_path, models, args.tasks,
                             args.eval_split, args.stage, args.layer, args.seed, args.datasets_set,
                             args.datasets_set_2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--results_path',
                        help="Path to results directory (if plotting results)",
                        type=str,
                        default="results")
    parser.add_argument('--data_path',
                        help="Path to main data directory",
                        type=str,
                        default='data')
    parser.add_argument('--utt_path',
                        help="Path to utterance (if plotting utterance)",
                        type=str,
                        default=None)
    parser.add_argument('--vib_path',
                        help="Path to VIB directory",
                        type=str,
                        default='vib')
    parser.add_argument('--models',
                        help="Models to save results from",
                        type=str,
                        nargs='+',
                        default=["wavlm-base-plus", "wavlm-base-plus-sv",
                                 "unispeech-sat-base-plus", "unispeech-sat-base-plus-sv",
                                 "spkrec-ecapa-voxceleb", "spkrec-xvect-voxceleb", "spkrec-resnet-voxceleb",
                                 "speakerverification_en_titanet_large"])
    parser.add_argument('--tasks',
                        help="Tasks to save results from",
                        type=str,
                        nargs='+',
                        default=['speaker', 'content', 'channel'])
    parser.add_argument('--datasets_set',
                        help="What set of datasets to use",
                        type=int,
                        default=1)
    parser.add_argument('--datasets_set_2',
                        help="What set of datasets to use, for functions that take two",
                        type=int,
                        default=3)
    parser.add_argument('--eval_split',
                        help="Evaluation split to get results from",
                        type=str,
                        choices=['dev', 'test'],
                        default='dev')
    parser.add_argument('--model_settings',
                        help="For plot that shows different model settings, what settings to include",
                        type=str,
                        nargs='+',
                        default=['spkrec-ecapa-voxceleb/trained/-1', 'spkrec-ecapa-voxceleb/random/-1'])
    parser.add_argument('--stage',
                        help="For VIB visualization, the disentanglement stage to use",
                        type=int,
                        default=1)
    parser.add_argument('--layer',
                        help="For VIB visualization,"
                        "VIB trained on which layer to use",
                        type=int,
                        default=-1)
    parser.add_argument('--visualization',
                        help="What to visualize",
                        type=str,
                        choices=['codelength', 'compression', 'accuracy',
                                 'train_loss_portion', 'eval_loss_portion', 'mdl_loss_portion',
                                 'wave', 'spectrogram', 'fbank', 'mfccs',
                                 'vib_losses', 'vib_weights', 'vib_samples', 'layer_similarity'],
                        default='compression')
    parser.add_argument('--single_task',
                        help="Some visualizations include all tasks in one plot by default, include this to plot separately",
                        action="store_true")
    parser.add_argument('--regenerate_legend',
                        help="For visualization that can save legend to separate file, whether to do so",
                        action='store_true')
    parser.add_argument('--ymax',
                        help="Whether to have a fixed upper limit of y axis for some plots",
                        type=int,
                        default=None)
    parser.add_argument('--font_size',
                        help="Font size to use for axis labels, ticks etc",
                        type=int,
                        default=18)
    parser.add_argument('--seed',
                        help="When plotting results for a specific seed, what seed to use",
                        type=int,
                        default=0)

    args = parser.parse_args()

    sys.exit(main(args))
