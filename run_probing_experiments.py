import sys
import argparse
import pickle

import numpy as np
import torch

import probe
import visualize
import vib_training
import utils


def main(args):
    """Runs over various sets of settings and calls `probe.main()` and if necessary
    `vib_training.main()` with the appropriate arguments. Outputs results to output and file.

    :param args: commandline arguments
    """
    print(args)
    modelinfo = utils.ModelInfo()
    tasks = args.tasks
    models = [modelinfo.hf_name(model) for model in args.models]
    seeds = args.seeds
    if args.vib_stage is None:
        vib_tasks = [None]
    else:
        vib_tasks = args.vib_tasks

    tiny_test_string = f'_tiny_{args.tiny_test}' if args.tiny_test is not None else ''

    for arg_model in models:
        if args.only_trained:
            random = [False]
        else:
            random = [False, True]
        for arg_random in random:
            if (arg_model == 'features' and arg_random == True):
                continue  # There are no random versions of the features
            if (arg_model == 'speakerverification_en_titanet_large' and arg_random == True):
                continue  # Randomly initialised version hasn't been implemented yet
            if (args.layers == 'all'):
                layers = [i for i in range(modelinfo.n_layers(arg_model))]
            else:
                layers = [modelinfo.layer_idx(arg_model, args.layers)]
            for arg_vib_task in vib_tasks:
                # Note that vib_tasks is [None] if we're not probing vibs
                # When overwriting representations, we only need to do it once for each model and
                # task, it is set to False after that
                vib_overwrite_representations = args.overwrite_representations
                all_results = {}
                train_vibs = {layer:
                              {seed: args.train_vibs if args.vib_stage else False for seed in seeds}
                              for layer in layers}
                for arg_task in tasks:
                    probe_overwrite_representations = args.overwrite_representations
                    for arg_layer in layers:
                        for arg_seed in seeds:
                            np.random.seed(arg_seed)
                            torch.manual_seed(arg_seed)
                            if train_vibs[arg_layer][arg_seed] == True:
                                vib_args = argparse.Namespace(data_path=args.data_path, eval_split=args.eval_split,
                                                              model=arg_model, layer=arg_layer, task=arg_vib_task, stage=args.vib_stage,
                                                              seed=arg_seed, datasets_set=args.vib_datasets_set, batch_size=args.batch_size, tiny_test=args.tiny_test,
                                                              bottleneck_dimensionality=args.bottleneck_dimensionality, info_loss_factor=args.info_loss_factor,
                                                              info_loss_multiplier=args.info_loss_multiplier, learning_rate=args.learning_rate, stage_1_tasks=args.stage_1_tasks,
                                                              overwrite_representations=vib_overwrite_representations, n_epochs=args.n_epochs,
                                                              random_control=False, evaluate_train=False)
                                vib_training.main(vib_args)
                                train_vibs[arg_layer][arg_seed] = False
                                vib_overwrite_representations = False

                            probe_args = argparse.Namespace(data_path=args.data_path, overwrite_representations=probe_overwrite_representations, eval_split=args.eval_split,
                                                            model=arg_model, layer=arg_layer, random=arg_random, task=arg_task,
                                                            probe=args.probe, hidden_size=args.hidden_size, tiny_test=args.tiny_test,
                                                            seed=arg_seed, datasets_set=args.datasets_set, batch_size=args.batch_size,
                                                            vib_stage=args.vib_stage, vib_task=arg_vib_task, vib_datasets_set=args.vib_datasets_set, no_mdl=args.no_mdl)

                            results = probe.main(probe_args)
                            all_results[arg_seed] = results
                            probe_overwrite_representations = False

                        expt_path = "_".join(["results/" + modelinfo.hf_name(arg_model),
                                              'random' if arg_random else 'trained', arg_task,
                                              str(arg_layer), str(args.datasets_set) +
                                              tiny_test_string, args.eval_split])
                        if args.vib_stage is not None:
                            expt_path += f'_{args.vib_stage}_{arg_vib_task}'
                        if args.probe == 'linear':
                            expt_path += '_linear'
                        if args.no_mdl == False:
                            mean_comp, std_comp = visualize.calculate_mean_and_std(
                                all_results, 'compression')
                        mean_acc, std_acc = visualize.calculate_mean_and_std(
                            all_results, 'eval_acc')

                        print("\r\n****************************************************")
                        if args.vib_stage is None:
                            print(
                                f"Results for layer {probe_args.layer} of {probe_args.model} for {probe_args.task} {"with randomly initialised weights" if probe_args.random else "with trained weights"}")
                        else:
                            print(
                                f"Results for stage {args.vib_stage} VIB trained on {arg_vib_task}, on top of layer {probe_args.layer} of {arg_model}, probed for {arg_task}")
                        if args.no_mdl == False:
                            print("Mean compression is", mean_comp, "+/-", std_comp)
                        print("Mean", args.eval_split, "accuracy is", mean_acc, ", +/-", std_acc)
                        print("****************************************************\r\n")

                        with open(expt_path + '.pkl', 'wb') as file:
                            pickle.dump(all_results, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--models',
                        help="Models to probe",
                        type=str,
                        nargs='+',
                        default=["wavlm-base-plus", "wavlm-base-plus-sv",
                                 "unispeech-sat-base-plus", "unispeech-sat-base-plus-sv",
                                 "spkrec-ecapa-voxceleb", "spkrec-xvect-voxceleb",
                                 "spkrec-resnet-voxceleb"])
    parser.add_argument('--tasks',
                        help="Tasks to probe models with",
                        type=str,
                        nargs='+',
                        default=['speaker', 'content', 'channel'])
    parser.add_argument('--layers',
                        help="What layers to probe ('all' or single layer index)",
                        type=str,
                        default='all')
    parser.add_argument('--only_trained',
                        help="Exclude random baseline",
                        action='store_true')
    parser.add_argument('--seeds',
                        help="Seeds to use",
                        type=int,
                        nargs='+',
                        default=[0, 42, 607])
    parser.add_argument('--data_path',
                        help="Path to main data directory",
                        type=str,
                        default='data')
    parser.add_argument('--overwrite_representations',
                        help="Put data through model even if pickle exists",
                        action="store_true")
    parser.add_argument('--eval_split',
                        help="Whether to evaluate on the dev or the test set",
                        type=str,
                        choices=['dev', 'test'], default='dev')
    parser.add_argument('--datasets_set',
                        help="What set of datasets to use",
                        type=int,
                        default=1)
    parser.add_argument('--vib_datasets_set',
                        help="What set of datasets to use when training vibs",
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
    parser.add_argument('--no_mdl',
                        help="Do regular probing, only reporting accuracy",
                        action="store_true")
    parser.add_argument('--probe',
                        help="Architecture of probe",
                        type=str,
                        choices=['MLP', 'linear'],
                        default='MLP')
    parser.add_argument('--hidden_size',
                        help="Size of hidden layer in MLP",
                        type=int,
                        default=500)
    parser.add_argument('--vib_stage',
                        help="VIB stage to use",
                        type=str,
                        choices=['1', '2'],
                        default=None)
    parser.add_argument('--vib_tasks',
                        help="VIB tasks to use",
                        type=str,
                        nargs='+',
                        default=['speaker', 'content', 'channel'])
    parser.add_argument('--train_vibs',
                        help="Include when to probe VIBs have not been trained yet",
                        action="store_true")
    parser.add_argument('--bottleneck_dimensionality',
                        help="The output size of the VIB encoders",
                        type=int,
                        default=64)
    parser.add_argument('--info_loss_factor',
                        help="A factor with which to multiply the info loss, before the following; by default increases gradually",
                        type=str,
                        default='incremental')
    parser.add_argument('--info_loss_multiplier',
                        help="A factor with which to multiply the info loss, after calculation",
                        type=float,
                        default=1.0)
    parser.add_argument('--learning_rate',
                        help="Learning rate for AdamW in VIB training",
                        type=float,
                        default=0.001)
    parser.add_argument('--n_epochs',
                        help="Amount of epochs to train VIBs for",
                        type=int,
                        default=100)
    parser.add_argument('--stage_1_tasks',
                        help="When in VIB stage 2, what VIBs from stage 1 to concatenate to representation",
                        type=str,
                        nargs='+',
                        choices=['content', 'channel', 'speaker'],
                        default=['content', 'channel'])

    args = parser.parse_args()

    sys.exit(main(args))
