# Forked from https://github.com/hmohebbi/disentangling_representations, with many changes.

import sys
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from vib import VIB, VIBConfig
import utils


def main(args):
    print(args)

    STAGE = args.stage
    LATENT_DIM = args.bottleneck_dimensionality
    LEARNING_RATE = args.learning_rate
    BETA_S1 = args.info_loss_factor
    BETA_S2 = args.info_loss_factor
    LAYER = args.layer
    INFO_MULTIPLIER = args.info_loss_multiplier

    sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(sys.modules[__name__].__file__), "..")))

    BATCH_SIZE = args.batch_size
    BETA = BETA_S1 if STAGE == "1" else BETA_S2
    EPOCHS = args.n_epochs
    EVAL_FREQ = 10
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.005
    MAX_GRAD_NORM = 1
    SELECTED_GPU = 0

    DATA_PATH = args.data_path

    TASK = args.task
    STAGE_1_TASKS = args.stage_1_tasks
    MODEL = args.model
    EVAL_SPLIT = args.eval_split

    print("Training stage", STAGE, "VIB on", MODEL, "for", TASK, "on", EVAL_SPLIT, flush=True)
    if STAGE == '2':
        print("Stage 1 vibs:", STAGE_1_TASKS)

    REPORTS_PATH = f"results/"
    MODEL_PATH = f"vib/"

    if not os.path.exists(REPORTS_PATH):
        os.makedirs(REPORTS_PATH)
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    # If run from run_probing_experiments.py, the seeds will
    # already have been set
    if __name__ == '__main__':
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{SELECTED_GPU}")
        if __name__ == '__main__':
            print('We will use the GPU:', torch.cuda.get_device_name(SELECTED_GPU))
    else:
        device = torch.device("cpu")
        if __name__ == '__main__':
            print('No GPU available, using the CPU instead.')

    # Load pre-trained model
    base_model, feature_extractor = utils.load_model(MODEL, device)

    modelinfo = utils.ModelInfo()
    layer = modelinfo.layer_idx(MODEL, LAYER)

    train_dataset, eval_dataset, n_classes = utils.get_datasets_for_task(
        DATA_PATH, MODEL, TASK, EVAL_SPLIT,
        tiny_test=args.tiny_test, datasets_set=args.datasets_set)

    reps_path = "_".join(["representations/" + modelinfo.hf_name(MODEL), 'trained', TASK,
                          str(args.datasets_set) +
                          (f"_tiny_{args.tiny_test}" if args.tiny_test is not None else "")])

    train_data = utils.get_probe_data(base_model, feature_extractor, train_dataset,
                                      f"{reps_path}_train.pkl", layer, device,
                                      MODEL, args.overwrite_representations)
    eval_data = utils.get_probe_data(base_model, feature_extractor, eval_dataset,
                                     f"{reps_path}_{EVAL_SPLIT}.pkl", layer, device,
                                     MODEL, args.overwrite_representations)

    embedding_size = int(train_data[0][0].shape[0])

    # Load trained stage 1 VIB models
    if STAGE == "2":
        stage_1_vibs = []
        for task in STAGE_1_TASKS:
            config = VIBConfig(
                input_dim=embedding_size,
                latent_dim=LATENT_DIM,
                stage="1",
                num_classes=utils.get_datasets_for_task(
                    DATA_PATH, MODEL, task, EVAL_SPLIT, datasets_set=args.datasets_set)[2],
                layer_weight_averaging=False,
                num_layers=None
            )
            vib = VIB(config)
            state_dict = torch.load(MODEL_PATH + "_".join(
                ["1", task, str(args.datasets_set),
                 modelinfo.hf_name(MODEL), str(modelinfo.layer_idx(MODEL, LAYER)), str(args.seed) +
                 (f'_tiny_{args.tiny_test}' if args.tiny_test is not None else ''),
                 EVAL_SPLIT]) + ".pth",
                map_location=torch.device(device))
            vib.load_state_dict(state_dict)
            vib.to(device)
            vib.eval()
            stage_1_vibs.append(vib)

    # Load trainable clfs
    layer_weight_averaging = False
    vib_config = VIBConfig(
        input_dim=embedding_size,
        latent_dim=LATENT_DIM,
        stage=STAGE,
        num_classes=n_classes,
        layer_weight_averaging=layer_weight_averaging,
        num_layers=None,
        stage_1_tasks=STAGE_1_TASKS if STAGE == '2' else None
    )
    vib = VIB(vib_config)
    vib.to(device)
    vib.train()

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                                  pin_memory=True,
                                  num_workers=4 if torch.cuda.is_available() else 0)
    eval_dataloader = DataLoader(eval_data, batch_size=BATCH_SIZE, shuffle=False,
                                 pin_memory=True,
                                 num_workers=4 if torch.cuda.is_available() else 0)

    training_steps = len(train_dataloader)
    total_training_steps = EPOCHS * training_steps

    optimizer = AdamW(params=vib.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=np.ceil(
                                                       WARMUP_RATIO * total_training_steps),
                                                   num_training_steps=total_training_steps)

    beta_reach_steps = (EPOCHS - 5) * training_steps
    beta = 0.1 if BETA == "incremental" else float(BETA)
    BETA_INCREMENT = (1.0 - beta) / beta_reach_steps if BETA == "incremental" else 0

    train_losses = {'Task': [], 'Info': [], 'Total': []}
    test_performances = []
    for epoch in range(EPOCHS):
        vib.train()
        for embs, labels in train_dataloader:
            embs = embs.to(device)

            # Forward VIB model
            if STAGE == "1":
                logits, mu, var = vib(embs, m=None)
            else:  # Stage 2
                stage_1_outs = []
                for stage_1_vib in stage_1_vibs:
                    if args.random_control:
                        cond = torch.randn(len(embs), LATENT_DIM).to(device)
                    else:
                        with torch.no_grad():
                            _, cond, _ = stage_1_vib(embs, m=None)
                    stage_1_outs.append(cond)
                logits, mu, var = vib(embs, m=None, cond=stage_1_outs)

            info_loss = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var, dim=-1)
            if BATCH_SIZE > 1:
                info_loss = info_loss.mean()

            task_loss = torch.nn.functional.cross_entropy(logits.cpu(), labels)

            total_loss = task_loss + INFO_MULTIPLIER * beta * info_loss

            # Perform optimization
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vib.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Store records
            train_losses['Task'].append(task_loss.item())
            train_losses['Info'].append(info_loss.item())
            train_losses['Total'].append(total_loss.item())
            if BETA == "incremental":
                beta = min(beta + BETA_INCREMENT, 1.0)

        # Evaluating on test set
        if (epoch + 1) % EVAL_FREQ == 0:
            vib.eval()
            correct = []
            for embs, labels in eval_dataloader:
                embs = embs.to(device)
                # Forward VIB model
                if STAGE == "1":
                    with torch.no_grad():
                        logits, mu, var = vib(embs, m=None)
                else:  # Stage 2
                    stage_1_outs = []
                    for stage_1_vib in stage_1_vibs:
                        with torch.no_grad():
                            _, cond, _ = stage_1_vib(embs, m=None)
                        stage_1_outs.append(cond)
                    logits, mu, var = vib(embs, m=None, cond=stage_1_outs)

                # Performance
                preds = torch.argmax(logits.cpu(), dim=-1)
                correct.extend(labels == preds)
            accuracy = sum(correct)/len(correct)
            if __name__ == '__main__':
                print(f"{EVAL_SPLIT} accuracy:", accuracy)
            test_performances.append(accuracy)

            if args.evaluate_train:
                correct = []
                for embs, labels in train_dataloader:
                    embs = embs.to(device)
                    # Forward VIB model
                    if STAGE == "1":
                        with torch.no_grad():
                            logits, mu, var = vib(embs, m=None)
                    else:  # Stage 2
                        stage_1_outs = []
                        for stage_1_vib in stage_1_vibs:
                            with torch.no_grad():
                                _, cond, _ = stage_1_vib(embs, m=None)
                            stage_1_outs.append(cond)
                        logits, mu, var = vib(embs, m=None, cond=stage_1_outs)

                    preds = torch.argmax(logits.cpu(), dim=-1)
                    correct.extend(labels == preds)
                accuracy = sum(correct)/len(correct)
                print(f"Train accuracy:", accuracy)

    # Saving reports
    postfix = f"_bs={BATCH_SIZE}_lr={LEARNING_RATE}_dim={LATENT_DIM}"
    if __name__ == '__main__':
        print(postfix)
    results = {'train_losses': train_losses, 'eval_accs': test_performances}
    with open("_".join([REPORTS_PATH + "vib", STAGE, TASK, str(args.datasets_set),
                        modelinfo.hf_name(MODEL), str(layer), str(args.seed) +
                        (f'_tiny_{args.tiny_test}' if args.tiny_test is not None else ''),
                        EVAL_SPLIT]) + ".pkl", 'wb') as f:
        pickle.dump(results, f)

    # save model
    torch.save(vib.state_dict(), MODEL_PATH + "_".join(
        [STAGE, TASK, str(args.datasets_set), modelinfo.hf_name(MODEL),
         str(layer), str(args.seed) +
         (f'_tiny_{args.tiny_test}' if args.tiny_test is not None else ''),
         EVAL_SPLIT]) + ".pth")

    # Save VIBConfig
    with open(MODEL_PATH + "_".join(
            [STAGE, TASK, str(args.datasets_set), modelinfo.hf_name(MODEL),
             str(layer), 'cfg']) + ".pkl", 'wb') as file:
        pickle.dump(vib_config, file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',
                        help="Path to main data directory",
                        type=str,
                        default='data')
    parser.add_argument('--model',
                        help="Model to train VIB on: huggingface path",
                        type=str,
                        default='microsoft/wavlm-base-plus')
    parser.add_argument('--layer',
                        help="What layer of the model to train a VIB on top of",
                        type=int,
                        default=-1)
    parser.add_argument('--stage',
                        help="What stage of disentanglement framework is applied",
                        type=str,
                        choices=['1', '2'],
                        default='1')
    parser.add_argument('--task',
                        help="Task to train on",
                        type=str,
                        choices=['content', 'channel', 'speaker'],
                        default='content')
    parser.add_argument('--stage_1_tasks',
                        help="When in stage 2, what VIBs from stage 1 to concatenate to representation",
                        type=str,
                        nargs='+',
                        choices=['content', 'channel', 'speaker'],
                        default=['content', 'channel'])
    parser.add_argument('--eval_split',
                        help="Whether to evaluate on the dev or the test set",
                        type=str,
                        choices=['dev', 'test'],
                        default='dev')
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
    parser.add_argument('--n_epochs',
                        help="Amount of epochs to train for",
                        type=int,
                        default=100)
    parser.add_argument('--datasets_set',
                        help="What set of datasets to use",
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
    parser.add_argument('--learning_rate',
                        help="Learning rate for AdamW",
                        type=float,
                        default=0.001)
    parser.add_argument('--seed',
                        help="Set seed to use for pytorch and numpy",
                        type=int,
                        default=0)
    parser.add_argument('--overwrite_representations',
                        help="Put data through model even if pickle exists",
                        action="store_true")
    parser.add_argument('--random_control',
                        help="Replaces concatenated embeddings with random values",
                        action="store_true")
    parser.add_argument('--evaluate_train',
                        help="Include to also evaluate performance on the train set",
                        action="store_true")

    args = parser.parse_args()

    main(args)
