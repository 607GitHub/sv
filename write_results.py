import utils
import pickle
from visualize import calculate_mean_and_std
import numpy as np


def write_mean_and_std_or_not(results_path, metric):
    try:
        with open(results_path, 'rb') as file:
            results = pickle.load(file)
    except:
        output.write(' &')
        return
    mean, std = calculate_mean_and_std(results, metric)
    output.write(f' & {"{:.2f}".format(round(mean, 2))} ({"{:.3f}".format(round(std, 3))})')


models = [('wavlm-base-plus', 'WavLM (general)'), ('wavlm-base-plus-sv', 'WavLM (SV)'),
          ('unispeech-sat-base-plus',
           'UniSpeech-SAT (general)'), ('unispeech-sat-base-plus-sv', 'UniSpeech-SAT (SV)'),
          ('spkrec-ecapa-voxceleb', 'ECAPA-TDNN'), ('spkrec-xvect-voxceleb',
                                                    'x-vector'), ('spkrec-resnet-voxceleb', 'ResNet'),
          ('speakerverification_en_titanet_large', 'TitaNet')]
eval_split = 'test'
datasets_set = 1
metric = 'eval_acc'

modelinfo = utils.ModelInfo()

output = open('results.txt', 'w')

if False:
    model = 'wavlm-base-plus'
    layer = modelinfo.layer_idx(model, '-1')
    output.write("\\toprule\r\nEncoder & Speaker & Content & Channel\\\\\r\n\\midrule\r\n")
    output.write(modelinfo.clean_name(model))
    for probe_task in ['speaker', 'content', 'channel']:
        path = f"results/{model}_{'trained'}_{probe_task}_{layer}_{datasets_set}_{eval_split}.pkl"
        write_mean_and_std_or_not(path, metric)
    output.write('\\\\\r\n')
    for vib_task in ['speaker', 'content', 'channel']:
        output.write("Stage 1 " + vib_task + ' VIB')
        for probe_task in ['speaker', 'content', 'channel']:
            path = f"results/{model}_{'trained'}_{probe_task}_{layer}_{datasets_set}_{eval_split}_1_{vib_task}.pkl"
            write_mean_and_std_or_not(path, metric)
        output.write('\\\\\r\n')
    output.write("Stage 2 speaker VIB")
    for probe_task in ['speaker', 'content', 'channel']:
        path = f"results/{model}_{'trained'}_{probe_task}_{layer}_{datasets_set}_{eval_split}_2_speaker.pkl"
        write_mean_and_std_or_not(path, metric)
    output.write('\\\\\r\n\\bottomrule\r\n\r\n\r\n')

if True:
    for task in ['speaker', 'content', 'channel']:
        output.write('table for ' + task + '\r\n\r\n\\toprule\r\n')
        output.write("Model & Compression & Random baseline & Feature baseline\\\\\r\n\\midrule\r\n")
        for model, name in models:
            output.write(name)
            layer = modelinfo.layer_idx(model, '-1')
            for weights in ['trained', 'random']:
                path = f"results/{model}_{weights}_{task}_{layer}_{datasets_set}_{eval_split}_linear.pkl"
                write_mean_and_std_or_not(path, metric)
            baseline_path = f"results/features_trained_{task}_{modelinfo.feature_baseline_layer(model)}_{datasets_set}_{eval_split}_linear.pkl"
            write_mean_and_std_or_not(baseline_path, metric)
            output.write('\\\\\r\n')
        output.write('\\bottomrule\r\n\r\n\r\n')
