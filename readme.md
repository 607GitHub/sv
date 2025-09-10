# Impact of content and channel on automatic speaker verification
## Introduction
This repository's goal is to allow reproduction and expansion of my Master's thesis project (see `docs/master-thesis.pdf`). It implements several well-known speaker verification models, and applies minimum description length probing to investigate the models' representation of speaker, content and channel. It also includes a partial fork of <https://github.com/hmohebbi/disentangling_representations>, which is used to investigate the possibility of further disentanglement of these attributes.

## Repository overview
- docs _Contains documents and slides about my project._
- images _Contains images used in my thesis or related media, both source files and generated pdfs._
- plots _Directory which plots are saved to; contains the plots used in my thesis._
- representations _This directory will be created upon running code from the project, and is used to store model representations to save on computational costs._
- results _Contains the pickled results from experiments._
- speechbrain_files _Contains files used for the randomly initialised versions of the Speechbrain models._
- splits_copy _Contains the splits files used for each dataset._
- vib _Default directory for VIB checkpoints._
- augment.py _Used in data augmentation for LJ Speech and SCC._
- efficient_CKA.py* _Calculates Linear CKA for pairwise layer similarity analysis._
- **evaluate.py** _Evaluates speaker verification model or VIB on VoxCeleb._
- features.py _Calculates acoustic features as a baseline._
- measures.py _Implements EER._
- mydatasets.py _Implements the used datasets._
- **probe.py** _Does a single probing experiment._
- probes.py _Implements an MLP and linear probe._
- readme.md _This file._
- requirements.txt _pip requirements to install environment._
- resample.py _Resamples required audio files in a dataset._
- **run_probing_experiments.py** _Runs set of probing experiments, including VIB training if applicable._
- **setup_datasets.py** _Sets up datasets to be used with the other modules, having downloaded the original archives._
- **similarity_analysis.py** _Performs pairwise layer similarity analysis using Linear CKA._
- synthesize.py _Used in data synthesis for SCC._
- utils.py _Home of ModelInfo class and several functions that are used in different modules._
- vib.py* _Implements VIB._
- vib_training.py* _Trains single VIB._
- **visualize.py** _Contains various plotting functions, including but not limited to those used in the thesis._

Modules in bold can be called with commandline arguments; other modules are only used from within other modules.  
*Originally forked from another project.

## Setting up environment
To be able to run this project's code, install the packages in `requirements.txt`. This was tested with Python 3.12.8, on Linux. Furthermore, to work with the models from Speechbrain, `https://github.com/607GitHub/speechbrain` should be cloned and installed to your environment. To work with TitaNet, `https://github.com/607GitHub/NeMo` should be cloned and installed to your environment.

## Setting up datasets
All used corpora are, at the moment of writing, freely available. This section gives directions on downloading the datasets and setting them up to work with the provided scripts. Note that the directory names of the datasets are partially hardcoded. To get the correct directory names and the splits used in the thesis, copy or rename `splits_files` and put the downloaded archives in the appropriate folders in there.

### Downloading required files
#### British Isles
The British Isles dataset is available at <https://openslr.org/83/>. It consists of 11 zipped archives and `line_index_all.csv`, which 12 files should be downloaded and put in a directory `british-isles` within your main data folder.
#### VOiCES
The VOiCES dataset can be obtained through the AWS Command Line Interface. Note that this does *not* require an AWS account, if you use the following command:
```
aws s3 cp --no-sign-request s3://lab41openaudiocorpus/VOiCES_devkit.tar.gz .
```
Run the command from a directory named `VOiCES` within your data folder.
#### LJ Speech
The LJ Speech dataset is available at <https://keithito.com/LJ-Speech-Dataset/>. Download it to a directory named `LJSpeech` within your data folder. Augmenting the LJ Speech dataset further requires MUSAN, available from <https://www.openslr.org/17/> and RIR, available from <https://www.openslr.org/28/>. Put both these archives (`rirs_noises.zip`, `musan.tar.gz`) in the root of your data folder. After generation, to save space, the directory `unaugmented`, which is left in case one wants to manually rerun augmentation with different settings, can be deleted.
#### VCTK
The VCTK dataset is available at <https://datashare.ed.ac.uk/handle/10283/3443>. Download it to a directory called `VCTK` within your data folder.
#### SCC
For SCC, the same datasets are required as for LJ Speech. In addition, for SCC v1, VCTK is required, and for SCC v2, VOiCES is required. Before generating the dataset, create a directory named `SCC`. Note that as the project is currently set up, both versions of SCC can not be used concurrently. Also note that on Windows, if the path to your data directory is already long, you might run into the maximum path length for SCC, as the generated files tend to have long filenames.
#### VoxCeleb
VoxCeleb can be requested via <https://cn01.mmai.io/keyreq/voxceleb>. When you have received the keys, download the Test set of VoxCeleb1 and extract it into your data directory. In addition, download <https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt> to the newly extracted `VoxCeleb1-test` folder. This dataset requires no further set-up.

### Setting up datasets
 To set up a dataset with existing splits, put the files from the appropriate directory within `splits_copy` in the directory for your dataset, and run:
```
python setup_datasets.py --data_path [your data path] --datasets [datasets that you want to set up]
```
To set up the dataset with newly generated splits, include `--generate_splits`.
After succesfully setting up the dataset, the archives can be removed to save space. If something went wrong, delete the `temp` and/or `wav` directories to try again.

Since getting the results in the thesis, improvements have been made to the channel augmentation used in LJ Speech and SCC. By default, the improved versions are generated. To get the old versions, for SCC, include `--clip_signal False --amplitude_factor 0.3 --preserve_utt_vol False`. For LJ Speech, include `--clip_signal False --amplitude_factor 1 --preserve_utt_vol False`. The provided SCC splits contained a bug in creation, leading to some examples appearing in multiple datasets, contrary to what is claimed in the thesis. This bug has been fixed in the code, but you can use the old splits files to generate the dataset with this issue. In general, to generate SCC 1, which, as described in the Limitations section of the thesis, uses the same speakers, sentences and channel settings in both the probing and disentanglement splits (although unique triples), pass `--SCC_version 1`. By default, SCC 2 is generated, which has 200 classes for speaker, content and channel, 100 for disentanglement and 100 for probing.
After generation, to save space, the archives can be deleted. However, if you might want to regenerate the datasets with different settings at some point, keep them. Note that generating SCC requires access to archives for different datasets.

## Reproducing thesis results
This section gives the commands required to reproduce the results from my thesis. To test if the environment, datasets etc are working before running the real experiments, it is possible to pass `--tiny_test [some small integer]` to the experiment scripts to run on a small section of the dataset. The commands below all perform evaluation on the dev split. To perform evaluation on the test split, pass `--eval_split test`.  
Note that the results files from the original experiments are included in the repository. If you want to make sure that you are visualizing your reproduced results, rename `results` and create an empty directory named `results`.

### Preparation
Set up the environment and download the datasets, as instructed above. Then run:
```
python setup_datasets.py --data_path [your data path] --datasets british_isles VOiCES VCTK
python setup_datasets.py --data_path [your data path] --datasets LJSpeech --clip_signal False --amplitude_factor 1
python setup_datasets.py --data_path [your data path] --datasets SCC --clip_signal False
```

### Table 5.1 and Figure 5.1
To get the main probing results, run:
```
python run_probing_experiments.py --data_path [your data path] --models features wavlm-base-plus wavlm-base-plus-sv unispeech-sat-base-plus unispeech-sat-base-plus-sv spkrec-ecapa-voxceleb spkrec-xvect-voxceleb spkrec-resnet-voxceleb speakerverification_en_titanet_large
```
Results are stored per model, trained/random baseline setting, task and layer, as `results/[model name]_[trained/random]_[task]_[layer number]_1_[eval split].pkl`. Results for all seeds are included in the same pickle. For the plots in Figure 5.1, run:
```
python visualize.py --ymax 16 --regenerate_legend
```

### Figure 5.2
To get the ECAPA-TDNN generalisation results, run: (the first line can be left out if you've already run the above)
```
python run_probing_experiments.py --data_path [your data path] --models features spkrec-ecapa-voxceleb
python run_probing_experiments.py --data_path [your data path] --models features spkrec-ecapa-voxceleb --datasets_set 2
python run_probing_experiments.py --data_path [your data path] --models features spkrec-ecapa-voxceleb --datasets_set 4
```
And to visualize: (note that the top line overwrites the earlier ECAPA-TDNN plot; to keep both different y axes, rename it before running)
```
python visualize.py --models spkrec-ecapa-voxceleb --ymax 13 --regenerate_legend
python visualize.py --models spkrec-ecapa-voxceleb --datasets_set 2 --ymax 13
python visualize.py --models spkrec-ecapa-voxceleb --datasets_set 4 --ymax 13
```

### Table 5.2
To get the probing results on trained VIBs, run:
```
python run_probing_experiments.py --data_path [your data path] --models wavlm-base-plus --only_trained --layers -1 --vib_stage 1 --datasets_set 4 --train_vibs
python run_probing_experiments.py --data_path [your data path] --models spkrec-ecapa-voxceleb --only_trained --layers -1 --vib_stage 1 --datasets_set 4 --train_vibs --bottleneck_dimensionality 32 --info_loss_multiplier 0.5 --n_epochs 200
python run_probing_experiments.py --data_path [your data path] --models spkrec-xvect-voxceleb --only_trained --layers -1 --vib_stage 1 --datasets_set 4 --train_vibs --bottleneck_dimensionality 32
python run_probing_experiments.py --data_path [your data path] --models wavlm-base-plus spkrec-ecapa-voxceleb spkrec-xvect-voxceleb --only_trained --layers -1 --vib_stage 1 --datasets_set 1
python run_probing_experiments.py --data_path [your data path] --models wavlm-base-plus --only_trained --layers -1 --vib_stage 2 --vib_tasks speaker --datasets_set 4 --train_vibs 
python run_probing_experiments.py --data_path [your data path] --models spkrec-ecapa-voxceleb --only_trained --layers -1 --vib_stage 2 --vib_tasks speaker --datasets_set 4 --train_vibs --bottleneck_dimensionality 32 --info_loss_multiplier 0.5 --n_epochs 200
python run_probing_experiments.py --data_path [your data path] --models spkrec-xvect-voxceleb --only_trained --layers -1 --vib_stage 2 --vib_tasks speaker --datasets_set 4 --train_vibs --bottleneck_dimensionality 32
python run_probing_experiments.py --data_path [your data path] --models wavlm-base-plus spkrec-ecapa-voxceleb spkrec-xvect-voxceleb --only_trained --layers -1 --vib_stage 2 --vib_tasks speaker --datasets_set 1
```
Results are stored per model, task, layer, vib stage and vib task, as `results/[model name]_[trained/random]_[task]_[layer number]_[datasets set]_[eval split]_[VIB stage]_[VIB task].pkl`. Results for all seeds are included in the same pickle.  
Table 5.2 also includes the regular models to compare against; for British Isles and VOiCES, these can be reused from Table 5.1, for SCC, the following needs to be run:
```
python run_probing_experiments.py --data_path [your data path] --models wavlm-base-plus spkrec-ecapa-voxceleb spkrec-xvect-voxceleb --datasets_set 4 --layers -1 --only_trained
```

### Figure 5.3
To get the decoder weight visualisations, run:
```
python visualize.py --models wavlm-base-plus spkrec-ecapa-voxceleb spkrec-xvect-voxceleb --visualization vib_weights --stage 2 --tasks speaker --datasets_set 3 --font_size 16
```
Note that the labels on the x axis look slightly different, because of an update that was made too late to be incorporated into the thesis.
To get the control WavLM experiments, the following commands can be used. NB: the control experiments **overwrite** the main WavLM VIBs. Make a backup of each version before you run the next experiment! For Figure 5.3a:
```
python vib_training.py --data_path [your data path] --model wavlm-base-plus --stage 2 --task speaker --stage_1_tasks speaker content channel
python visualize.py --models wavlm-base-plus --visualization vib_weights --stage 2 --tasks speaker --datasets_set 3 --font_size 16
```
For Figure 5.3b:
```
python vib_training.py --data_path [your data path] --model wavlm-base-plus --stage 2 --task speaker --random_control
python visualize.py --models wavlm-base-plus --visualization vib_weights --stage 2 --tasks speaker --datasets_set 3 --font_size 16
```

### Figure 5.4
To get the encoder samples, run:
```
python visualize.py --models spkrec-ecapa-voxceleb spkrec-xvect-voxceleb --visualization vib_samples --datasets_set 4
python visualize.py --models spkrec-ecapa-voxceleb spkrec-xvect-voxceleb --visualization vib_samples --datasets_set 1
```

### Table 5.3
To get the speaker verification models and trained VIBs evaluated on VoxCeleb, run:
```
python evaluate.py --data_path [your data path] --model wavlm-base-plus-sv
python evaluate.py --data_path [your data path] --model unispeech-sat-base-plus-sv
python evaluate.py --data_path [your data path] --model spkrec-ecapa-voxceleb
python evaluate.py --data_path [your data path] --model spkrec-xvect-voxceleb
python evaluate.py --data_path [your data path] --model spkrec-resnet-voxceleb
python evaluate.py --data_path [your data path] --model speakerverification_en_titanet_large
python evaluate.py --data_path [your data path] --model spkrec-ecapa-voxceleb --vib_stage 1 --vib_task speaker
python evaluate.py --data_path [your data path] --model spkrec-ecapa-voxceleb --vib_stage 1 --vib_task content
python evaluate.py --data_path [your data path] --model spkrec-ecapa-voxceleb --vib_stage 1 --vib_task channel
python evaluate.py --data_path [your data path] --model spkrec-ecapa-voxceleb --vib_stage 2 --vib_task speaker
python evaluate.py --data_path [your data path] --model spkrec-xvect-voxceleb --vib_stage 1 --vib_task speaker
python evaluate.py --data_path [your data path] --model spkrec-xvect-voxceleb --vib_stage 1 --vib_task content
python evaluate.py --data_path [your data path] --model spkrec-xvect-voxceleb --vib_stage 1 --vib_task channel
python evaluate.py --data_path [your data path] --model spkrec-xvect-voxceleb --vib_stage 2 --vib_task speaker
```

### Figure A.1
To compute the layer similarity results, run:
```
python similarity_analysis.py --data_path [your data path] --models unispeech-sat-base-plus unispeech-sat-base-plus-sv
python similarity_analysis.py --data_path [your data path] --models spkrec-ecapa-voxceleb spkrec-resnet-voxceleb
python similarity_analysis.py --data_path [your data path] --models speakerverification_en_titanet_large spkrec-ecapa-voxceleb features
```
Depending on your hardware, different experiments might require different batch sizes to fit in memory. This should have a minimal effect on the results. I do not report the used batch sizes here, but a minimum of 4 was used for each experiment. For the visualization, use:
```
python visualize.py --models unispeech-sat-base-plus+unispeech-sat-base-plus-sv spkrec-ecapa-voxceleb+spkrec-resnet-voxceleb speakerverification_en_titanet_large+spkrec-ecapa-voxceleb+features --visualization layer_similarity --font_size 8
python visualize.py --models spkrec-ecapa-voxceleb+spkrec-resnet-voxceleb speakerverification_en_titanet_large+spkrec-ecapa-voxceleb+features --visualization layer_similarity --font_size 14
```
Note that for the layer similarity calculation in the thesis, the feature baseline is lacking the x-vector F-bank.

### Table A.1
The results shown in Table A.1 are obtained using the same commands as used for Table 5.1. The accuracy metric is stored in the same dictionary as the compression metric. Note that in the thesis, the contents of Tables A.1 and A.2 were inadvertently reversed: the numbers in Table A.1 correspond to the linear probe, and the numbers in Table A.2 correspond to the MLP probe.

### Table A.2
To get the linear probing results, run:
```
python run_probing_experiments.py --data_path [your data path] --models wavlm-base-plus wavlm-base-plus-sv unispeech-sat-base-plus unispeech-sat-base-plus-sv spkrec-ecapa-voxceleb spkrec-xvect-voxceleb spkrec-resnet-voxceleb speakerverification_en_titanet_large --layers -1 --probe linear --no_mdl
```
The results are saved under a filename with the same format as for the main probing experiments, with a `_linear` suffix.

## Known issues
- The contents of Tables A.1 and A.2 are reversed. i.e., you should read the caption to Table A.1 with the numbers of Table A.2, and the caption to Table A.2 with the numbers of Table A.1.
- The caption to Table A.1 says the MLP probe was trained on half of the train set. This is incorrect, it was trained on the full train set.
- TitaNet does not work in the currently provided environment.
- There was a bug in the selection of splits for SCC, as a result of which Table 4.1 is incorrect. If you want to reproduce my version of SCC, use the provided splits, if you want your splits to adhere to Table 4.1, generate new splits.
- In the thesis, 'spectogram' is a common misspelling of 'spectrogram'.
- Results shown on the May 2025 poster should be not be used. Relative results are consistent with final results, but due to two bugs in the codebase at the time, reported compression for all attributes and models is way too high.
- Conventionally, x-vector is considered to have 7 layers, like ECAPA-TDNN. Following Speechbrain's implementation, in this project, x-vector is considered to have 17 layers, with each convolution, activation and pooling operation being considered as separate layers. If you want to use the conventional 7 layers, only include layers (1-indexed) 3, 6, 9, 12, 15, 16 and 17 in your results.