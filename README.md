# GuideGATE

GuideGATE is a deep-learning framework for drug-target interaction (DTI)
prediction. This repository contains the model implementation, preprocessing
scripts, and the datasets used in the experiments.

## System Requirements

- Python 3.8
- PyTorch >= 1.7.1
- DGL >= 0.7.1
- DGL-LifeSci >= 0.2.8
- NumPy >= 1.20.2
- scikit-learn >= 0.24.2
- pandas >= 1.2.4
- PrettyTable >= 2.2.1
- RDKit ~= 2021.03.2
- YACS ~= 0.1.8

## Installation

Create and activate a Conda environment:

```bash
conda create --name guidegate python=3.8
conda activate guidegate
```

Install PyTorch 1.7.1 with CUDA 10.2 and the required dependencies:

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
conda install -c conda-forge rdkit==2021.03.2
conda install -c conda-forge transformers
pip install "dgl>=0.7.1" "dgllife>=0.2.8" "numpy>=1.20.2" "scikit-learn>=0.24.2" "pandas>=1.2.4" "prettytable>=2.2.1" "yacs~=0.1.8"
```

Adjust the PyTorch and CUDA versions if a different CUDA toolkit is installed
on your system.

## Pretrained BERT Models

Download both pretrained models from
[Google Drive](https://drive.google.com/file/d/1hQJYe4bAIcfjaE-lKWnDElZRAyREpceu/view?usp=drive_link):

- `prot_bert` is used to generate protein sequence embeddings.
- `ChemBERTa-77M-MLM` is used to generate SMILES embeddings.

Extract or place both downloaded model directories at the repository root.
Keep their directory names unchanged because the preprocessing scripts load
the models using these paths:

```text
guidegate/
|-- prot_bert/
|-- ChemBERTa-77M-MLM/
|-- code/
|-- dataset/
`-- preprocess/
```

## Datasets

The `dataset` folder contains all experimental data used in GuideGATE:

- [BindingDB](https://www.bindingdb.org/bind/index.jsp)
- [BioSNAP](https://github.com/kexinhuang12345/MolTrans)
- [Human](https://github.com/lifanchen-simm/transformerCPI)
- [C. elegans](https://github.com/lifanchen-simm/transformerCPI)
## Generate Embeddings

Before training, generate the protein and SMILES embeddings for the selected
dataset:

```bash
python preprocess/get_embeddings.py --dataset bindingdb --device cuda:0
```

Replace `bindingdb` with `biosnap`, `human`, or `celegans`. Use `--device cpu`
when CUDA is unavailable. The command generates `proteinsembeddings.npy` and
`smilesembeddings.npy`. Once generated, the embeddings do not need to be
created again. The repository stores data under `dataset/`; make sure the
`data_path` in `preprocess/get_embeddings.py` also points to `./dataset` before
running this command.

## Reproduce Results

Run all commands from the repository root. For standard GuideGATE experiments,
`${dataset}` can be `bindingdb`, `biosnap`, `human`, or `celegans`, and
`${split_task}` can be `cluster`, `random`, or `cold`:

```bash
python code/main.py --data ${dataset} --type ${split_task}
```

For cross-domain GuideGATE + TGSA experiments, `${dataset}` can be `bindingdb`
or `biosnap`:

```bash
python code/main.py --data ${dataset} --type cluster
```

## Split the Datasets

To create dataset splits, run the preprocessing script from the repository
root. `${dataset}` can be `bindingdb`, `biosnap`, `human`, or `celegans`, and
`${split_type}` can be `cluster`, `random`, or `cold`:

```bash
python preprocess/split_dataset.py --dataset ${dataset} --split_settings ${split_type}
```
