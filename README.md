# CRFIE

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![ArXiv](https://img.shields.io/badge/Paper-ArXiv-red.svg)](https://arxiv.org/abs/2212.08929)
[![ACL](https://img.shields.io/badge/Paper-ACL-red.svg)](https://aclanthology.org/2023.acl-long.766)


This repo contains the code used for the ACL 2023 paper [Modeling Instance Interactions for Joint Information Extraction with Neural High-Order Conditional Random Field](https://aclanthology.org/2023.acl-long.766). 

## Requirements

- Python 3.7
- Python packages
  - PyTorch 1.0+ (Install the CPU version if you use this tool on a machine without GPUs)
  - transformers 3.0.2 (Using transformers 3.1+ may cause some model loading issue)
  - tqdm
  - lxml
  - nltk

## Getting Started

### Pre-processing
Our preprocessing mainly adapts from [OneIE](https://blender.cs.illinois.edu/software/oneie/The_OneIE_Package.pdf).

#### Preprocess DyGIE++
The `prepreocessing/process_dygiepp.py` script converts [datasets in DyGIE+ format](https://github.com/dwadden/dygiepp/tree/master/scripts/data/ace-event) to the input format. <br/>Example:

```
python preprocessing/process_dygiepp.py -i train.json -o train.oneie.json
```

Arguments:
- -i, --input: Path to the input file.
- -o, --output: Path to the output file.

#### Preprocess ACE2005
The `prepreocessing/process_ace.py` script converts raw ACE2005 datasets to the input format. Example:

```
python preprocessing/process_ace.py -i <INPUT_DIR>/LDC2006T06/data -o <OUTPUT_DIR> \
      -s resource/splits/ACE05-E -b bert-large-cased -c <BERT_CACHE_DIR> -l english
```

Arguments:
- -i, --input: Path to the input directory (`data` folder in your LDC2006T06 package).
- -o, --output: Path to the output directory.
- -b, --bert: Bert model name.
- -c, --bert_cache_dir: Path to the BERT cache directory.
- -s, --split: Path to the split directory. We provide document id lists for all datasets used in our paper in `resource/splits`.
- -l, --lang: Language (options: english, chinese).

#### Preprocess ERE
The `prepreocessing/process_ere.py` script converts raw ERE datasets (LDC2015E29, LDC2015E68, LDC2015E78, LDC2015E107) to the input format. 

```
python preprocessing/process_ere.py -i <INPUT_DIR>/data -o <OUTPUT_DIR> -b bert-large-cased -c <BERT_CACHE_DIR> -l english -d normal
```

Arguments:
- -i, --input: Path to the input directory (`data` folder in your ERE package).
- -o, --output: Path to the output directory.
- -b, --bert: Bert model name.
- -c, --bert_cache_dir: Path to the BERT cache directory.
- -d, --dataset: Dataset type: normal, r2v2, parallel, or spanish.
- -l, --lang: Language (options: english, spanish).

This script currently supports:
- LDC2015E29_DEFT_Rich_ERE_English_Training_Annotation_V1
- LDC2015E29_DEFT_Rich_ERE_English_Training_Annotation_V2
- LDC2015E68_DEFT_Rich_ERE_English_Training_Annotation_R2_V2
- LDC2015E78_DEFT_Rich_ERE_Chinese_and_English_Parallel_Annotation_V2
- LDC2015E107_DEFT_Rich_ERE_Spanish_Annotation_V2


### Training
- `cd` to the root directory of this package
- Set the environment variable PYTHONPATH to the current directory.
  For example, if you unpack this package to `~/High-order-IE`, run: `export PYTHONPATH=~/High-order-IE`
  
Because our framework is a pipeline schema, you should first train the Node Identification model and save the checkpoint in a directory.
- Run the command line to train an identification model: `python train_ident.py -c <CONFIG_FILE_PATH>`.

Then train the high-order classification model.
- `python train.py -c <CONFIG_FILE_PATH>`.
- One example configuration file is in `config/baseline.json`. Fill in the following paths in the configuration file:
  - BERT_CACHE_DIR: Pre-trained BERT models, configs, and tokenizers will be downloaded to this directory.
  - TRAIN_FILE_PATH, DEV_FILE_PATH, TEST_FILE_PATH: Path to the training/dev/test/files.
  - OUTPUT_DIR: The model will be saved to subfolders in this directory.
  - VALID_PATTERN_DIR: Valid patterns created based on the annotation guidelines or training set. Example files are provided in `resource/valid_patterns`.
  - Set NER_SCORE and SPLIT_TRAIN to be true: Our base pipeline model with different scoring functions of OneIE.
  - IDENT_MODEL_PATH: Path to a checkpoint of the saved node identification model.
  The following hyper-parameters control the high-order part:
  - USE_*: Whether to use the corresponding high-order factor.
  - DECOMP_SIZE and MFVI_ITER: Hyperparameters of mean field variational inference (refer to paper).

### Evaluation
Example command line to test a file input: `python predict.py -m <best.role.mdl> -i <input_dir> -o <output_dir> --format json`
  + Arguments:
    - -m, --model_path: Path to the trained model.
    - -i, --input_dir: Path to the input directory. LTF format sample files can be found in the `input` directory.
    - -o, --output_dir: Path to the output directory (json format). Output files are in the JSON format. Sample files can be found in the `output` directory.
    - --gpu: (optional) Use GPU
    - -d, --device: (optional) GPU device index (for multi-GPU machines).
    - -b, --batch_size: (optional) Batch size. For a 16GB GPU, a batch size of 10~15 is a reasonable value.
    - --max_len: (optional) Max sentence length. Sentences longer than this value will be ignored. You may need to decrease `batch_size` if you set `max_len` to a larger number.
    - --lang: (optional) Model language.
    - --format: Input file format (txt, ltf, or json).

### Data Format

Processed input example:
```json
{
    "doc_id": "AFP_ENG_20030401.0476",
    "sent_id": "AFP_ENG_20030401.0476-5",
    "entity_mentions": [
        {
            "id": "AFP_ENG_20030401.0476-5-E0",
            "start": 0,
            "end": 1,
            "entity_type": "GPE",
            "mention_type": "UNK",
            "text": "British"
        },
        ...
    ],
    "relation_mentions": [
        {
            "relation_type": "ORG-AFF",
            "id": "AFP_ENG_20030401.0476-5-R0",
            "arguments": [
                {
                    "entity_id": "AFP_ENG_20030401.0476-5-E1",
                    "text": "Chancellor",
                    "role": "Arg-1"
                },
                ...
            ]
        },
        ...
    ],
    "event_mentions": [
        {
            "event_type": "Personnel:Nominate",
            "id": "AFP_ENG_20030401.0476-5-EV0",
            "trigger": {
                "start": 9,
                "end": 10,
                "text": "named"
            },
            "arguments": [
                {
                    "entity_id": "AFP_ENG_20030401.0476-5-E4",
                    "text": "head",
                    "role": "Person"
                }
            ]
        }
    ],
    "tokens": [
        ...
    ],
    "pieces": [
        ...
    ],
    "token_lens": [
        ...
    ],
    "sentence": ...
}
```

The "start" and "end" of entities and triggers are token indices. The "arguments" of a relation refer to its head entity and tail entity.


Output example:
```json
{
    "doc_id": "HC0003PYD",
    "sent_id": "HC0003PYD-16",
    "token_ids": [
        ...
    ],
    "tokens": [
        ...
    ],
    "graph": {
        "entities": [
            [
                3,
                5,
                "GPE",
                "NAM",
                1.0
            ],
            ...
        ],
        "triggers": [
            ...
        ],
        "relations": [
            ...
        ],
        "roles": [
            ...
        ]
    }
}
```

## Citation
```bibtex
@inproceedings{jia-etal-2023-modeling,
    title = "Modeling Instance Interactions for Joint Information Extraction with Neural High-Order Conditional Random Field",
    author = "Jia, Zixia and Yan, Zhaohui and Han, Wenjuan and Zheng, Zilong and Tu, Kewei",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.766",
    doi = "10.18653/v1/2023.acl-long.766",
    pages = "13695--13710"
}
```

## Acknowledgments
The codebase of this repo is extended from [OneIE v0.4.8](https://blender.cs.illinois.edu/software/oneie/The_OneIE_Package.pdf)
