# BIAS in sampling
because 1st citation is always chosen



# Epochs
single epoch but huge training time
5hrs for single epoch


# Efficient, context-aware legal citation recommendation
Code for the paper "Efficient, context-aware legal citation recommendation" by Louis Kapp.
## Setup
1. Download the preprocessed citation dataset from [Stanford's RegLab announcement blog](https://reglab.stanford.edu/data/bva-case-citation-dataset/) and unzip it into `data/text/original`.
2. Download the original citation vocabulary from [Stanford's RegLab announcement blog](https://reglab.stanford.edu/data/bva-case-citation-dataset/) and save it as `data/vocabs/original.pkl`.
3. Create a conda environment from the [env.yaml](env.yaml) file which installs all necessary dependencies with `conda env create -f env.yaml`.
4. Follow the steps outlined below to reproduce the training results.

## Preprocessing

All preprocessing scripts can be found in the [preprocessing](preprocessing) folder.

1. Run [downsize_vocab](preprocessing/downsize_vocab.py) to create multiple files with differing (smaller) vocabulary sizes. Files will be saved in the [data/vocabs](data/vocabs) directory.
2. Run [same_size_vocab](preprocessing/same_size_vocab.py) to create a vocab file of the original size but using a new schema. Will be saved in the [data/vocabs](data/vocabs) directory.
3. Run [preprocess_texts](preprocessing/preprocess_texts.py) to create binary dataset files corresponding to the different vocab sizes. Files will be saved in the [data/text/preprocessed](data/text/preprocessed) directory. Make sure to have enough disk space available (~ 10 GB).
4. Run [unite_preprocessed_texts](preprocessing/unite_preprocessed_texts.py) to create a single binary dataset file for each vocab size. Files will be saved in the [data/text/preprocessed](data/text/preprocessed) directory. This will take up another ~10 GB of disk space. However, after this is done, you can delete the dataset chunks that were created in step 3.
5. Run [split](preprocessing/split.py) to create a train, development and test set. It is recommended to use shuffled=False, since shuffling could artifically boost the models performance. Without shuffling, we can ensure that the all data splits will consist of samples from different opinion texts. With shuffling, even though the samples are different, they might still be from the same opinion text which could lead to a biased evaluation.

## Datasets

There are 3 different dataset classes that we experimented with:
1. The version in [dataset_low_ram](dataset_low_ram.py) loads each sample from a single binary .pt file. It is not used in the final version.

2. [dataset_original](dataset_original.py) replicates the original data sampling from Zihan Huang, Charles Low, Mengqiu Teng, Hongyi Zhang, Daniel E. Ho, Mark Krass and Matthias Grabmair. It is not being used in the final version.

3. The final dataset version is [dataset_single_file](dataset_single_file.py) which stores all data samples in a single file to reduce disk I/O. It is used in [trainer/task](trainer/task.py).

## Training and Evaluation

Run [trainer/task](trainer/task.py) to train and evaluate a model on teh development set. The following arguments are available:
- --output_fp: filepath to the directory where model checkpoints and evaluations can be saved. If you use [weights and biases](https://wandb.ai/) for logging, this directory is not too important since checkpoints and metrics will be saved there.
- --checkpoint: checkpoint to load the model from. Is an integer that corresponds to the epoch number.
- --vocab_size: size of the citation vocabulary that you want to use
- --dataset_type: either "ordered" or "shuffled". Determines whether the dataset is shuffled before being split into train, development and test set. "Ordered" is recommended.

## Performance testing
Run [trainer/test](trainer/test.py) to compute Recall@1/3/5/20 on the test set. The following arguments are available:
- --output_fp: filepath to the directory where model checkpoints and evaluations can be saved. If you use [weights and biases](https://wandb.ai/) for logging, this directory is not too important since checkpoints and metrics will be saved there.
- --checkpoint: checkpoint to load the model from. Is an integer that corresponds to the epoch number.
- --vocab_size: size of the citation vocabulary that you want to use
- --dataset_type: either "ordered" or "shuffled". Determines whether the dataset is shuffled before being split into train, development and test set. "Ordered" is recommended.

## [Weights and biases](https://wandb.ai/) setup
If you want to use weights and biases for logging training, evaluation or testing, you need to create a [keys.py][keys.py] file in the root directory of this project. Save your API key there like this: `WANDB_API_KEY: str = "<YOUR_API_KEY>"
`. You can find your API key on the [weights and biases website](https://wandb.ai/authorize).

### My weights and biases logs
You can find my logs of evaluation metrics publicly in [this wandb project](https://wandb.ai/advanced-nlp/legal-citation-rec?workspace=user-).
