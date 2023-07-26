""" Trains/runs a language model on data available tokenized sentence-per-line format.

The main interface to running experiments with this codebase.

Usage:
      python rnns_stacks/run_lm.py <config.yaml>
"""

import torch
import yaml
import os
from tqdm import tqdm
from argparse import ArgumentParser
from dataset import Dataset
from training_regimen import train
import utils
import transformer
# import reporter
import wandb


def create_args(config_file):
    args = yaml.load(open(config_file))
    args['training']['learning_rate'] = float(args['training']['learning_rate'])
    args['training']['weight_decay'] = float(args['training']['weight_decay'])
    args['training']['dropout'] = float(args['training']['dropout'])
    args['training']['zero_init_noise'] = float(args['training']['zero_init_noise'])

    if args['lm']['lm_type'] in {'BertForMaskedLM', 'BertForMaskedLMCustom'}:
        args['name'] = f"_mask_{args['training']['mask_prob']}"
    else:
        raise NotImplementedError('Model not supported.')

    if args['language']['name'] == 'lda':
        args['name'] = "topic{}_word{}_{}_{}_lr{}_wd{}_hiddenlayers{}_heads{}_hiddendim{}_{}_{}_{}_dropout{}_noise{}".format(
            args['language']['num_topics'],
            args['language']['num_words'],
            args['lm']['lm_type'],
            args['training']['optimizer'],
            args['training']['learning_rate'],
            args['training']['weight_decay'],
            args['lm']['num_layers'],
            args['lm']['num_heads'],
            args['lm']['hidden_dim'],
            args['lm']['embedding_type'],
            args['lm']['token_embedding_type'],
            args['training']['objective'],
            args['training']['dropout'],
            args['training']['zero_init_noise'],
        ) + args['name']
    else:
        raise NotImplementedError('Language not supported.')

    if args['training']['mask_correct_prob'] > 0:
        args['name'] += f"_correct{args['training']['mask_correct_prob']}"
    if args['training']['mask_random_prob'] > 0:
        args['name'] += f"_random{args['training']['mask_random_prob']}"
    if (not args['lm']['residual']) and (not args['lm']['attn_output_fc']) and (not args['lm']['bert_intermediate']) \
            and (not args['lm']['bert_output']) and (not args['lm']['bert_head_transform']):
        args['name'] += '_noMany'
    else:
        if not args['lm']['residual']:
            args['name'] += '_noRes'
        if not args['lm']['attn_output_fc']:
            args['name'] += '_noAttnOutFC'
        if not args['lm']['bert_intermediate']:
            args['name'] += '_noBertIntermediate'
        if not args['lm']['bert_output']:
            args['name'] += '_noBertOutput'
        if not args['lm']['bert_head_transform']:
            args['name'] += '_noBertHeadTransform'
    if args['lm']['freeze_uniform_attention']:
        args['name'] += '_freezeUniformAttention'
    if args['lm']['freeze_id_value_matrix']:
        args['name'] += '_freezeWvI'
    if args['lm']['freeze_block_value_matrix']:
        args['name'] += '_freezeWvBlock'
    if args['lm']['freeze_decoder_to_I']:
        args['name'] += '_freezeWdecI'
    if args['lm']['no_softmax']:
        args['name'] += '_noSoftmax'

    # Determine whether CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args['device'] = device
    return args


def load_dataset(args):
    dataset = Dataset(args)
    return dataset


def init_lm(args):
    return transformer.PytorchTransformerModel(args)


if __name__ == '__main__':
    argp = ArgumentParser()
    argp.add_argument('config')
    args = argp.parse_args()
    args = create_args(args.config)

    # Must load dataset first, since args['language']['vocab_size'] depends on data
    dataset = load_dataset(args)

    name_base = args['name']
    for experiment_index in range(args['experiment']['repeat']):
        args['name'] = name_base + str(experiment_index)
        if args['name'] in {
            # list of runs to skip e.g. because they have been completed previously
        }:
            print(f"Skipped {args['name']}")
            continue

        # Construct the language model
        print('Construct the language model with args', args)
        lm_model = init_lm(args)

        # Prepare to write results
        output_dir = utils.get_results_dir_of_args(args)
        tqdm.write('Writing results to {}'.format(output_dir))
        os.makedirs(utils.get_results_dir_of_args(args), exist_ok=True)

        wandb.init(
            project="transformer_topic_model_LDA",
            name=args['name'],
            reinit=True,
        )
        wandb.config = args

        # Train and load most recent parameters
        train(args, lm_model, dataset.get_train_dataloader(), dataset.get_dev_dataloader(),
              steps_between_logging=50,
              steps_between_evals=1000,
              dataset=dataset,
              )
