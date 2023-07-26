""" Loading data from disk and providing DataLoaders for PyTorch.
"""
import copy
from inspect import currentframe, getframeinfo
import os
import sys
from tqdm import tqdm
from collections import Counter, defaultdict, namedtuple
import json
import pickle
import random

import torch
from torch.utils.data import DataLoader
import utils
import torch.nn as nn


class Dataset:
    """ Loading data from disk and providing DataLoaders for PyTorch.

    Note: adds START token to the beginning of each sequence.
    """

    def __init__(self, args):
        self.args = args
        if 'vocab' in args['language']:
            self.vocab = args['language']['vocab']
        else:
            self.vocab, _, self.closing_bracket_ids = utils.get_vocab_of_bracket_types(args['language']['bracket_types'])
        args['language']['vocab_size'] = len(self.vocab)
        self.batch_size = args['training']['batch_size']

        paths = utils.get_corpus_paths_of_args(args)

        self.train_dataset = ObservationIterator(
                self.load_tokenized_dataset(paths['train']),
                paths['train_out'],
                mask_prob=args['training']['mask_prob'],
                vocab=self.vocab,
                mask_correct_prob=args['training']['mask_correct_prob'],
                mask_random_prob=args['training']['mask_random_prob'],
            )
        self.dev_dataset = ObservationIterator(
                self.load_tokenized_dataset(paths['dev']),
                paths['dev_out'],
                mask_prob=args['training']['mask_prob'],
                vocab=self.vocab,
                mask_correct_prob=args['training']['mask_correct_prob'],
                mask_random_prob=args['training']['mask_random_prob'],
            )
        self.test_dataset = ObservationIterator(
                self.load_tokenized_dataset(paths['test']),
                paths['test_out'],
                mask_prob=args['training']['mask_prob'],
                vocab=self.vocab,
                mask_correct_prob=args['training']['mask_correct_prob'],
                mask_random_prob=args['training']['mask_random_prob'],
            )

    def load_tokenized_dataset(self, filepath):
        """Reads in a conllx file; generates Observation objects

        For each sentence in a conllx file, generates a single Observation
        object.

        Args:
          filepath: the filesystem path to the conll dataset

        Returns:
          A list of Observations
        """
        tqdm.write('Getting dataset from {}'.format(filepath))
        observations = []
        lines = (x for x in open(filepath))
        for line in lines:
            tokens = [x.strip() for x in line.strip().split()]
            tokens = ['START'] + tokens
            if self.vocab:
                tokens = [self.vocab[x] for x in tokens]
            observation = torch.tensor(tokens)
            observations.append(observation)
        return observations

    def custom_pad(self, batch_observations):
        if self.args['training']['mask_prob'] == 0.0:
            seqs = [x[0][:-1].clone().detach() for x in
                    batch_observations]  # Cut out the last token
            attention_masks = None
        else:
            seqs = [x[0].clone().detach() for x in
                    batch_observations]  # Keep the last token
            attention_masks = [x[2].clone().detach() for x in batch_observations]
            attention_masks = nn.utils.rnn.pad_sequence(attention_masks, batch_first=True).to(self.args['device'])

        lengths = [len(x) for x in seqs]
        seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True).to(self.args['device'])
        labels = [x[1].clone().detach() for x in batch_observations]
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100).to(self.args['device'])
        return seqs, labels, attention_masks, lengths

    def get_train_dataloader(self, shuffle=True):
        """Returns a PyTorch dataloader over the training dataset.

        Args:
          shuffle: shuffle the order of the dataset.
          use_embeddings: ignored

        Returns:
          torch.DataLoader generating the training dataset (possibly shuffled)
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=shuffle)

    def get_dev_dataloader(self):
        """Returns a PyTorch dataloader over the development dataset.

        Args:
          use_embeddings: ignored

        Returns:
          torch.DataLoader generating the development dataset
        """
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=False)

    # def get_test20_dataloader(self):
    #     return DataLoader(self.test20_dataset, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=False)

    def get_test_dataloader(self):
        """Returns a PyTorch dataloader over the test dataset.

        Args:
          use_embeddings: ignored

        Returns:
          torch.DataLoader generating the test dataset
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=False)


class ObservationIterator:
    """ List Container for lists of Observations and labels for them.

    Used as the iterator for a PyTorch dataloader.
    """

    def __init__(
            self,
            observations,
            output_path=None,
            mask_prob=0.0,
            vocab=None,
            mask_correct_prob=0.0,
            mask_random_prob=0.0,
    ):
        assert 0.0 <= mask_correct_prob <= 1.0
        assert 0.0 <= mask_random_prob <= 1.0
        assert 0.0 <= mask_correct_prob + mask_random_prob <= 1.0

        self.observations = copy.deepcopy(observations)
        self.attention_mask_all = None
        masked_all = None
        if mask_prob != 0.0:
            assert vocab is not None, 'need vocab to perform additional masking'
            self.attention_mask_all = []
            masked_all = []
            for i, observation in enumerate(observations):
                seq_len, = observation.shape
                assert 0.0 <= mask_prob <= 1.0
                masked = []
                loop_cnt = 0
                while len(masked) == 0:  # Mask at least one token to be meaningful
                    rand = torch.rand((seq_len,))
                    mask_arr = (rand < mask_prob) * (observation != vocab['PAD']) * \
                               (observation != vocab['START']) * (observation != vocab['END'])
                    masked = torch.flatten(mask_arr.nonzero()).tolist()
                    loop_cnt += 1
                    if loop_cnt >= 100:
                        frameinfo = getframeinfo(currentframe())
                        print(f"\t Warning: possible infinite loop detected at file {frameinfo.filename}, line {frameinfo.lineno}")
                        break
                self.attention_mask_all.append(1 - mask_arr.long())
                if mask_correct_prob == 0.0 and mask_random_prob == 0.0:
                    self.observations[i][masked] = vocab['MASK']
                else:
                    all_token_ids = list(vocab.values())
                    for j in masked:
                        rand_mask_type = random.random()
                        if rand_mask_type < 1.0 - mask_correct_prob - mask_random_prob:
                            self.observations[i][j] = vocab['MASK']
                        elif rand_mask_type < 1.0 - mask_correct_prob:
                            self.observations[i][j] = random.choice(all_token_ids)
                masked_all.append(masked)
                if (i + 1) % 100000 == 0:
                    print(f"\t processed mask for {i + 1} sentences.")
        self.set_labels(
            observations,
            output_path=output_path,
            masked_all=masked_all,
        )

    def set_labels(
            self,
            observations,
            output_path=None,
            masked_all=None,
    ):
        """ Constructs and stores label for each observation.

        Args:
          observations: A list of observations describing a dataset
        """
        if masked_all is not None:
            assert output_path is None, 'Need to decide whether pre-computed output is masked.'
            self.labels = []
            for i, observation in enumerate(observations):
                labels = -100 * torch.ones(observation.shape, dtype=torch.long)
                labels[masked_all[i]] = observation[masked_all[i]]
                self.labels.append(labels)
            return

        if output_path is None:
            self.labels = []
            for observation in tqdm(observations, desc='[computing labels]'):
                label = copy.deepcopy(observation)  # LM must predict EOS
                label = label[1:]
                self.labels.append(label)
        else:
            with open(output_path, 'rb') as f:
                self.labels = pickle.load(f)
        assert len(self.observations) == len(self.labels)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        if self.attention_mask_all is not None:
            return self.observations[idx], self.labels[idx], self.attention_mask_all[idx]
        return self.observations[idx], self.labels[idx]
