"""
Utilities for determining paths to corpora, results, models
given config dictionaries describing an experiment, as well
as determining canonical vocabulary ordering
"""

import os
import string
import re
import copy

def get_identifier_iterator():
    """ Returns an iterator to provide unique ids to bracket types.
    """
    ids = iter(list(string.ascii_lowercase))
    k = 1
    while True:
        try:
            str_id = next(ids)
        except StopIteration:
            ids = iter(list(string.ascii_lowercase))
            k += 1
            str_id = next(ids)
        yield str_id*k

def get_results_dir_of_args(args):
    """
    Takes a (likely yaml-defined) argument dictionary
    and returns the directory to which results of the
    experiment defined by the arguments will be saved
    """
    return args['reporting']['reporting_loc']

def get_corpus_paths_of_args(args):
    paths = {
            'train': args['corpus']['train_corpus_loc'],
            'dev': args['corpus']['dev_corpus_loc'],
            'test': args['corpus']['test_corpus_loc'],
            'train_out': None,
            'dev_out': None,
            'test_out': None,
        }
    if 'train_output_loc' in args['corpus']:
        paths['train_out'] = args['corpus']['train_output_loc']
    if 'dev_output_loc' in args['corpus']:
        paths['dev_out'] = args['corpus']['dev_output_loc']
    if 'test_output_loc' in args['corpus']:
        paths['test_out'] = args['corpus']['test_output_loc']
    return paths

def get_lm_path_of_args(args):
    results_dir = get_results_dir_of_args(args)
    return os.path.join(results_dir, args['name']+'.params')
