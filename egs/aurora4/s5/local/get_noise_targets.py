#!/usr/bin/env python

# Copyright 2020  Desh Raj (Johns Hopkins University)
# Apache 2.0

# This script prepares targets for noise labels. It takes
# an utt2noise file and for each utt, it creates a one-hot
# output label vector.

from __future__ import division

import argparse
import logging
import numpy as np
import subprocess
import sys
import itertools
from collections import defaultdict

sys.path.insert(0, 'steps')
import libs.common as common_lib


def get_args():
    parser = argparse.ArgumentParser(
        description="""This script prepares targets for noise labels. It takes
            an utt2noise file and for each utt, it creates a one-hot output
            label vector.
            """)

    parser.add_argument("utt2noise", type=str,
                        help="Input utt2noise file")
    parser.add_argument("utt2num_frames", type=str,
                        help="Input utt to num_frames")
    parser.add_argument("targets_ark", type=str,
                        help="Targets ark file")

    args = parser.parse_args()
    return args


def groupby(iterable, keyfunc):
    """Wrapper around ``itertools.groupby`` which sorts data first."""
    iterable = sorted(iterable, key=keyfunc)
    for key, group in itertools.groupby(iterable, keyfunc):
        yield key, group

def run(args):
    utt2noise = {}
    with common_lib.smart_open(args.utt2noise) as f:
        for line in f:
            parts = line.strip().split()
            utt2noise[parts[0]] = parts[1]

    utt2num_frames = {}
    with common_lib.smart_open(args.utt2num_frames) as f:
        for line in f:
            parts = line.strip().split()
            utt2num_frames[parts[0]] = int(parts[1])

    
    utt2targets = {}
    label_types = list(set(utt2noise.values()))

    one_hot_encode = {}
    for i,label in enumerate(label_types):
        one_hot = [0]*len(label_types)
        one_hot[i] = 1
        one_hot_encode[label] = one_hot

    utt2targets = {}
    for utt in utt2num_frames:
        num_frames = utt2num_frames[utt]
        label = utt2noise[utt]
        targets_mat = np.tile(one_hot_encode[label], (num_frames,1))
        utt2targets[utt] = targets_mat

    with common_lib.smart_open(args.targets_ark, 'w') as f:
        for utt_id in sorted(utt2targets.keys()):
            common_lib.write_matrix_ascii(f, utt2targets[utt_id].tolist(), key=utt_id)

def main():
    args = get_args()
    try:
        run(args)
    except Exception:
        raise

if __name__ == "__main__":
    main()