#!/usr/bin/env python3

from gcfit import analysis

import logging
import argparse
import datetime


def pos_int(arg):
    '''ensure arg is a positive integer, for use as `type` in ArgumentParser'''

    if not arg.isdigit():
        mssg = f"invalid positive int value: '{arg}'"
        raise argparse.ArgumentTypeError(mssg)

    return int(arg)


def main():

    # ----------------------------------------------------------------------
    # Command line argument parsing
    # ----------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description='Generate and save confidence intervals for models based '
                    'on given run output file(s). Parallelized and designed to '
                    'be run on a computing cluster, to fill in any files which '
                    'did not have their CIs created at runtime.'
    )

    parser.add_argument('filenames', nargs='+',
                        help='Name of the run output file')

    parser.add_argument('-N', default=1000, type=pos_int,
                        help='Number of models used to compute CIs')

    parser.add_argument('--Ncpu', default=1, type=pos_int,
                        help='Number of processes used in parallelization')

    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite saved CIs if already exist. Be careful')

    parser.add_argument('--mask', nargs=3, default=False,
                        metavar=('PARAM', 'LOWER_LIM', 'UPPER_LIM'),
                        help='Apply a mask to each run based on a param')

    parser.add_argument('-o', '--output', nargs='+', default=None,
                        help='Alternative files only for saving CI outputs')

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    if args.output is None:
        args.output = args.filenames

    if args.debug:
        now = datetime.datetime.now()
        config = {
            'level': logging.DEBUG if args.debug else logging.INFO,
            'format': ('%(process)s|%(asctime)s|'
                       '%(name)s:%(module)s:%(funcName)s:%(message)s'),
            'datefmt': '%H:%M:%S',
            'filename': f'gen_model_CI_{now.isoformat()}.log'
        }

        logging.basicConfig(**config)

    # ----------------------------------------------------------------------
    # Generate Models
    # ----------------------------------------------------------------------

    rc = analysis.RunCollection.from_files(args.filenames)

    if args.mask:
        mask_prm = args.mask[0]
        mask_dnlim, mask_uplim = float(args.mask[1]), float(args.mask[2])

        for run in rc:
            run.slice_on_param(mask_prm, mask_dnlim, mask_uplim)

    mc = rc.get_CImodels(N=args.N, Nprocesses=args.Ncpu, load=False)

    mc.save(args.output, overwrite=args.overwrite)
