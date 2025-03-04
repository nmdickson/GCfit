#!/usr/bin/env python3

import gcfit

import json
import logging
import pathlib
import argparse

import emcee.moves


default_dir = f"{pathlib.Path.home()}/.GCfit"

move_choices = {
    'stretchmove': emcee.moves.StretchMove, 'walkmove': emcee.moves.WalkMove,
    'demove': emcee.moves.DEMove, 'desnookermove': emcee.moves.DESnookerMove,
    'mhmove': emcee.moves.MHMove, 'redbluemove': emcee.moves.RedBlueMove,
    'gaussianmove': emcee.moves.GaussianMove, 'kdemove': emcee.moves.KDEMove,
}

bound_choices = {'none', 'single', 'multi', 'balls', 'cubes'}
sample_choices = {'auto', 'unif', 'rwalk', 'rstagger',
                  'slice', 'rslice', 'hslice'}


def pos_int(arg):
    '''ensure arg is a positive integer, for use as `type` in ArgumentParser'''

    if not arg.isdigit():
        mssg = f"invalid positive int value: '{arg}'"
        raise argparse.ArgumentTypeError(mssg)

    return int(arg)


class RepeatedCallHandler(logging.FileHandler):
    '''logging Handler which stops repetition of the same log messages

    A handler class which writes formatted logging records to disk files,
    with a very rough check for often repeated messages, which are noted and
    replaced by a single message.

    Currently only well-supported in serial cases (but will somewhat work in
    parallel).
    '''

    _current_record = None
    _repetition_count = 0

    def emit(self, record):

        # first time, don't bother checking
        if self._current_record is None:
            super().emit(record)

        # repeated record, don't emit just count
        elif record.getMessage() == self._current_record.getMessage():
            self._repetition_count += 1

        # new record, emit then restart
        else:

            if (N := self._repetition_count) > 0:
                self._current_record.msg += f' (repeated {N-1} times)'
                super().emit(self._current_record)

            self._repetition_count = 0
            super().emit(record)

        self._current_record = record

    def close(self):
        '''On close, attempt to write out the currently repeating record first
        '''
        if (N := self._repetition_count) > 0:
            self._current_record.msg += f' (repeated {N-1} times)'
            super().emit(self._current_record)

        # Usual close method
        super().close()


def main():

    # ----------------------------------------------------------------------
    # Command line argument parsing
    # ----------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='fit some GCs')

    parser.add_argument('cluster', help='Common name of the cluster to model')

    # ----------------------------------------------------------------------
    # Common arguments to all samplers
    # ----------------------------------------------------------------------

    shared_parser = argparse.ArgumentParser(add_help=False)

    parallel_group = shared_parser.add_mutually_exclusive_group()
    parallel_group.add_argument("--Ncpu", default=2, type=pos_int,
                                help="Number of `multiprocessing` processes")
    parallel_group.add_argument("--mpi", action="store_true",
                                help="Run with MPI rather than multiprocessing")

    shared_parser.add_argument('--restrict-to', default=None,
                               choices={None, 'local', 'core'},
                               help='Optionally restrict datafiles used to '
                                    '"core" or "local" cluster files')
    shared_parser.add_argument('--savedir', default=default_dir,
                               help='location of saved sampling runs')
    shared_parser.add_argument('-i', '--initials',
                               help='alternative JSON file '
                                    'with different intials')
    shared_parser.add_argument('-p', '--priors', dest='param_priors',
                               help='alternative JSON file '
                                    'with different priors')

    shared_parser.add_argument('--fix', dest='fixed_params', nargs='*',
                               help='Parameters to fix, '
                                    'not estimate from the sampler')

    shared_parser.add_argument('--exclude', nargs='*',
                               dest='excluded_likelihoods',
                               help='Likelihood components to '
                                    'exclude from posteriors')

    shared_parser.add_argument('--hyperparams', dest='hyperparams',
                               action='store_true',
                               help="Use Bayesian hyperparams")

    shared_parser.add_argument('--verbose', action='store_true')
    shared_parser.add_argument('--debug', action='store_true')

    # ----------------------------------------------------------------------
    # Subparsers for each sampler
    # ----------------------------------------------------------------------

    subparsers = parser.add_subparsers(title="Sampler",
                                       dest="sampler", required=True,
                                       help="Which Sampler algorithm to use in "
                                            "fitting the cluster")

    # ----------------------------------------------------------------------
    # MCMC sampling with emcee
    # ----------------------------------------------------------------------

    parser_MCMC = subparsers.add_parser('MCMC', parents=[shared_parser])

    parser_MCMC.add_argument('-N', '--Niters', required=True, type=pos_int,
                             help='Number of sampling iterations')
    parser_MCMC.add_argument('--Nwalkers', required=True, type=pos_int,
                             help='Number of walkers for MCMC sampler')

    parser_MCMC.add_argument('--moves', type=str.lower, nargs='*',
                             default=['stretchmove'],
                             choices=move_choices.keys(),
                             help="Alternative MCMC move proposal algorithm to "
                                  "use. Multiple moves will be given equal "
                                  "random weight")

    parser_MCMC.add_argument('--continue', dest='cont_run', action='store_true',
                             help='Continue from previous saved run')
    parser_MCMC.add_argument('--backup', action='store_true',
                             help='Create continuous backups during run')

    parser_MCMC.add_argument('--show-progress', action='store_true',
                             dest='progress', help="Display progress bar")

    parser_MCMC.set_defaults(fit_func=gcfit.MCMC_fit)

    # ----------------------------------------------------------------------
    # Nested Sampling with dynesty
    # ----------------------------------------------------------------------
    # TODO make the "current_batch" storage optional

    parser_nest = subparsers.add_parser('nested', parents=[shared_parser])

    parser_nest.add_argument('--pfrac', default=1.0, type=float,
                             help='Posterior weighting fraction f_p')
    parser_nest.add_argument('--dlogz', default=0.25, type=float,
                             help='Δln(Z) tolerance initial stopping condition')
    parser_nest.add_argument('--maxfrac', default=0.8, type=float,
                             help='The fractional threshold, relative to the '
                                  'peak weight, used to determine likelihood '
                                  'bounds for dynamic sampling. Default to 0.8')
    parser_nest.add_argument('--eff-samples', default=5000, type=pos_int,
                             help='Number of effective posterior samples '
                                  'stopping condition')
    parser_nest.add_argument('--maxiter', default=None, type=pos_int,
                             help='Maximum number of iterations allowed. May '
                                  'end sampling before the stopping conditions '
                                  'are met')
    parser_nest.add_argument('--init-maxiter', default=None, type=pos_int,
                             help='Maximum number of iterations allowed in the '
                                  'baseline run. Default is no limit')
    parser_nest.add_argument('--N-per-batch', default=100, type=pos_int,
                             dest='Nlive_per_batch',
                             help='Number of live points to add each batch. '
                                  'See dynesty for info on defaults')
    parser_nest.add_argument('--bound-type', default='multi',
                             choices=bound_choices,
                             help='Method used to bound sampling on the prior')
    parser_nest.add_argument('--sample-type', default='auto',
                             choices=sample_choices,
                             help='Method used to sample uniformly within the '
                                  'likelihood, based on the provided bounds')
    parser_nest.add_argument('--plat-wt-func', action='store_true',
                             help="Use custom `util.plateau_weight_function`")

    parser_nest.set_defaults(fit_func=gcfit.nested_fit)

    args = parser.parse_args()

    # ----------------------------------------------------------------------
    # Args preprocessing
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Common arguments
    # ----------------------------------------------------------------------

    if args.initials:

        if (init_file := pathlib.Path(args.initials)).is_file():

            with open(init_file, 'r') as init_of:
                args.initials = json.load(init_of)

        else:
            parser.error(f"Cannot access '{init_file}': No such file")

    if args.param_priors:

        if (bnd_file := pathlib.Path(args.param_priors)).is_file():

            with open(bnd_file, 'r') as init_of:
                args.param_priors = json.load(init_of)

        else:
            parser.error(f"Cannot access '{bnd_file}': No such file")

    pathlib.Path(args.savedir).mkdir(exist_ok=True)

    # ----------------------------------------------------------------------
    # MCMC specific arguments
    # ----------------------------------------------------------------------

    if args.sampler == 'MCMC':

        if args.cont_run:
            raise NotImplementedError

        args.moves = [move_choices[mv]() for mv in args.moves]

    # ----------------------------------------------------------------------
    # Nested Sampling specific arguments
    # ----------------------------------------------------------------------

    elif args.sampler == 'nested':

        # TODO add more of these options
        args.initial_kwargs = {
            'maxiter': args.init_maxiter or float('inf'),
            'nlive': args.Nlive_per_batch,
            'dlogz': args.dlogz
        }

        args.batch_kwargs = {
            'maxiter': args.maxiter or float('inf'),
            'nlive_new': args.Nlive_per_batch
        }

        del args.dlogz
        del args.maxiter
        del args.init_maxiter
        del args.Nlive_per_batch

    # ----------------------------------------------------------------------
    # Setup logging
    # ----------------------------------------------------------------------

    if debug := args.debug:
        args.verbose = True

    del args.debug

    config = {
        'level': logging.DEBUG if debug else logging.INFO,
        'format': ('%(process)s|%(asctime)s|'
                   '%(name)s:%(module)s:%(funcName)s:%(message)s'),
        'datefmt': '%H:%M:%S'
    }

    if args.verbose:
        log_filename = f"{args.savedir}/fitter_{args.cluster}.log"
        config['handlers'] = [RepeatedCallHandler(log_filename)]
    else:
        config['handlers'] = [logging.NullHandler()]

    logging.basicConfig(**config)

    # ----------------------------------------------------------------------
    # Call gcfit
    # ----------------------------------------------------------------------

    del args.sampler

    fit_func = args.fit_func
    del args.fit_func

    logging.debug(f"{args=}")

    fit_func(**vars(args))
