import fnmatch
import unittest

from fitter import util
from fitter import data
from fitter import probabilities as prob

import numpy as np
import astropy.units as u


# class TestModel(unittest.TestCase):

#     def setUp(self):
#         self.obs = data.Observations('TEST')
#         self.model = data.Model(self.obs.initials, self.obs)

#     def test_getattr(self):
#         '''test the getattr, it should be able to retrieve values from theta
#         and from limepy'''

#     def test_mf(self):
#         '''actually maybe not this one, these test should maybe go in ssptool'''

#     def test_values(self):
#         '''test that the values of everything are what we would expect'''


#     def test_units(self):
#         '''test that all the units are assigned correctly'''


class TestObservations(unittest.TestCase):

    def setUp(self):
        self.obs = data.Observations('TEST')

    def test_mdata(self):
        '''test if all the observation level metadata is loaded correctly'''

        mdata = {'FeHe': -1, 'b': 10., 'l': 300., 'μ': 10.}

        self.assertEqual(self.obs.mdata, mdata)

    def test_loading_datasets(self):
        '''test if all datasets were loaded correctly (i.e. exist in obs)
        (probably just using the dataests property)
        '''
        ds_keys = {'mass_function', 'number_density', 'proper_motion/high_mass',
                   'proper_motion/low_mass', 'pulsar', 'velocity_dispersion'}

        self.assertEqual(self.obs.datasets.keys(), ds_keys)

    def test_loading_initials(self):
        '''test if all initials were loaded correctly (i.e. exist in obs)
        and that the correct ones were assigned to defaults
        '''
        initials = {'W0': 5., 'M': 5., 'rh': 5., 'ra': 5., 'g': 1., 'd': 5.,
                    'delta': 0.5, 's2': 0.5, 'F': 0.5, 'BHret': 0.5}

        # a1,a2,a3 should be defaults (don't exist in test file)
        for a in ('a1', 'a2', 'a3'):
            initials[a] = data.DEFAULT_INITIALS[a]

        self.assertEqual(self.obs.initials, initials)

    def test_get_dataset(self):
        '''test that we can __getitem__ datasets from observations'''

        ds = self.obs._dict_datasets['number_density']

        self.assertIs(self.obs.datasets['number_density'], ds)

    def test_get_variable(self):
        '''test that we can __getitem__ specific Variables from observations
        both using ['{dataset}/{variable}'] and ['{dataset}']['{variable}']
        '''
        ds = self.obs._dict_datasets['number_density']
        v = ds._dict_variables['r']

        self.assertIs(self.obs['number_density/r'], v)

        self.assertIs(self.obs['number_density']['r'], v)

    def test_valid_likelihoods(self):
        '''test that the `valid_likelihoods` property works as intended'''

        # Also contains 'pulsar' dataset, but with invalid variables
        likelihoods = {
            ('mass_function', prob.likelihood_mass_func),
            ('number_density', prob.likelihood_number_density),
            ('proper_motion/high_mass', prob.likelihood_pm_tot),
            ('proper_motion/high_mass', prob.likelihood_pm_ratio),
            ('proper_motion/high_mass', prob.likelihood_pm_R),
            ('proper_motion/high_mass', prob.likelihood_pm_T),
            ('proper_motion/low_mass', prob.likelihood_pm_R),
            ('proper_motion/low_mass', prob.likelihood_pm_T),
            ('velocity_dispersion', prob.likelihood_LOS)
        }

        valids = self.obs.valid_likelihoods

        self.assertCountEqual(valids, likelihoods)

        for v in valids:
            self.assertIn(v, likelihoods)


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.obs = data.Observations('TEST')
        self.ds = self.obs['pulsar']

    def test_mdata(self):
        '''test if all the dataset metadata is loaded correctly'''

        mdata = {'source': 'Test Source', 'm': 1.5}

        self.assertEqual(self.ds.mdata, mdata)

    def test_loading_variables(self):
        '''test if all variables were loaded correctly (i.e. exist in dataset)
        (probably just using the variables property)
        '''
        var_keys = {'r', 'Δr', 'a', 'Δa,up', 'Δa,down'}

        self.assertEqual(self.ds.variables.keys(), var_keys)

    def test_get_variable(self):
        '''test that we can __getitem__ specific Variables from dataset'''

        v = self.ds._dict_variables['r']

        self.assertIs(self.ds['r'], v)

    def test_build_err(self):
        '''test that `build_err` builds the correct error, both asym and not'''

        r = self.ds['r']
        val = np.array([15, 15, 15, 15, 15])

        # Symmetric (single value) errors

        e_sym = self.ds.build_err('r', r, val * self.ds['r'].unit)

        np.testing.assert_array_equal(e_sym.value, np.array([2, 2, 2, 2, 2]))

        # Asymmetric (up and down) errors

        e_asym = self.ds.build_err('a', r, val * self.ds['a'].unit)

        np.testing.assert_array_equal(e_asym.value, np.array([1, 1, 1, -1, -1]))


class TestVariable(unittest.TestCase):

    def setUp(self):
        self.obs = data.Observations('TEST')
        self.ds = self.obs['pulsar']
        self.var = self.ds['r']

    def test_mdata(self):
        '''test if all the variable metadata is loaded correctly'''

        mdata = {'unit': 'arcsec'}

        self.assertEqual(self.var.mdata, mdata)

    def test_values(self):
        vals = np.array([0.1, 0.5, 1.0, 1.5, 2.0])

        np.testing.assert_array_equal(vals, self.var.value)

    def test_units(self):
        '''test if the variable has the correct units set'''

        self.assertIs(self.var.unit, u.Unit('arcsec'))


# test every cluster data file in resources to ensure compliance
class TestResources(unittest.TestCase):

    def _check_for_error(self, key, dataset):
        if (f'Δ{key},up' in dataset) and (f'Δ{key},down' in dataset):
            self.assertTrue(True)
        else:
            self.assertIn(f'Δ{key}', dataset)

    def test_data_compliance(self):

        for cluster in util.cluster_list():

            with self.subTest(cluster=cluster):

                obs = data.Observations(cluster)

                # test initials (make sure there is no extra initial parameters)

                extra = obs.initials.keys() - data.DEFAULT_INITIALS.keys()

                self.assertEqual(len(extra), 0)

                # test datasets for required variables

                for key, dataset in obs.datasets.items():

                    # Pulsars

                    if fnmatch.fnmatch(key, '*pulsar*'):
                        self.assertIn('r', dataset)

                        if 'P' in dataset:

                            self.assertIn('Pdot_meas', dataset)
                            self._check_for_error('Pdot_meas', dataset)

                        elif 'Pb' in dataset:

                            self.assertIn('Pbdot_meas', dataset)
                            self._check_for_error('Pbdot_meas', dataset)

                        else:
                            assert False, f"None of ('P', 'Pb') in {dataset}"

                    # LOS Velocity Dispersion

                    elif fnmatch.fnmatch(key, '*velocity_dispersion*'):
                        self.assertIn('r', dataset)
                        self.assertIn('σ', dataset)
                        self._check_for_error('σ', dataset)

                    # Number Density

                    elif fnmatch.fnmatch(key, '*number_density*'):
                        self.assertIn('r', dataset)
                        self.assertIn('Σ', dataset)
                        self._check_for_error('Σ', dataset)

                    # Proper Motion Dispersion

                    elif fnmatch.fnmatch(key, '*proper_motion*'):
                        self.assertIn('r', dataset)

                        # make sure that atleast one PM is there
                        pm_fields = ('PM_tot', 'PM_ratio', 'PM_R', 'PM_T')
                        self.assertTrue(
                            any([field in dataset for field in pm_fields]),
                            f"None of {pm_fields} in {dataset}"
                        )

                        # Check for corresponding errors
                        if 'PM_tot' in dataset:
                            self._check_for_error('PM_tot', dataset)

                        if 'PM_ratio' in dataset:
                            self._check_for_error('PM_ratio', dataset)

                        if 'PM_R' in dataset:
                            self._check_for_error('PM_R', dataset)

                        if 'PM_T' in dataset:
                            self._check_for_error('PM_T', dataset)

                    # Mass Function

                    elif fnmatch.fnmatch(key, '*mass_function*'):
                        self.assertIn('N', dataset)
                        self.assertIn('bin', dataset)
                        self.assertIn('Δmbin', dataset)
                        self.assertIn('mbin_mean', dataset)
                        self.assertIn('mbin_width', dataset)


if __name__ == '__main__':
    unittest.main()
