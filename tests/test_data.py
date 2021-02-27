import fnmatch
import unittest

from fitter import data

# test data module, all loader stuff on test resources

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.obs = data.Observations('TEST')

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
        # should all be == 99.
        initials = {'W0': 99., 'M': 99., 'rh': 99., 'ra': 99., 'g': 99.,
                    'delta': 99., 's2': 99., 'F': 99., 'BHret': 99., 'd': 99.}

        # a1,a2,a3 should be defaults (don't exist in test file)
        for a in ('a1', 'a2', 'a3'):
            initials[a] = data.DEFAULT_INITIALS[a]

        self.assertEqual(self.obs.initials, initials)

    def test_get_dataset(self):
        '''test that we can __getitem__ datasets from observations'''

        ds = self.obs._dict_datasets['number_density']

        self.assertIs(self.obs.datasets['number_density'], ds)

    def test_loading_variables(self):
        '''test if all variables were loaded correctly (i.e. exist in dataset)
        (probably just using the variables property)
        '''
        var_keys = {'r', 'Σ', 'ΔΣ'}

        self.assertEqual(self.obs['number_density'].variables.keys(), var_keys)

    def test_get_variables_from_observations(self):
        '''test that we can __getitem__ specific Variables from observations'''
        ds = self.obs._dict_datasets['number_density']
        v = ds._dict_variables['r']

        self.assertIs(self.obs['number_density/r'], v)

        self.assertIs(self.obs['number_density']['r'], v)

    def test_observations_mdata(self):
        '''test if all the observation level metadata is loaded correctly'''

        mdata = {'FeHe': 99.}

        self.assertEqual(self.obs.mdata, mdata)

    def test_dataset_mdata(self):
        '''test if all the dataset metadata is loaded correctly'''

        mdata = {'source': 'De Boer et al. (2019)'}

        self.assertEqual(self.obs._dict_datasets['number_density'].mdata, mdata)

    def test_variable_mdata(self):
        '''test if all the variable metadata is loaded correctly'''

        ds = self.obs._dict_datasets['number_density']
        mdata = {'unit': 'arcsec'}

        self.assertEqual(ds._dict_variables['r'].mdata, mdata)


# test every cluster data file in resources to ensure compliance
# TODO update docs with which fields are required
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
                            self._check_for_error('P', dataset)

                            self.assertIn('Pdot_meas', dataset)
                            self._check_for_error('Pdot_meas', dataset)

                        elif 'Pb' in dataset:
                            self._check_for_error('Pb', dataset)

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
