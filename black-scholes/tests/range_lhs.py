import unittest
from utils.sampler import latin_hypercube_sampling


class TestLHC(unittest.TestCase):

    def test_shape(self):
        # create dictionary of factors
        dict_ranges = dict()
        dict_ranges['x1'] = (2, 7)
        dict_ranges['x2'] = (8, 10)
        dict_ranges['x3'] = (100, 200)
        # create samples from LHS
        example_sample_lhs = latin_hypercube_sampling(number_sample_points=100000,
                                                      dict_range_for_each_variable=dict_ranges)

        self.assertEqual(example_sample_lhs.shape, (100000, 3), "Should be (100000,3)")

    def test_range(self):
        # create dictionary of factors
        dict_ranges = dict()
        dict_ranges['x1'] = (2, 7)
        dict_ranges['x2'] = (100, 200)
        # create samples from LHS
        example_sample_lhs = latin_hypercube_sampling(number_sample_points=100000,
                                                      dict_range_for_each_variable=dict_ranges)

        min_value_first_factor = example_sample_lhs[:, 0].min()
        max_value_first_factor = example_sample_lhs[:, 0].max()

        min_value_second_factor = example_sample_lhs[:, 1].min()
        max_value_second_factor = example_sample_lhs[:, 1].max()

        self.assertLessEqual(dict_ranges['x1'][0], min_value_first_factor,
                             "Min of first factor should be greater than equal to 2")

        self.assertLessEqual(max_value_first_factor, dict_ranges['x1'][1],
                             "Max of first factor should be less than equal to 7")

        self.assertLessEqual(dict_ranges['x2'][0], min_value_second_factor,
                             "Min of first factor should be greater than equal to 100")

        self.assertLessEqual(max_value_second_factor, dict_ranges['x2'][1],
                             "Max of first factor should be less than equal to 200")


if __name__ == '__main__':
    unittest.main()
