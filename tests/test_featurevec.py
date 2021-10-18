import unittest

import numpy as np
import pandas as pd

from tests.conftest import init_biased_random_data, init_biased_duplicate_data
from fvecs import FeatureVeca
from fvecs.utils import cooccurance_matrix, cart2pol, pol2cart


class TestFeatureVec(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rdata, cls.ry, cls.ry_hat, cls.cat_feats, cls.num_feats = \
            init_biased_random_data(10, 0.4)  # random large dataset just to see things run
        cls.rclassify_feature_vec_1 = FeatureVec('classify',
                                                 feature_names=cls.rdata.columns,
                                                 max_depth=3,
                                                 max_sentences=20000,
                                                 exp_rand_tree_size=True,
                                                 tree_generator=None)
        cls.rdata_dup, cls.ry_dup, cls.ry_hat_dup, cls.cat_feats_dup, cls.num_feats_dup = init_biased_duplicate_data(10,
                                                                                                                     0.4)  # random dataset with duplicate features
        cls.rclassify_feature_vec_2 = FeatureVec('classify',
                                                 feature_names=cls.rdata_dup.columns,
                                                 max_depth=3,
                                                 max_sentences=20000,
                                                 exp_rand_tree_size=True,
                                                 tree_generator=None)

    def test_fit(self):
        self.rclassify_feature_vec_1.fit(self.rdata, self.ry, self.cat_feats, bagging=False)
        self.assertEqual(np.sum(self.rclassify_feature_vec_1._stds), 0)  # no bagging
        self.assertEqual(len(self.rclassify_feature_vec_1._vectors), self.rdata.shape[-1])

        self.rclassify_feature_vec_1.fit(self.rdata, self.ry, self.cat_feats, bagging=10, restart=True)
        self.assertGreater(np.sum(self.rclassify_feature_vec_1._stds), 0)

        self.rclassify_feature_vec_2.fit(self.rdata_dup, self.ry_dup, self.cat_feats_dup, bagging=False)
        dup_difference = np.sum(np.linalg.norm(self.rclassify_feature_vec_2.vectors[:7] - \
                                               self.rclassify_feature_vec_2.vectors[7:], axis=-1))
        rnd_idxs = np.random.permutation(14)
        rnd_difference = np.sum(np.linalg.norm(self.rclassify_feature_vec_2.vectors[rnd_idxs[:7]] - \
                                               self.rclassify_feature_vec_2.vectors[rnd_idxs[7:]], axis=-1))
        self.assertGreater(rnd_difference, dup_difference)

    def test__prepare_input(self):
        _X, mapping = self.rclassify_feature_vec_1._prepare_input(self.rdata, self.cat_feats)
        self.assertGreater(_X.shape[-1], self.rdata.shape[-1])
        self.assertTrue(all(pd.DataFrame(_X).apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())))


class TestUtils(unittest.TestCase):
    def test_cooccurance_matrix(self):
        rules = ['X00002 <= 11.0 & X00000 > 0.3 & X00002 <= 141.0 & X00001 > 0.4 & X00001 > 0.5 & X00000 <= 2']
        num_features = 5

        cm1 = cooccurance_matrix(rules, num_features, window_size=1, encoded_original_mapping=None)
        self.assertEqual(cm1[2, 0], 2)
        self.assertEqual(cm1[1, 1], 2)
        # Testing relation between different window sizes
        cm2 = cooccurance_matrix(rules, num_features, window_size=2, encoded_original_mapping=None)
        self.assertEqual(cm2[1, 0], 3)
        self.assertTrue(np.all(cm1 <= cm2))

        # Testing multiple rules sum
        rules = ['X00002 <= 11.0 & X00000 > 0.3 & X00002 <= 141.0 & X00001 > 0.4 & X00001 > 0.5 & X00000 <= 2',
                 'X00002 <= 11.0 & X00000 > 0.3 & X00002 <= 141.0 & X00001 > 0.4 & X00001 > 0.5 & X00000 <= 2']
        cm3 = cooccurance_matrix(rules, num_features, window_size=1, encoded_original_mapping=None)
        self.assertTrue(np.all(cm3 == 2 * cm1))
        # Testing feature assignment correctness
        cm4 = cooccurance_matrix(rules, num_features, window_size=1, encoded_original_mapping={0: 0, 1: 2, 2: 3, 3: 1, 4: 4})
        self.assertEqual(cm3[1, 2], cm4[2, 3])
        self.assertEqual(cm3[0, 4], cm4[0, 4])

    def test_cart2pol_pol2cart(self):
        x = [0, 0]
        self.assertEqual(cart2pol(x[0], x[1]), (0, 0))
        x = np.random.random(2)
        x_pol = cart2pol(x[0], x[1])
        x_cart = pol2cart(x_pol[0], x_pol[1])
        self.assertGreater(0.01, np.abs(x_cart[0] - x[0]) / np.abs(x[0]))
        self.assertGreater(0.01, np.abs(x_cart[1] - x[1]) / np.abs(x[1]))
