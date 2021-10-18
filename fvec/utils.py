"""This file contains helper functions for the Feature Vectors method."""

import numpy as np
import re
from typing import List

SEARCH_PATTERN = 'X\d\d\d\d\d'  # Encoded features' names are represented with this pattern e.g. second feature would be X00001


# --------------------- FeatureVec utils -----------------------------------------------
def cooccurance_matrix(rules: List[str], num_features: int, window_size: int = 1, encoded_original_mapping: dict = None) -> np.ndarray:
    """
    Computes the co-occurrence matrix of features in the list of rules extracted by the tree-based model.
    :param rules: The list of all the rules extracted from the tree-based model
    :param num_features: Number of original features in the dataset
    :param window_size: The size of the co-occurance window
    :param encoded_original_mapping: If not None, determines which original feature a given encoded feature is assigned to
    (to handle categorical variables)
    :return: numpy array the co-occurance matrix of features in the dataset
    """
    if encoded_original_mapping is None:
        encoded_original_mapping = {i: i for i in range(num_features)}
    cooccurrence = np.zeros((num_features, num_features))
    for rule in rules:
        # Features appear as hard-coded Xddddd patterns in the rules:
        rule_encoded_features = [int(f[1:]) for f in re.findall(SEARCH_PATTERN, rule)]
        group = [encoded_original_mapping[encoded_feature] for encoded_feature in rule_encoded_features]
        if len(group) < (2 * window_size + 1):
            continue
        for i, feature_idx in enumerate(group):
            context_features = group[max(i - window_size, 0): i] + group[i + 1: (i + 1 + window_size)]
            # using a loop instead of a numpy assignment since we might have non unique indices which numpy will change to uniques
            # thus causing a bug:
            for context_feature_idx in context_features:
                cooccurrence[feature_idx, context_feature_idx] += 1
    return cooccurrence


def cart2pol(x, y) -> (np.float, np.float):
    """
    Transforms 2-d points from Cartesian coordinates to polar
    :param x: x component of the 2-d vector
    :param y: y component of the 2-d vector
    :return: polar representation of 2-d vector
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi) -> (np.float, np.float):
    """
    Transforms 2-d points from polar coordinates to Cartesian
    :param rho: radius coordinate of the 2-d vector 
    :param phi: angle coordinate of the 2-d vector
    :return: Cartesian coordinates of the 2-d vector
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def normalize_angles(vectors) -> np.ndarray:
    """
    Normalizes angles such that the mean of the angles are zero-mean.
    :param vectors: a numpy array where rows are the vectors
    :return: rotates vectors s.t. the average angle coordinate is zero
    """
    rho, phi = cart2pol(vectors[:, 0], vectors[:, 1])
    phi -= np.mean(phi)
    x, y = pol2cart(rho, phi)
    return np.stack([x, y], -1)
