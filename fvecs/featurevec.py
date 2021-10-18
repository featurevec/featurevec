# -*- coding: utf-8 -*-

"""    
This code implements the feature-vectors algorithm.
For a given dataset and a given tree-based model, it extracts a 2-D embedding vector for each feature and visualizes
the interaction among features.

- ``FeatureVec`` implements the feature-vectors algorithm and its output visualization

Copyright 2021
Licensed under the MIT License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://opensource.org/licenses/MIT
"""

import os
import warnings

import pandas as pd
import plotly.express as px
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

from fvecs.fv_rulefit import RuleFit
from fvecs.utils import *

# Constants :
MAX_SENTENCES = 20000  # Default value for maximum rules extracted from the tree
MAX_DEPTH = 3  # Deafult value of maximum (average) depth of trees
EPSILON = 1e-12  # Default value for minimum norm of embeddings (for numerical stability)
AXIS_SIZE = 1.1  # Used for plotting a large enough figure
NUM_SEMICIRCLES = 10  # Number of semi-circles to draw in the final figure


class PlotColumns:
    x = 'x'
    y = 'y'
    angles = 'angels'
    names = 'names'


class FeatureVec(object):
    def __init__(self, mode: str,
                 feature_names: list = None,
                 max_depth: int = MAX_DEPTH,
                 max_sentences: int = MAX_SENTENCES,
                 exp_rand_tree_size: bool = False,
                 tree_generator=None):

        """
        :param mode: 'classify' for classification tasks or 'regress' for regression tasks
        :param max_depth: integer with the maximum (average) depth of the trained trees within the ensemble
        :param feature_names: list with names of the features within the data
        :param max_sentences: integer with the maximum number of extracted sentences
        :param exp_rand_tree_size: If True (default), it allows to have trees with different sizes. Otherwise False.
        :param tree_generator: Tree generator model (overwrites above features), one of the RandomForestClassifier or RandomForestRegressor
        """
        if tree_generator is None:
            assert (mode == 'classify') or (mode == 'regress'), \
                f'Mode should either be "classify" or "regress", but {mode} was given instead'
            assert (max_sentences > 2 ** max_depth), 'Number of sentences should be larger than the tree decision routes.'
            assert isinstance(exp_rand_tree_size, bool), 'exp_rand_tree_size should be True or False.'
            num_trees = max_sentences // (2 ** max_depth)
            if mode == 'classify':
                tree_generator = RandomForestClassifier(n_estimators=num_trees, max_depth=max_depth)
            else:
                tree_generator = RandomForestRegressor(n_estimators=num_trees, max_depth=max_depth)
        else:
            assert isinstance(tree_generator, RandomForestRegressor) or isinstance(tree_generator, RandomForestClassifier), \
                'tree_generator should be a scikit-learn random forest instance.'
            exp_rand_tree_size = False  # We don't want to touch the given tree_generator
        self._feature_names = list(feature_names)
        self._rf = RuleFit(
            tree_generator, max_rules=max_sentences, exp_rand_tree_size=exp_rand_tree_size)

        # initiate attributes:
        self._init_attributes()

    def fit(self, X: np.ndarray, y: np.ndarray, categorical_feats: list = None,
            restart: bool = False, bagging: int = 0, window_size: int = 1):
        """
        Fit the tree model.
        :param categorical_feats: List of the column names that are categorical
        :param X: The input of the dataset to be explained (a numpy array or a Pandas dataframe)
        :param y: outputs (encoded integer class label for classification or real value for regression)
        :param restart: If True, it will train the tree generator model from scratch.
        :param bagging: If >0, it is the number of iterations of baggings performed to extract confidence intervals.
        :param window_size: The neighborhood window size for two features to be considered adjacent.
        """
        assert isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame), \
            'X has to be a numpy array of a DataFrame'
        assert isinstance(y, np.ndarray) or isinstance(y, list), 'y should be a numpy array or a list'
        assert len(np.array(y).shape) == 1, 'y should be 1-d'
        assert (len(X) == len(y)), 'X and y must be the same length'
        assert isinstance(bagging, int), 'bagging must be an integer'
        assert isinstance(restart, bool), 'restart should be True or False'

        if self._feature_names is None:
            if isinstance(X, pd.DataFrame):
                self._feature_names = list(X.columns)
            else:  # If X is a numpy array, name features by index numbers
                log_feats = int(np.ceil(np.log10(X.shape[-1])))
                self._feature_names = [str(x).zfill(log_feats) for x in range(0, X.shape[-1])]
        else:
            assert (len(self._feature_names) == X.shape[-1]), \
                'X must have the same number of features as feature_names'
        X = pd.DataFrame(X, columns=self._feature_names)
        if restart or self._rf.not_trained:
            if categorical_feats is None:
                warnings.warn('categorical_feats is not provided. All non-string features will be treated as numerical.')
                is_numerical = pd.DataFrame(X).apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())
                categorical_feats = X.columns[~is_numerical]
            _X, encoded_original_mapping = self._prepare_input(X, categorical_feats)
        else:
            _X, encoded_original_mapping = X.values, None
        # Re-initiate instance attributes
        self._init_attributes()
        self._rf.fit(_X, np.array(y), restart=restart)
        rules = self._rf.get_rules()
        cm = cooccurance_matrix(rules, X.shape[-1], window_size, encoded_original_mapping)
        dimred = TruncatedSVD(2)  # 2-d dimensionality reduction
        vectors = normalize_angles(dimred.fit_transform(cm))
        norms = np.linalg.norm(vectors, axis=-1)
        vectors /= max(EPSILON, np.max(norms))

        self._vectors = vectors
        self._importance = np.linalg.norm(self._vectors, axis=-1)
        self._stds = np.zeros(self._vectors.shape)

        if bagging:
            all_vectors = []
            for _ in range(bagging):
                self._rf.bag_trees(_X, y)
                rules_bag = self._rf.get_rules()
                cm_bag = cooccurance_matrix(rules_bag, X.shape[-1], window_size, encoded_original_mapping)
                vectors_bag = dimred.fit_transform(cm_bag)
                vectors_bag = normalize_angles(vectors_bag)
                norms_bag = np.linalg.norm(vectors_bag, axis=-1)
                all_vectors.append(vectors_bag / max(EPSILON, np.max(norms_bag)))
            self._stds = np.std(all_vectors, 0)

    @staticmethod
    def _prepare_input(X: np.ndarray, categorical_feats: list):
        """
        Transforms the dataset input into numerical format by encoding categorical variables.
        :param X: a numpy array where each row is a data point
        :param categorical_feats: List of the feature names that are categorical
        :return: (1) numerical_X: Transformed array where categorical variables are one-hot encoded
                (2) feature_assignment: Maps each feature in the transformed array to its original feature
        """
        encoder = OneHotEncoder(drop='if_binary')  # one-hot encoder for categorical variables

        encoded_X = []
        encoded_original_mapping = {}  # Mapping from encoded feature number to original feature number
        encoded_idx = 0

        for feature_idx, feature in enumerate(X):
            column = X[feature].values.reshape((-1, 1))
            if feature in categorical_feats:
                encoded_column = encoder.fit_transform(column).toarray()
            else:  # if the feature is not categorical
                encoded_column = column.astype(float)
            for _ in range(encoded_column.shape[-1]):  # update mapping
                encoded_original_mapping[encoded_idx] = feature_idx
                encoded_idx += 1
            encoded_X.append(encoded_column)
        return np.concatenate(encoded_X,-1), encoded_original_mapping

    @property
    def importance(self):
        return self._importance

    @property
    def vectors(self):
        return self._vectors

    @property
    def confidence_bounds(self):
        if self._stds is None:
            return None
        return 3 * self._stds

    @property
    def angles(self):
        if self._angels is None:
            self._angels = np.arctan2(self._vectors[:, 1], self._vectors[:, 0])
        return self._angels

    def _init_attributes(self):
        """
        Initiate user visible attributes each time fitting a model
        """
        self._angels = None
        self._vectors = None
        self._importance = None
        self._stds = None

    def plot(self, dynamic: bool = True, confidence: bool = True, fpath: str = None,
             font_size: float = 15., marker_size: float = 10.,
             confidence_line_width: float = 1., confidence_line_opacity: float = 0.5,
             confidence_line_color: str = 'gray', confidence_line_style: str = 'solid'):
        """
        Creates and show a plot with the feature-vectors

        :param dynamic: If True (default) the output is a dynamic html plot. Otherwise, it will be an image.
        :param confidence: If True it will show the confidence interval. Otherwise, intervals are not shown.
        :param fpath: Path to save the image. For a dynamic figure, the path should be a .html file.
        :param font_size: font size of the plot texts
        :param marker_size: marker size of the feature vectors
        :param confidence_line_width: width of the feature vector confidence lines
        :param confidence_line_opacity: opacity of the confidence lines
        :param confidence_line_color: color of the confidence lines
        :param confidence_line_style: line style of the confidence lines
        """

        assert self._vectors is not None, "You should first fit the tree-based model."

        angles = np.arctan2(self._vectors[:, 1], self._vectors[:, 0])
        max_angle = np.max(np.abs(angles))
        feature_names = self._feature_names + ['origin', '']
        plot_vectors = np.concatenate([self._vectors, [[0, 0], [0, 0]]])
        plot_angles = np.concatenate([angles, [-max_angle, max_angle]])
        plot_data = np.stack([plot_vectors[:, 1], plot_vectors[:, 0], plot_angles, feature_names], axis=-1)
        plot_df = pd.DataFrame(
            data=plot_data,
            columns=[PlotColumns.x, PlotColumns.y, PlotColumns.angles, PlotColumns.names]
        )
        plot_df[[PlotColumns.x, PlotColumns.y, PlotColumns.angles]] = \
            plot_df[[PlotColumns.x, PlotColumns.y, PlotColumns.angles]].apply(pd.to_numeric)

        fig = px.scatter(
            plot_df, x=PlotColumns.x, y=PlotColumns.y, color=PlotColumns.angles, width=1000, height=500,
            hover_name=feature_names,
            hover_data={PlotColumns.x: False, PlotColumns.y: False, PlotColumns.angles: False, PlotColumns.names: False},
            color_continuous_scale=px.colors.sequential.Rainbow)

        fig.update_yaxes(visible=False, showticklabels=False, range=[0, AXIS_SIZE])
        fig.update_xaxes(visible=False, showticklabels=False, range=[-AXIS_SIZE, AXIS_SIZE])
        if not dynamic:
            for i in range(len(plot_vectors) - 2):
                fig.add_annotation(
                    x=plot_vectors[:, 1][i],
                    y=plot_vectors[:, 0][i],
                    text=feature_names[i],
                    font=dict(size=15),
                    axref=PlotColumns.x,
                    ayref=PlotColumns.y,
                    ax=plot_vectors[:, 1][i],
                    ay=plot_vectors[:, 0][i],
                    arrowhead=2,
                )
        fig.update_traces(marker=dict(size=marker_size), textfont_size=font_size)
        fig.update(layout_coloraxis_showscale=False)
        fig.update_layout(showlegend=False)
        for i in range(NUM_SEMICIRCLES):  # Draws semi-circles with same origins for better visualization of importance
            fig.add_shape(
                type='circle',
                x0=(i + 1) / 10 * AXIS_SIZE,
                y0=(i + 1) / 10 * AXIS_SIZE,
                x1=-(i + 1) / 10 * AXIS_SIZE,
                y1=-(i + 1) / 10 * AXIS_SIZE,
                line_color="red", opacity=0.5, line=dict(dash='dot', width=3))
        if confidence:
            for vector, std, angle in zip(self._vectors, self._stds, angles):
                fig.add_shape(
                    type='circle',
                    x0=vector[1] + 3 * std[1],
                    y0=vector[0] + 3 * std[0],
                    x1=vector[1] - 3 * std[1],
                    y1=vector[0] - 3 * std[0],
                    line_color=confidence_line_color,
                    opacity=confidence_line_opacity,
                    line=dict(dash=confidence_line_style, width=confidence_line_width))
        fig.show()

        if fpath:
            assert os.path.exists(os.path.split(fpath)[0]), 'The folder containing the save path does not exist!'
            if os.path.exists(fpath):
                warnings.warn('Figure already exists. Overwriting!')
            if dynamic:                
                pre, ext = os.path.splitext(fpath)
                if ext != '.html':
                    print('For a dynamic figure, path extension should be an html file > changing suffix to .html')
                    fpath = pre + '.html'
                fig.write_html(fpath)
            else:
                fig.write_image(fpath)
