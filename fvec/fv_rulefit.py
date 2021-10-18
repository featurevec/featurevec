"""
Most of the code is based on https://github.com/christophM/rulefit/blob/master/rulefit/rulefit.py
Linear model of tree-based decision rules
This method partly implements the RuleFit algorithm
The module structure is the following:
- ``RuleCondition`` implements a binary feature transformation
- ``Rule`` implements a Rule composed of ``RuleConditions``
- ``RuleEnsemble`` implements an ensemble of ``Rules``
- ``RuleFit`` implements the RuleFit algorithm
"""
from functools import reduce

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils.validation import check_is_fitted, NotFittedError


class RuleCondition:
    """Class for binary rule condition
    Warning: this class should not be used directly.
    """

    def __init__(self,
                 feature_index,
                 threshold,
                 operator,
                 support,
                 feature_name=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.operator = operator
        self.support = support
        self.feature_name = feature_name

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.feature_name:
            feature = self.feature_name
        else:
            feature = self.feature_index
        return "%s %s %s" % (feature, self.operator, self.threshold)

    def transform(self, X):
        """Transform dataset.
        Parameters
        ----------
        X: array-like matrix, shape=(n_samples, n_features)
        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, 1)
        """
        if self.operator == "<=":
            res = 1 * (X[:, self.feature_index] <= self.threshold)
        elif self.operator == ">":
            res = 1 * (X[:, self.feature_index] > self.threshold)
        return res

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash((self.feature_index, self.threshold, self.operator, self.feature_name))


class Winsorizer:
    """Performs Winsorization 1->1*
    Warning: this class should not be used directly.
    """

    def __init__(self, trim_quantile=0.0):
        self.trim_quantile = trim_quantile
        self.winsor_lims = None

    def train(self, X):
        # get winsor limits
        self.winsor_lims = np.full((2, X.shape[1]), fill_value=np.inf)  # small efficiency modification
        self.winsor_lims[0, :] = -np.inf
        if self.trim_quantile > 0:
            self.winsor_lims = np.percentile(X, axis=0, q=(self.trim_quantile * 100, 100 - self.trim_quantile * 100))  # small efficiency modification

    def trim(self, X):
        # small efficiency modification
        X_ = np.where(X > self.winsor_lims[1, :], np.tile(self.winsor_lims[1, :], [X.shape[0], 1]),
                      np.where(X < self.winsor_lims[0, :], np.tile(self.winsor_lims[0, :], [X.shape[0], 1]), X))
        return X_


class FriedScale:
    """Performs scaling of linear variables according to Friedman et al. 2005 Sec 5
    Each variable is first Winsorized l->l*, then standardised as 0.4 x l* / std(l*)
    Warning: this class should not be used directly.
    """

    def __init__(self, winsorizer=None):
        self.scale_multipliers = None
        self.winsorizer = winsorizer

    def train(self, X):
        # get multipliers
        if self.winsorizer is not None:
            X_trimmed = self.winsorizer.trim(X)
        else:
            X_trimmed = X

        scale_multipliers = np.ones(X.shape[1])
        for i_col in range(X.shape[1]):  # small efficiency modification
            num_uniq_vals = len(np.unique(X[:, i_col]))
            if num_uniq_vals > 2:  # don't scale binary variables which are effectively already rules
                scale_multipliers[i_col] = 0.4 / (1.0e-12 + np.std(X_trimmed[:, i_col]))
        self.scale_multipliers = scale_multipliers

    def scale(self, X):
        if self.winsorizer is not None:
            return self.winsorizer.trim(X) * self.scale_multipliers
        else:
            return X * self.scale_multipliers


class Rule():
    """Class for binary Rules from list of conditions
    Warning: this class should not be used directly.
    """

    def __init__(self,
                 rule_conditions, prediction_value):
        self.conditions = set(rule_conditions)
        self.support = min([x.support for x in rule_conditions])
        self.prediction_value = prediction_value
        self.rule_direction = None

    def transform(self, X):
        """Transform dataset.
        Parameters
        ----------
        X: array-like matrix
        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, 1)
        """
        rule_applies = [condition.transform(X) for condition in self.conditions]
        return reduce(lambda x, y: x * y, rule_applies)

    def __str__(self):
        return " & ".join([x.__str__() for x in self.conditions])

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return sum([condition.__hash__() for condition in self.conditions])

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


def extract_rules_from_tree(tree, feature_names=None):
    """Helper to turn a tree into as set of rules
    """
    rules = set()

    def traverse_nodes(node_id=0,
                       operator=None,
                       threshold=None,
                       feature=None,
                       conditions=[]):
        if node_id != 0:
            if feature_names is not None:
                feature_name = feature_names[feature]
            else:
                feature_name = feature
            rule_condition = RuleCondition(feature_index=feature,
                                           threshold=threshold,
                                           operator=operator,
                                           support=tree.n_node_samples[node_id] / float(tree.n_node_samples[0]),
                                           feature_name=feature_name)
            new_conditions = conditions + [rule_condition]
        else:
            new_conditions = []

        # if not terminal node
        if tree.children_left[node_id] != tree.children_right[node_id]:
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]

            left_node_id = tree.children_left[node_id]
            traverse_nodes(left_node_id, "<=", threshold, feature, new_conditions)

            right_node_id = tree.children_right[node_id]
            traverse_nodes(right_node_id, ">", threshold, feature, new_conditions)
        else:  # a leaf node
            if len(new_conditions) > 0:
                new_rule = Rule(new_conditions, tree.value[node_id][0][0])
                rules.update([new_rule])
            else:
                pass  # tree only has a root node!
            return None

    traverse_nodes()

    return rules


class RuleEnsemble():
    """Ensemble of binary decision rules
    This class implements an ensemble of decision rules that extracts rules from
    an ensemble of decision trees.
    Parameters
    ----------
    tree_list: List or array of DecisionTreeClassifier or DecisionTreeRegressor
        Trees from which the rules are created
    feature_names: List of strings, optional (default=None)
        Names of the features
    Attributes
    ----------
    rules: List of Rule
        The ensemble of rules extracted from the trees
    """

    def __init__(self,
                 tree_list,
                 feature_names=None):
        self.tree_list = tree_list
        self.feature_names = feature_names
        self.rules = set()
        ## TODO: Move this out of __init__
        self._extract_rules()
        self.rules = list(self.rules)

    def _extract_rules(self):
        """Recursively extract rules from each tree in the ensemble
        """
        for tree in self.tree_list:
            rules = extract_rules_from_tree(tree[0].tree_, feature_names=self.feature_names)
            self.rules.update(rules)

    def filter_rules(self, func):
        self.rules = filter(lambda x: func(x), self.rules)

    def filter_short_rules(self, k):
        self.filter_rules(lambda x: len(x.conditions) > k)

    def transform(self, X, coefs=None):
        """Transform dataset.
        Parameters
        ----------
        X:      array-like matrix, shape=(n_samples, n_features)
        coefs:  (optional) if supplied, this makes the prediction
                slightly more efficient by setting rules with zero
                coefficients to zero without calling Rule.transform().
        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, n_out)
            Transformed dataset. Each column represents one rule.
        """
        rule_list = list(self.rules)
        if coefs is None:
            res = []
            for i_rule in np.arange(len(rule_list)):
                res.append(rule_list[i_rule].transform(X))
            return np.array(res).T
        else:  # else use the coefs to filter the rules we bother to interpret
            res = []
            for i_rule in np.arange(len(rule_list)):
                if coefs[i_rule] != 0:
                    res.append(rule_list[i_rule].transform(X))
            res = np.array(res).T
            res_ = np.zeros([X.shape[0], len(rule_list)])
            res_[:, coefs != 0] = res
            return res_

    def __str__(self):
        return (map(lambda x: x.__str__(), self.rules)).__str__()


class RuleFit(BaseEstimator, TransformerMixin):
    # TODO: CR @Amirata by Yuval: Change the tree_generator description to "... Must be RandomForestClassifier or RandomForestRegressor ..." to align the code implementation
    """Rulefit class
    Parameters
    ----------
        tree_generator: Optional: this object will be used as provided to generate the rules.
                        This will override almost all the other properties above.
                        Must be RandomForestClassifier or RandomForestRegressor, optional (default=None)
        rfmode:         'regress' for regression or 'classify' for binary classification.
        tree_size:      Number of terminal nodes in generated trees. If exp_rand_tree_size=True,
                        this will be the mean number of terminal nodes.
        sample_fract:   fraction of randomly chosen training observations used to produce each tree.
                        FP 2004 (Sec. 2)
        max_rules:      approximate total number of rules generated for fitting. Note that actual
                        number of rules will usually be lower than this due to duplicates.
        exp_rand_tree_size: If True, each boosted tree will have a different maximum number of
                        terminal nodes based on an exponential distribution about tree_size.
                        (Friedman Sec 3.3)
        random_state:   Integer to initialise random objects and provide repeatability.
    Attributes
    ----------
    rule_ensemble: RuleEnsemble
        The rule ensemble
    feature_names: list of strings, optional (default=None)
        The names of the features (columns)
    """

    def __init__(
            self,
            tree_generator,
            max_rules=2000,
            exp_rand_tree_size=True,
            random_state=None):
        
        self.tree_generator = tree_generator
        if isinstance(tree_generator, RandomForestClassifier):
            self.rfmode = 'classify'
        elif isinstance(tree_generator, RandomForestRegressor):
            self.rfmode = 'regress'
        else:
            raise ValueError('Feature Vectors only supports scikit-learn Random Forest models.')
        
        try:
            check_is_fitted(self.tree_generator)
            self.not_trained = False
        except NotFittedError as e:
            self.not_trained = True
            
        self.tree_size = 2 ** tree_generator.max_depth
        self.exp_rand_tree_size = exp_rand_tree_size
        self.max_rules = max_rules
        self.random_state = random_state
    
    def fit(self, X, y=None, feature_names=None, restart=True):
        """Fit and estimate linear combination of rule ensemble
        """
        ## Enumerate features if feature names not provided
        N = X.shape[0]
        if feature_names is None:
            self.feature_names = ['X' + str(x).zfill(5) for x in range(0, X.shape[1])]
        else:
            self.feature_names = feature_names
        ## initialise tree generator
        if restart:
            self.not_trained = True
            self.tree_generator = clone(self.tree_generator)
        ## fit tree generator if not already fit
        if self.not_trained:
            if self.exp_rand_tree_size:  # randomise tree size as per Friedman 2005 Sec 3.3
                np.random.seed(self.random_state)
                tree_sizes = np.random.exponential(
                    scale=self.tree_size - 2, size=int(np.ceil(self.max_rules * 2 / self.tree_size)))
                tree_sizes = np.asarray([2 + np.floor(tree_sizes[i_]) for i_ in np.arange(len(tree_sizes))], dtype=int)
                i = int(len(tree_sizes) / 4)
                tree_sizes = tree_sizes[0: np.where(np.cumsum(tree_sizes) < self.max_rules)[0][-1] + 2]
                self.tree_generator.set_params(warm_start=True)
                curr_est_ = 0
                for i_size in np.arange(len(tree_sizes)):
                    size = tree_sizes[i_size]
                    self.tree_generator.set_params(n_estimators=curr_est_ + 1)
                    self.tree_generator.set_params(max_leaf_nodes=size)
                    random_state_add = self.random_state if self.random_state else 0
                    self.tree_generator.set_params(
                        random_state=i_size + random_state_add)  # warm_state=True seems to reset random_state, such that the trees are highly correlated, unless we manually change the random_sate here.
                    self.tree_generator.get_params()['n_estimators']
                    self.tree_generator.fit(np.copy(X, order='C'), np.copy(y, order='C'))
                    curr_est_ = curr_est_ + 1
                self.tree_generator.set_params(warm_start=False)
            else:
                self.tree_generator.fit(X, y)
            self.not_trained = False
        tree_list = self.tree_generator.estimators_
        self._extract_rules(tree_list)
        return self

    def bag_trees(self, X, y):
        """Applies bagging to the list of trees to extract jittered versions of feature vectors.
        """
        tree_list = self.tree_generator.estimators_
        bag_idxs = np.random.choice(len(tree_list), len(tree_list))
        bagged_tree_list = []
        for bag_idx in bag_idxs:
            bagged_tree_list.append(tree_list[bag_idx])
        self._extract_rules(bagged_tree_list)
        return self

    def _extract_rules(self, tree_list):
        """Given a tree-based model, extracts decision rules.
        """
        tree_list = [[x] for x in tree_list]
        ## extract rules
        self.rule_ensemble = RuleEnsemble(
            tree_list=tree_list, feature_names=self.feature_names)
        ## concatenate original features and rules
#         X_concat = self.rule_ensemble.transform(X)
#         self.X_concat = X_concat
#         self.n_features = X.shape[-1]
        return self

    def predict(self, X):
        """Predict outcome for X
        """
        X_concat = np.zeros([X.shape[0], 0])
        rule_coefs = self.coef_[-len(self.rule_ensemble.rules):]
        if len(rule_coefs) > 0:
            X_rules = self.rule_ensemble.transform(X, coefs=rule_coefs)
            if X_rules.shape[0] > 0:
                X_concat = np.concatenate((X_concat, X_rules), axis=1)
        return self.lscv.predict(X_concat)

    def predict_proba(self, X):
        """Predict outcome probability for X, if model type supports probability prediction method
        """

        if 'predict_proba' not in dir(self.lscv):
            error_message = '''
            Probability prediction using predict_proba not available for
            model type {lscv}
            '''.format(lscv=self.lscv)
            raise ValueError(error_message)

        X_concat = np.zeros([X.shape[0], 0])
        rule_coefs = self.coef_[-len(self.rule_ensemble.rules):]
        if len(rule_coefs) > 0:
            X_rules = self.rule_ensemble.transform(X, coefs=rule_coefs)
            if X_rules.shape[0] > 0:
                X_concat = np.concatenate((X_concat, X_rules), axis=1)
        return self.lscv.predict_proba(X_concat)

    def transform(self, X=None, y=None):
        """Transform dataset.
        Parameters
        ----------
        X : array-like matrix, shape=(n_samples, n_features)
            Input data to be transformed. Use ``dtype=np.float32`` for maximum
            efficiency.
        Returns
        -------
        X_transformed: matrix, shape=(n_samples, n_out)
            Transformed data set
        """
        return self.rule_ensemble.transform(X)

    def get_rules(self):
        # TODO: CR @Amirata by Yuval: adjust documentation (no input vars, the output is not a pd.DataFrame)
        """Return the estimated rules
        Parameters
        ----------
        exclude_zero_coef: If True (default), returns only the rules with an estimated
                           coefficient not equalt to  zero.
        subregion: If None (default) returns global importances (FP 2004 eq. 28/29), else returns importance over
                           subregion of inputs (FP 2004 eq. 30/31/32).
        Returns
        -------
        rules: List of rules 
        """

        rule_ensemble = list(self.rule_ensemble.rules)
        return [(rule.__str__()) for rule in rule_ensemble]
