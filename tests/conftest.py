import uuid
import numpy as np
import pandas as pd


def init_biased_random_data(max_unique, p_bias):
    """
    Creates random data for testing which has 10 attributes, 3 of them protected (Sex, Gender, Race)
    The Sex attribute is unbalanced (p males, 1-p females)
    The Gender attribute is balanced (50-50)
    - There is data bias for Sex
    - There is model bias for Race
    :param max_unique:
    :param p_bias:
    :return:
    """
    np.random.seed(42)
    n_samples = 10000
    string_option = [str(uuid.uuid1())[2:10] for _ in range(max_unique + 3)]  # some parts of the uuid are almost constant

    data = {
        'Low_Corr_Bias_Cont': np.random.normal(0, scale=10, size=n_samples),  # Continuous (with negative values)
        'Low_Corr_Bias_CatFloats': np.random.choice(np.linspace(0., 1., 50), size=n_samples),
        'Low_Corr_Bias_OrdinalNatural': np.random.randint(0, 2 * n_samples, size=n_samples),  # ordinal(natural numbers)
        'Low_Corr_bias_CatInts': np.random.randint(-3, 3, size=n_samples),  # Categorical int (6 different values)
        'Low_Corr_Bias_String': np.random.choice(string_option, size=n_samples),  # string (self.max_unique + 3 different options)
        'Sex': np.random.choice(['Male', 'Female'], size=n_samples, p=[p_bias, 1 - p_bias]),  # 2 string options, UNBALANCED
        'Gender': np.random.choice(['Male', 'Female'], size=n_samples, p=[.5, .5]),  # 2 string options, equal
    }
    x = pd.DataFrame(data=data)

    y = (x['Sex'].values == 'Male').astype(int)  # add data bias for SEX
    male_label_inds = y[y == 1]
    female_label_inds = y[y == 0]
    y[np.random.randint(0, len(male_label_inds), size=int(0.5 * len(male_label_inds)))] = 0
    y[np.random.randint(0, len(female_label_inds), size=int(0.5 * len(female_label_inds)))] = 1

    y_hat = np.random.randint(0, 2, size=n_samples) # random predictions

    categorical_features = ['Low_Corr_bias_CatInts', 'Low_Corr_Bias_String', 'Sex', 'Gender']
    numerical_features = ['Low_Corr_Bias_Cont', 'Low_Corr_Bias_CatFloats', 'Low_Corr_Bias_OrdinalNatural']
    return x, y, y_hat, categorical_features, numerical_features


def init_biased_duplicate_data(max_unique, p_bias):
    x, y, y_hat, categorical_features, numerical_features = init_biased_random_data(max_unique, p_bias)
    new_x = pd.DataFrame(np.concatenate([x.values] * 2, -1),
                         columns=list(x.columns) + [i+'_dup' for i in x.columns])
    new_cat_feats = categorical_features + [i+'_dup' for i in categorical_features]
    new_num_feats = numerical_features + [i+'_dup' for i in numerical_features]
    return new_x, y, y_hat, new_cat_feats, new_num_feats
