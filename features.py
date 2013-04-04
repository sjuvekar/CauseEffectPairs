import numpy as np
from sklearn.base import BaseEstimator
from scipy.special import psi
from scipy.stats.stats import pearsonr
from scipy import stats
import pickle

class FeatureMapper:
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        for feature_name, column_names, extractor in self.features:
            extractor.fit(X[column_names], y)

    def transform(self, X):
        extracted = []
        for feature_name, column_names, extractor in self.features:
            fea = extractor.transform(X[column_names])
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]

    def fit_transform(self, X, y=None):
        extracted = []
        for feature_name, column_names, extractor in self.features:
            fea = extractor.fit_transform(X[column_names], y)
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            with open("XXX", "wb") as output:
                pickle.dump(extracted, output)
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]

def identity(x):
    return x

def rng(x):
    return (np.max(x) - np.min(x))

def bollinger(x):
    m = np.mean(x)
    mx = np.max(x)
    mn = np.min(x)
    if np.allclose(mx, mn):
      return (mx - m)
    else:
      return ( (mx - m)/(mx - mn) )

def count_unique(x):
    return len(set(x))

def normalized_entropy(x):
    x = (x - np.mean(x)) / np.std(x)
    x = np.sort(x)
    
    hx = 0.0;
    for i in range(len(x)-1):
        delta = x[i+1] - x[i];
        if delta != 0:
            hx += np.log(np.abs(delta));
    hx = hx / (len(x) - 1) + psi(len(x)) - psi(1);

    return hx

def entropy_difference(x, y):
    return normalized_entropy(x) - normalized_entropy(y)

def correlation(x, y):
    return pearsonr(x, y)[0]

def correlation_magnitude(x, y):
    return abs(correlation(x, y))

def linregress(x, y):
    return stats.linregress(x, y)

def ttest_ind_t(x, y):
    return float(stats.ttest_ind(x, y)[0])

def ttest_ind_p(x, y):
    return stats.ttest_ind(x, y)[1]

def ttest_rel_t(x, y):
    return float(stats.ttest_rel(x, y)[0])

def ttest_rel_p(x, y):
    return stats.ttest_rel(x, y)[1]

def ks_2samp(x, y):
    return stats.ks_2samp(x, y)

def kruskal(x, y):
    return stats.kruskal(x, y)

def bartlett(x, y):
    return stats.bartlett(x, y)

def levene(x, y):
    return stats.levene(x, y)

def shapiro(x):
    return stats.shapiro(x)

def fligner(x, y):
    return stats.fligner(x, y)

def mood(x, y):
    return stats.mood(x, y)

def oneway(x, y):
    return stats.oneway(x, y)

class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return_value = np.array([self.transformer(x) for x in X], ndmin=2).T
        if return_value.shape[1] == 1:
            return return_value
        else:
            return return_value.T

class MultiColumnTransform(BaseEstimator):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return_value = np.array([self.transformer(*x[1]) for x in X.iterrows()], ndmin=2).T
        if return_value.shape[1] == 1:
            return return_value
        else:
            return return_value.T
