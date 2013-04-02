import numpy as np
from sklearn.base import BaseEstimator
from scipy.special import psi
from scipy.stats.stats import pearsonr
from scipy.stats import linregress, ttest_ind, ttest_rel, ks_2samp, kruskal

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
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]

def identity(x):
    return x

def rng(x):
    return (np.max(x) - np.min(x))

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

def linregress_slope(x, y):
    return linregress(x, y)[0]

def linregress_int(x, y):
    return linregress(x, y)[1]

def linregress_r(x, y):
    return linregress(x, y)[2]

def linregress_p(x, y):
    return linregress(x, y)[3]

def linregress_std(x, y):
    return linregress(x, y)[4]

def ttest_ind_t(x, y):
    return float(ttest_ind(x, y)[0])

def ttest_ind_p(x, y):
    return ttest_ind(x, y)[1]

def ttest_rel_t(x, y):
    return float(ttest_rel(x, y)[0])

def ttest_rel_p(x, y):
    return ttest_rel(x, y)[1]

def ks_2samp_D(x, y):
    return ks_2samp(x, y)[0]

def ks_2samp_p(x, y):
    return ks_2samp(x, y)[1]

def kruskal_H(x, y):
    return kruskal(x, y)[0]

def kruskal_p(x, y):
    return kruskal(x, y)[1]

class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(x) for x in X], ndmin=2).T

class MultiColumnTransform(BaseEstimator):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(*x[1]) for x in X.iterrows()], ndmin=2).T
