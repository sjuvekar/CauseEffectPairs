import features as f
import transformed_features as tf
import gaussian_process as gp
import independent_component as indcomp
import numpy as np
import scipy

def feature_extractor():
    features = [('Number of Samples', 'A', f.SimpleTransform(transformer=len)),
                ('A: Number of Unique Samples', 'A', f.SimpleTransform(transformer=f.count_unique)),
                ('B: Number of Unique Samples', 'B', f.SimpleTransform(transformer=f.count_unique)),
                ('A: Mean', 'A', f.SimpleTransform(transformer=np.mean)),
                ('B: Mean', 'B', f.SimpleTransform(transformer=np.mean)),
                ('A: Max', 'A', f.SimpleTransform(transformer=np.max)),
                ('B: Max', 'B', f.SimpleTransform(transformer=np.max)),
                ('A: Min', 'A', f.SimpleTransform(transformer=np.min)),
                ('B: Min', 'B', f.SimpleTransform(transformer=np.min)),
                ('A: Range', 'A', f.SimpleTransform(transformer=f.rng)),
                ('B: Range', 'B', f.SimpleTransform(transformer=f.rng)),
                ('A: Median', 'A', f.SimpleTransform(transformer=f.median)),
                ('B: Media', 'B', f.SimpleTransform(transformer=f.median)),
                ('A: Percentile 25', 'A', f.SimpleTransform(transformer=f.percentile25)),
                ('B: Percentile 25', 'B', f.SimpleTransform(transformer=f.percentile25)),
                ('A: Percentile 75', 'A', f.SimpleTransform(transformer=f.percentile75)),
                ('B: Percentile 75', 'B', f.SimpleTransform(transformer=f.percentile75)),
                ('A: Bollinger', 'A', f.SimpleTransform(transformer=f.bollinger)),
                ('B: Bollinger', 'B', f.SimpleTransform(transformer=f.bollinger)),
                ('A: Std', 'A', f.SimpleTransform(transformer=np.std)),
                ('B: Std', 'B', f.SimpleTransform(transformer=np.std)),
                ('A: Variation', 'A', f.SimpleTransform(transformer=scipy.stats.variation)),
                ('B: Variation', 'B', f.SimpleTransform(transformer=scipy.stats.variation)),
                ('A: Skew', 'A', f.SimpleTransform(transformer=scipy.stats.skew)),
                ('B: Skew', 'B', f.SimpleTransform(transformer=scipy.stats.skew)),
                ('A: Kurtosis', 'A', f.SimpleTransform(transformer=scipy.stats.kurtosis)),
                ('B: Kurtosis', 'B', f.SimpleTransform(transformer=scipy.stats.kurtosis)),
                ('A: Normalized Entropy', 'A', f.SimpleTransform(transformer=f.normalized_entropy)),
                ('B: Normalized Entropy', 'B', f.SimpleTransform(transformer=f.normalized_entropy)),
                ('linregress', ['A','B'], f.MultiColumnTransform(f.linregress)),
                ('linregress_rev', ['B','A'], f.MultiColumnTransform(f.linregress)),
                ('complex_regress', ['A', 'B'], f.MultiColumnTransform(tf.complex_regress)),
                ('complex_regress rev', ['B', 'A'], f.MultiColumnTransform(tf.complex_regress)),
                ('gaussian process', ['A', 'B'], f.MultiColumnTransform(gp.gaussian_fit_likelihood)),
                ('Independent Components', ['A', 'B'], f.MultiColumnTransform(indcomp.independent_component)),
                ('ttest_ind_t', ['A','B'], f.MultiColumnTransform(f.ttest_ind_t)),
                ('ttest_ind_p', ['A','B'], f.MultiColumnTransform(f.ttest_ind_p)),
                ('ks_2samp', ['A','B'], f.MultiColumnTransform(f.ks_2samp)),
                ('kruskal', ['A','B'], f.MultiColumnTransform(f.kruskal)),
                ('bartlett', ['A', 'B'], f.MultiColumnTransform(f.bartlett)),
                ('levene', ['A','B'], f.MultiColumnTransform(f.levene)),
                ('A: shapiro', 'A', f.SimpleTransform(transformer=f.shapiro)),
                ('B: shapiro', 'B', f.SimpleTransform(transformer=f.shapiro)),
                ('fligner', ['A','B'], f.MultiColumnTransform(f.fligner)),
                ('mood', ['A','B'], f.MultiColumnTransform(f.mood)),
                ('oneway', ['A','B'], f.MultiColumnTransform(f.oneway)),
                ('Pearson R', ['A','B'], f.MultiColumnTransform(f.correlation)),
                ('Pearson R Magnitude', ['A','B'], f.MultiColumnTransform(f.correlation_magnitude)),
                ('Entropy Difference', ['A','B'], f.MultiColumnTransform(f.entropy_difference)),
                ('Mutual Information', ['A', 'B'], f.MultiColumnTransform(tf.mutual_information))]

    combined = f.FeatureMapper(features)
    return combined


