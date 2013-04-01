import data_io
import features as f
import numpy as np
import scipy
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

def feature_extractor():
    features = [('Number of Samples', 'A', f.SimpleTransform(transformer=len)),
                ('A: Number of Unique Samples', 'A', f.SimpleTransform(transformer=f.count_unique)),
                ('B: Number of Unique Samples', 'B', f.SimpleTransform(transformer=f.count_unique)),
                ('A: Mean', 'A', f.SimpleTransform(transformer=np.mean)),
                ('B: Mean', 'B', f.SimpleTransform(transformer=np.mean)),
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
                ('linregress slope', ['A','B'], f.MultiColumnTransform(f.linregress_slope)),
                ('linregress int', ['A','B'], f.MultiColumnTransform(f.linregress_int)),
                ('linregress r', ['A','B'], f.MultiColumnTransform(f.linregress_r)),
                ('linregress p', ['A','B'], f.MultiColumnTransform(f.linregress_p)),
                ('linregress std', ['A','B'], f.MultiColumnTransform(f.linregress_std)),
                ('ttest_ind_t', ['A','B'], f.MultiColumnTransform(f.ttest_ind_t)),
                ('ttest_ind_p', ['A','B'], f.MultiColumnTransform(f.ttest_ind_p)),
                #('ttest_rel_t', ['A','B'], f.MultiColumnTransform(f.ttest_rel_t)),
                #('ttest_rel_p', ['A','B'], f.MultiColumnTransform(f.ttest_rel_p)),
                ('ks_2samp_D', ['A','B'], f.MultiColumnTransform(f.ks_2samp_D)),
                ('ks_2samp_p', ['A','B'], f.MultiColumnTransform(f.ks_2samp_p)),
                ('kruskal_H', ['A','B'], f.MultiColumnTransform(f.kruskal_H)),
                ('kruskal_p', ['A','B'], f.MultiColumnTransform(f.kruskal_p)),
                ('Pearson R', ['A','B'], f.MultiColumnTransform(f.correlation)),
                ('Pearson R Magnitude', ['A','B'], f.MultiColumnTransform(f.correlation_magnitude)),
                ('Entropy Difference', ['A','B'], f.MultiColumnTransform(f.entropy_difference))]
    combined = f.FeatureMapper(features)
    return combined

def get_pipeline():
    features = feature_extractor()
    steps = [("extract_features", features),
             ("classify", RandomForestRegressor(n_estimators=1024, 
                                                verbose=2,
                                                n_jobs=1,
                                                min_samples_split=10,
                                                random_state=1))]
    return Pipeline(steps)

def main():
    print("Reading in the training data")
    train = data_io.read_train_pairs()
    target = data_io.read_train_target()

    print("Extracting features and training model")
    classifier = get_pipeline()
    classifier.fit(train, target.Target)

    print("Saving the classifier")
    data_io.save_model(classifier)
    
if __name__=="__main__":
    main()
