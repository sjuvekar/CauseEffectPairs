import data_io
import pickle
import feature_extractor as fe
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.grid_search import GridSearchCV
import pandas as pd
import csv
from time import time
import numpy

def get_pipeline():
    features = fe.feature_extractor()
    classifier = GradientBoostingClassifier(n_estimators=1024,
                                          random_state = 1,
                                          subsample = .8,
                                          min_samples_split=10,
                                          max_depth = 6,
                                          verbose=3)
    steps = [("extract_features", features),
             ("classify", classifier)]
    myP = Pipeline(steps)
#    params = {"classify__n_estimators": [768, 1024, 1536], "classify__min_samples_split": [1, 5, 10], "classify__min_samples_leaf": [1, 5, 10]}
#    grid_search = GridSearchCV(myP, params, n_jobs=8)
#    return grid_search
#   return myP
    return (features, classifier)


def get_types(data):
    data['Bin-Bin'] = (data['A type']=='Binary')&(data['B type']=='Binary')
    data['Num-Num'] = (data['A type']=='Numerical')&(data['B type']=='Numerical')
    data['Cat-Cat'] = (data['A type']=='Categorical')&(data['B type']=='Categorical')

    data[['A type','B type']] = data[['A type','B type']].replace('Binary',1)
    data[['A type','B type']] = data[['A type','B type']].replace('Categorical',1)
    data[['A type','B type']] = data[['A type','B type']].replace('Numerical',0)
    return data

def combine_types(data, data_info):
    data = pd.concat([data,data_info],axis = 1)
    types = []
    for a,b in zip(data['A type'], data['B type']):
        types.append(a + b)
    data['types'] = types
    #data['types'] = [x + y for x in data['A type'] for y in data['B type']]
    return data

def main():
    t1 = time()
    print("Reading in the training data")
    train = data_io.read_train_pairs()
    train_info = data_io.read_train_info()
    train = combine_types(train, train_info)

    #make function later
    train = get_types(train)
    target = data_io.read_train_target()

    print "Reading SUP data..."
    for i in range(1,4):
      print "SUP", str(i)
      sup = data_io.read_sup_pairs(i)
      sup_info = data_io.read_sup_info(i)
      sup = combine_types(sup, sup_info)
      sup = get_types(sup)
      sup_target = data_io.read_sup_target(i)
      train = train.append(sup)
      target = target.append(sup_target)

    print "Train size = ", str(train.shape)
    print("Extracting features and training model")
    (feature_trans, classifier) = get_pipeline()
    orig_train = feature_trans.fit_transform(train)
    orig_train = numpy.nan_to_num(orig_train) 

    print("Train-test split")
    trainX, testX, trainY, testY = train_test_split(orig_train, target.Target, random_state = 1)
    print "TrainX size = ", str(trainX.shape)
    print "TestX size = ", str(testX.shape)

    print("Saving features")
    data_io.save_features(orig_train)

    classifier.fit(trainX, trainY)
    print("Saving the classifier")
    data_io.save_model(classifier)
 
    testX = numpy.nan_to_num(testX)
    print "Score on held-out test data ->", classifier.score(testX, testY)
    
    #features = [x[0] for x in classifier.steps[0][1].features ]

    #csv_fea = csv.writer(open('features.csv','wb'))
    #imp = sorted(zip(features, classifier.steps[1][1].feature_importances_), key=lambda tup: tup[1], reverse=True)
    #for fea in imp:
    #    print fea[0], fea[1]
    #    csv_fea.writerow([fea[0],fea[1]])


    feature_importrance = classifier.feature_importances_
    logger = open("feature_importance.csv","a")
    for fi in feature_importrance:
      logger.write(str(fi))
      logger.write("\n")

    t2 = time()
    t_diff = t2 - t1
    print "Time Taken (min):", round(t_diff/60,1)

if __name__ == "__main__":
  main()
