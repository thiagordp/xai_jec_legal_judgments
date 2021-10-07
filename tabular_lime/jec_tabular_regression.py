"""
Prediction compensations for JEC judgments using tabular attributes and explaining using LIME
"""

# Load train and test sets
import logging

import lime
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


def load_data():
    x_train = pd.read_csv("data/jec/jec_attributes_proc_train.csv").values
    x_test = pd.read_csv("data/jec/jec_attributes_proc_test.csv").values
    y_test = pd.read_csv("data/jec/jec_attributes_test_labels.csv")["valor"].values
    y_train = pd.read_csv("data/jec/jec_attributes_train_labels.csv")["valor"].values
    features = pd.read_csv("data/jec/jec_attributes_proc_train.csv").columns

    return x_train, x_test, y_train, y_test, features


def jec_lime_regression():
    x_train, x_test, y_train, y_test, features = load_data()

    # Fit model
    # rf = RandomForestRegressor(n_estimators=500, random_state=42)
    rf = VotingRegressor(n_jobs=4, verbose=True, estimators=[
        ("mlp", MLPRegressor(hidden_layer_sizes=(256, 256, 256, 256, 256,),
                             max_iter=50,
                             early_stopping=True,
                             shuffle=True,
                             activation="relu",
                             batch_size=16, verbose=True)),
        ('bagging', BaggingRegressor(n_estimators=50, n_jobs=8)),
        ('xgb', XGBRegressor(n_estimators=50, max_depth=10, n_jobs=8)),
        ('gd', GradientBoostingRegressor(max_depth=10, max_leaf_nodes=100))
    ])
    # rf = MLPRegressor(
    #     hidden_layer_sizes=(128, 128),
    #     verbose=True,
    #     batch_size=32,
    #     activation="relu",
    #     early_stopping=True,
    #     max_iter=5000,
    #     random_state=42,
    #     shuffle=True)

    rf.fit(x_train, y_train)

    pred = rf.predict(x_test)

    mae = metrics.mean_absolute_error(y_test, pred)
    r2 = metrics.r2_score(y_test, pred)

    # Check performance
    # logging.info("Feature importances: \n%s" % str(rf.feature_importances_))
    logging.info("MAE %f" % mae)
    logging.info("R2 %f" % r2)

    # Create explainer
    ids = list(pd.read_csv("data/jec/jec_attributes_test_labels.csv")["index"])
    for i in range(len(x_test)):
        id_x = ids[i]
        explainer = LimeTabularExplainer(x_train, feature_names=features, class_names=['valor'],
                                         # categorical_features=categorical_features,
                                         verbose=True, mode='regression')
        exp = explainer.explain_instance(x_test[i], rf.predict, num_features=20)
        exp.save_to_file('data/jec/predict_tabular_reg_%d.html' % id_x)
