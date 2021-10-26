"""

"""
import logging

import dice_ml
from dice_ml import Dice

from sklearn.datasets import load_iris, load_boston
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import pandas as pd



def load_data():
    x_train = pd.read_csv("data/jec/jec_attributes_proc.csv").values
    x_test = []#pd.read_csv("data/jec/jec_attributes_proc_test.csv").values
    y_test = []#pd.read_csv("data/jec/jec_attributes_test_labels.csv")["valor"].values
    y_train = pd.read_csv("data/jec/jec_attributes_labels.csv")["valor"].values
    features = pd.read_csv("data/jec/jec_attributes_proc.csv").columns

    return x_train, x_test, y_train, y_test, features

def example_dice():
    outcome_name = "target"
    boston_data = load_boston()
    df_boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
    df_boston[outcome_name] = pd.Series(boston_data.target)
    df_boston.head()
    df_boston.info()
    print(df_boston.describe())
    continuous_features_boston = df_boston.drop(outcome_name, axis=1).columns.tolist()
    target = df_boston[outcome_name]

    # Split data into train and test
    datasetX = df_boston.drop(outcome_name, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(datasetX,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=0)

    categorical_features = x_train.columns.difference(continuous_features_boston)

    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    transformations = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, continuous_features_boston),
            ('cat', categorical_transformer, categorical_features)])

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    regr_boston = Pipeline(steps=[('preprocessor', transformations),
                                  ('regressor', RandomForestRegressor())])
    model_boston = regr_boston.fit(x_train, y_train)

    d_boston = dice_ml.Data(dataframe=df_boston, continuous_features=continuous_features_boston, outcome_name=outcome_name)
    # We provide the type of model as a parameter (model_type)
    m_boston = dice_ml.Model(model=model_boston, backend="sklearn", model_type='regressor')

    exp_genetic_boston = Dice(d_boston, m_boston, method="genetic")

    # Multiple queries can be given as input at once
    query_instances_boston = x_train[2:3]
    genetic_boston = exp_genetic_boston.generate_counterfactuals(query_instances_boston,
                                                                 total_CFs=2,
                                                                 desired_range=[30, 45])
    print(genetic_boston.visualize_as_dataframe(show_only_changes=True))

    # Multiple queries can be given as input at once
    query_instances_boston = x_train[17:18]
    genetic_boston = exp_genetic_boston.generate_counterfactuals(query_instances_boston, total_CFs=4, desired_range=[40, 50])
    genetic_boston.visualize_as_dataframe(show_only_changes=True)


def jec_dice():
    x_train, x_test, y_train, y_test, features = load_data()

    regressor = RandomForestRegressor()
    regressor.fit(x_train, y_train)
    continuous_features = ["iextrav", "atraso"]
    outcome_name = "valor"

    jec_df_x = pd.DataFrame(x_train, columns=features)
    y_train_df = pd.DataFrame(y_train, columns=["valor"])
    jec_df = pd.concat([jec_df_x, y_train_df], axis=1)
    jec_df = jec_df.apply(pd.to_numeric, errors='coerce', downcast='float')
    d_boston = dice_ml.Data(dataframe=jec_df, continuous_features=continuous_features, outcome_name=outcome_name)
    m_boston = dice_ml.Model(model=regressor, backend="sklearn", model_type='regressor')
    exp_genetic_boston = Dice(d_boston, m_boston, method="genetic")

    x_test_df = pd.DataFrame(x_test, columns=features)
    x_test_df = x_test_df.apply(pd.to_numeric, errors='coerce', downcast='float')
    print(jec_df_x.dtypes)
    logging.info(jec_df_x.info())
    query_instances_boston = jec_df_x.iloc[[2]]
    query_instances_boston = query_instances_boston.apply(pd.to_numeric, errors='coerce', downcast='float')

    genetic_boston = exp_genetic_boston.generate_counterfactuals(query_instances_boston,
                                                                 total_CFs=4,
                                                                 verbose=True,
                                                                 desired_range=[9000, 10000])
    print(genetic_boston.visualize_as_dataframe(show_only_changes=True))

    # Multiple queries can be given as input at once
    query_instances_boston = x_test_df.iloc[[17]]
    genetic_boston = exp_genetic_boston.generate_counterfactuals(query_instances_boston, total_CFs=4, desired_range=[5000, 10000])
    genetic_boston.visualize_as_dataframe(show_only_changes=True)


def jec_dice_2():
    x_train, x_test, y_train, y_test, features = load_data()

    #regressor = RandomForestRegressor()
    regressor = MLPRegressor(
        hidden_layer_sizes=(16, 16, 16, 16),
        verbose=False,
        batch_size=32,
        activation="relu",
        early_stopping=True,
        max_iter=5000,
        random_state=42,
        shuffle=True)
    regressor.fit(x_train, y_train)

    continuous_features =list(set(features))# ["iextrav", "atraso"]
    outcome_name = "valor"

    df_jec = pd.DataFrame(x_train, columns=features)
    df_jec[outcome_name] = pd.Series(y_train)

    logging.info(df_jec.head())
    logging.info(df_jec.info())
    logging.info(df_jec.describe())

    target = df_jec[outcome_name]
    datasetX = df_jec.drop(outcome_name, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(datasetX,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=5)

    categorical_features = x_train.columns.difference(continuous_features)
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    transformations = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, continuous_features),
            ('cat', categorical_transformer, categorical_features)])

    regr_boston = Pipeline(steps=[('preprocessor', transformations),
                                  ('regressor', RandomForestRegressor())])
    model_boston = regr_boston.fit(x_train, y_train)

    d_boston = dice_ml.Data(dataframe=df_jec, continuous_features=continuous_features, outcome_name=outcome_name)
    # We provide the type of model as a parameter (model_type)
    m_boston = dice_ml.Model(model=model_boston, backend="sklearn", model_type='regressor')

    exp_genetic_boston = Dice(d_boston, m_boston, method="genetic")
    query_instances_boston = x_test[1:2]

    print("actual output", y_test[1:2])
    genetic_boston = exp_genetic_boston.generate_counterfactuals(query_instances_boston,
                                                                 total_CFs=2,
                                                                 desired_range=[15000, 25000])
    # query_instances_boston.to_excel("data/jec/conterfactuals/original.xlsx", index=False)
    results = genetic_boston.visualize_as_dataframe(show_only_changes=True)
    str_res = genetic_boston.to_json()
    with open("data/jec/conterfactuals/conter_fact.json", "w+") as fp:
        fp.write(str_res)
    print(genetic_boston.to_json())
    # print()
    # results.to_excel("data/jec/conterfactuals/conter_fact.xlsx", index=False)
    # print(genetic_boston.visualize_as_dataframe(show_only_changes=True))
