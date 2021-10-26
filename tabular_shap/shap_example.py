import pandas as pd
import shap
import xgboost


def load_data():
    x_train = pd.read_csv("data/jec/jec_attributes_proc_train.csv").values
    x_test = pd.read_csv("data/jec/jec_attributes_proc_test.csv").values
    y_test = pd.read_csv("data/jec/jec_attributes_test_labels.csv")["label"].values
    y_train = pd.read_csv("data/jec/jec_attributes_train_labels.csv")["label"].values
    features = pd.read_csv("data/jec/jec_attributes_proc_train.csv").columns

    return x_train, x_test, y_train, y_test, features


def shap_test():
    x_train, x_test, y_train, y_test, features = load_data()
    # train an XGBoost model
    model = xgboost.XGBRegressor()
    model.fit(x_train, y_train)

    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    explainer = shap.Explainer(model, feature_names=features)
    shap_values = explainer(x_test[1:2])

    # visualize the first prediction's explanation
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values, x_test)


if __name__ == '__main__':
    shap_test()
