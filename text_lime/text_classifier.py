import glob
import logging

import nltk
import pandas as pd
import sklearn
import sklearn.ensemble
import sklearn.metrics
from lime.lime_text import LimeTextExplainer
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

nltk.download('stopwords')

DATASET_LABEL_PRESO = "/media/trdp/Arquivos/Studies/dev_msc/Datasets/stf_hc_dataset/Preso/"
DATASET_LABEL_SOLTO = "/media/trdp/Arquivos/Studies/dev_msc/Datasets/stf_hc_dataset/Solto/"
DATASET_LABEL_PROC = "/media/trdp/Arquivos/Studies/dev_msc/Datasets/processos_transp_aereo/txts_atualizados_sd_manual/novos/"


def process_text(text):
    text = text.lower()
    text = text.replace("\n", " \n ")
    for char in ",.;:-_/*@#$%&(“…”){}[]":
        text = text.replace(char, " ")

    for word in stopwords.words('portuguese'):
        token = " " + word + " "
        text = text.replace(token, " ")
    text = text.replace("  ", " ")

    return text


def load_jec_data():
    logging.info("Loading JEC data")

    data = []
    list_docs = glob.glob(DATASET_LABEL_PROC + "*.txt")

    logging.info("Loading Text files")

    # Open df
    df = pd.read_excel("data/attributes_jec.xlsx", sheet_name=0)
    count = {1: 0, 0: 0}
    for doc_path in list_docs:
        doc_id = int(doc_path.replace(DATASET_LABEL_PROC, "").replace(".txt", ""))
        text = open(doc_path).read()
        try:
            value = df[df["Sentença"] == doc_id]["Valor individual do dano moral"].tolist()[0]
            if value > 1:
                label = 1
            else:
                label = 0
            count[label] += 1

            data.append([doc_id, process_text(text), label])
        except:
            continue
    logging.info("Counts: %s" % str(count))
    logging.info("Shuffing")
    df = pd.DataFrame(data, columns=["id", "text", "label"])
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.sample(frac=1).reset_index(drop=True)

    logging.info("Saving")
    train_df, test_df = train_test_split(df, test_size=0.2)
    train_df[["id", "text"]].to_csv("data/jec/train_text.csv", index=False)
    train_df[["id", "label"]].to_csv("data/jec/train_labels.csv", index=False)
    test_df[["id", "text"]].to_csv("data/jec/test_text.csv", index=False)
    test_df[["id", "label"]].to_csv("data/jec/test_labels.csv", index=False)


def load_stf_data():
    logging.info("Loading STF data")

    data = []
    list_preso = glob.glob(DATASET_LABEL_PRESO + "*.txt")

    logging.info("Loading Text files")
    for doc_path in list_preso:
        text = open(doc_path).read()

        data.append([process_text(text), 1])

    list_solto = glob.glob(DATASET_LABEL_SOLTO + "*.txt")
    for doc_path in list_solto:
        text = open(doc_path).read()
        data.append([process_text(text), 0])

    logging.info("Shuffing")
    df = pd.DataFrame(data, columns=["text", "label"])
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.sample(frac=1).reset_index(drop=True)

    logging.info("Saving")
    train_df, test_df = train_test_split(df, test_size=0.3)
    train_df["text"].to_csv("data/train_text.csv")
    train_df["label"].to_csv("data/train_labels.csv")
    test_df["text"].to_csv("data/test_text.csv")
    test_df["label"].to_csv("data/test_labels.csv")


def _data_preparation():
    logging.info("Data preparation")
    categories = ['alt.atheism', 'soc.religion.christian']
    newsgroups_train = list(pd.read_csv("data/jec/train_text.csv", index_col=0)["text"])
    labels_train = list(pd.read_csv("data/jec/train_labels.csv", index_col=0)["label"])
    newsgroups_test = list(pd.read_csv("data/jec/test_text.csv", index_col=0)["text"])
    labels_test = list(pd.read_csv("data/jec/test_labels.csv", index_col=0)["label"])
    class_names = ['improcedente', 'procedente']

    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False, max_features=1000)
    train_vectors = vectorizer.fit_transform(newsgroups_train)
    test_vectors = vectorizer.transform(newsgroups_test)

    features = vectorizer.get_feature_names()

    return newsgroups_train, newsgroups_test, train_vectors, test_vectors, vectorizer, labels_train, labels_test, class_names, features


def train_predict_rf(newsgroups_train, newsgroups_test, labels_train, labels_test, train_vectors, test_vectors):
    logging.info("Train RF")

    # rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf = MLPClassifier(
        hidden_layer_sizes=(128, 128),
        verbose=True,
        batch_size=32,
        activation="tanh",
        early_stopping=True,
        max_iter=5000,
        random_state=42,
        shuffle=True)

    rf.fit(train_vectors, labels_train)

    pred = rf.predict(test_vectors)

    acc = metrics.accuracy_score(labels_test, pred)
    f1 = metrics.f1_score(labels_test, pred, average='binary')

    # logging.info("Feature importances: \n%s" % str(rf.feature_importances_))
    logging.info("F1 %f" % f1)
    logging.info("Acc %f" % acc)

    return rf


def legal_text_lime():
    load_jec_data()

    logging.info("Starting")
    newsgroups_train, newsgroups_test, train_vectors, test_vectors, vectorizer, labels_train, labels_test, class_names, features = _data_preparation()

    clf = train_predict_rf(newsgroups_train, newsgroups_test, labels_train, labels_test, train_vectors, test_vectors)

    # features_imp_data = [[feature, importance] for feature, importance in zip(features, clf.feature_importances_)]
    # df = pd.DataFrame(features_imp_data, columns=["Feature", "importance"])
    # df.to_excel("data/jec/importances.xlsx", index=False)

    list_index = [214, 231, 15, 23, 53]
    list_sentenca_num = [293, 885, 1115, 1013, 137]

    for idx, sentenca_num in zip(list_index, list_sentenca_num):
        c = make_pipeline(vectorizer, clf)
        logging.info(c.predict_proba([newsgroups_test[idx]]))

        logging.info("Start explanation")
        explainer = LimeTextExplainer(class_names=class_names, verbose=True)

        exp = explainer.explain_instance(newsgroups_test[idx], c.predict_proba, num_features=10)
        logging.info('Document id: %d' % idx)
        logging.info('Probability(christian) =%s' % str(c.predict_proba([newsgroups_test[idx]])[0, 1]))
        logging.info('True class: %s' % class_names[labels_test[idx]])
        logging.info(exp.as_list())

        logging.info('Original prediction: %s' % str(clf.predict_proba(test_vectors[idx])[0, 1]))

        fig = exp.as_pyplot_figure()
        fig.set_figheight(5)
        fig.set_figwidth(9)

        plt.savefig('data/jec/predict_doc_%d.pdf' % sentenca_num, dpi=300)
        # plt.show()

        exp.save_to_file('data/jec/predict_doc_%d.html' % sentenca_num)
