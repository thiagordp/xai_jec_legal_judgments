import logging

import numpy as np
import pandas as pd
import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


def load_attributes(inputs):
    logging.info("Loading attributes")

    dict_outputs = dict()

    attributes_df = pd.read_excel("data/attributes.xlsx", sheet_name=0)
    attributes_df.dropna(subset=['tipo_juiz'], inplace=True, axis=0)

    list_not_found = []
    for key in tqdm.tqdm(inputs):

        filtered_df = attributes_df.loc[attributes_df["judgement"] == int(key)]

        if len(filtered_df.index) == 0:
            list_not_found.append(key)
            # logging.error("Found no attributes for judgment %s" % key)
            continue

        dict_attr = filtered_df.to_dict('records')[0]
        dict_outputs[key] = dict_attr
    logging.error("Found no attributes for %d judgments: %s" % (len(list_not_found), str(list_not_found)))
    logging.info("Finished loading attributes")

    return dict_outputs


def represent_bow_tf(dict_inputs=None, vectorizer=None, predict=False):
    logging.info("Starting text representation")

    if dict_inputs is None:
        dict_inputs = dict()

    logging.info("Transforming dict to list")
    list_inputs = dict_inputs.values()
    if not predict:
        logging.info("Fit Transform BOW")
        vectorizer = CountVectorizer(ngram_range=(1, 4), max_features=25000, min_df=1)
        tf_inputs = vectorizer.fit_transform(list_inputs).toarray()

        feature_names = vectorizer.get_feature_names()
        logging.info("Features: %s" % len(feature_names))
    else:
        logging.info("Fit Transform BOW")
        tf_inputs = vectorizer.transform(list_inputs).toarray()
        feature_names = vectorizer.get_feature_names()

    dict_outputs = {}

    logging.info("Transforming to dict")
    for i, key in enumerate(dict_inputs.keys()):
        dict_outputs[key] = list(tf_inputs[i])

    logging.info("Finished representation")

    return dict_outputs, vectorizer, feature_names


def append_attributes_to_bow(dict_inputs, dict_attributes):
    logging.info("Appending Attributes to inputs")

    ignore_attrib_fields = list()

    for key_input in tqdm.tqdm(dict_inputs.keys()):
        dict_attri_case = dict_attributes[key_input]

        for key_attr in dict_attri_case.keys():
            value = dict_attri_case[key_attr]

            if isinstance(value, float) or isinstance(value, (int, np.integer)):
                dict_inputs[key_input].append(value)
            elif isinstance(value, list):
                dict_inputs[key_input].extend(value)

    logging.info("Finished appending")

    return dict_inputs


def __process_judge(judges, distinct_judges, type_judges, distinct_type_judges):
    judges = [str(judge_name).strip().lower().replace(" ", "_").replace(".", "") for judge_name in judges]
    type_judges = [str(judge_type).strip().lower().replace(" ", "_").replace(".", "") for judge_type in type_judges]

    judges = pd.get_dummies(judges, prefix='juiz')
    type_judges = pd.get_dummies(type_judges, prefix="tipo_juiz")
    judges.reindex(sorted(judges.columns), axis=1)
    judges.sort_index(axis=1, inplace=True)
    type_judges.sort_index(axis=1, inplace=True)

    if distinct_judges is not None:
        for distinct in distinct_judges:
            if distinct not in judges.columns:
                judges[distinct] = 0

    if distinct_type_judges is not None:

        for distinct_type in distinct_type_judges:

            if distinct_type not in type_judges.columns:
                type_judges[distinct_type] = 0

    list_distinct_judges = sorted(set(judges.columns))
    logging.info("list_dist_judges       %s", str(list_distinct_judges))
    list_distinct_type_judges = sorted(set(type_judges.columns))
    logging.info("list_dist_type_judges  %s", str(list_distinct_type_judges))

    # Remove column not in the default set of judges
    if distinct_judges is not None:
        for distinct_judge in list_distinct_judges:
            if distinct_judge not in distinct_judges:
                del judges[distinct_judge]
    if distinct_type_judges is not None:
        for distinct_type_judge in list_distinct_type_judges:
            if distinct_type_judge not in distinct_type_judges:
                del type_judges[distinct_type_judge]

    judges.sort_index(axis=1, inplace=True)
    type_judges.sort_index(axis=1, inplace=True)

    judges = [list(judge) for judge in list(judges.to_numpy())]
    type_judges = [list(type_judge) for type_judge in list(type_judges.to_numpy())]

    return judges, list_distinct_judges, type_judges, list_distinct_type_judges


def __process_has_x(feature, transf_feature):
    if transf_feature is None:
        transf_feature = LabelEncoder()

        transf_feature.fit(feature)

    return transf_feature.transform(feature), transf_feature


def __process_loss(feature):
    feature = [float(str(num).replace("-", "0").replace(",", ".")) for num in feature]
    return np.array(feature)


def __process_time_delay(feature):
    delay_minutes = list()

    for time_delay in feature:
        time_delay = time_delay.replace("- (superior a 4)", "00:00:00")
        time_delay = time_delay.replace("-", "00:00:00")
        time_delay = time_delay.split(" ")[-1]
        splits = time_delay.split(":")

        for i in range(3 - len(splits)):
            splits.append("00")

        seconds = float(splits[-1].strip()) / 60
        minutes = float(splits[-2].strip())
        hours = float(splits[-3].strip()) * 60

        delay_minutes.append(hours + minutes + seconds)

    return np.array(delay_minutes)


def transform_attributes(dict_attrib, transformer=None):
    """

    :param transformer:
    :param dict_attrib:
    :return:
        Transformed attributes
        Dict of transformers
    """

    logging.info("Transforming attributes")

    raw_data_df = pd.DataFrame.from_dict(dict_attrib, orient='index')

    # raw_data_df.to_excel("test.xlsx", index=False)

    # Extract attributes
    days_list = list(raw_data_df["dia"])
    months_list = list(raw_data_df["mes"])
    years_list = list(raw_data_df["ano"])
    day_week_list = list(raw_data_df["dia_semana"])
    judges = list(raw_data_df["juiz"])
    type_judges = list(raw_data_df["tipo_juiz"])

    if transformer is not None:
        transf_judges = transformer.get("juiz")
        transf_type_judges = transformer.get("tipo_juiz")
        has_permanent_loss_transf = transformer.get("extravio_permanente")
        has_temporally_loss_transf = transformer.get("extravio_temporario")
        has_luggage_violation_transf = transformer.get("tem_violacao_bagagem")
        has_flight_delay_transf = transformer.get("tem_atraso_voo")
        has_flight_cancellation_transf = transformer.get("tem_cancelamento_voo")
        # is_consumers_fault_transf = transformer.get("culpa_consumidor")
        has_adverse_flight_conditions_transf = transformer.get("tem_condicao_adversa_voo")
        has_no_show_transf = transformer.get("tem_no_show")
        has_overbooking_transf = transformer.get("tem_overbooking")
        has_cancel_refunding_transf = transformer.get("tem_cancelamento_usuario_ressarcimento")
        has_offer_disagreement_transf = transformer.get("tem_desacordo_oferta")

    else:
        transf_judges = None
        transf_type_judges = None
        has_permanent_loss_transf = None
        has_temporally_loss_transf = None
        has_luggage_violation_transf = None
        has_flight_delay_transf = None
        has_flight_cancellation_transf = None
        is_consumers_fault_transf = None
        has_adverse_flight_conditions_transf = None
        has_no_show_transf = None
        has_overbooking_transf = None
        has_cancel_refunding_transf = None
        has_offer_disagreement_transf = None

    judges, transf_judges, type_judges, transf_type_judges = __process_judge(judges, transf_judges, type_judges,
                                                                             transf_type_judges)
    # If transformer is None (usually when formatting training data, the input to the functions will be None.
    # Else, the transformer is passed as input, and it is used to transform the data instead of creating a new transformer
    has_permanent_loss_list, has_permanent_loss_transf = __process_has_x(raw_data_df["extravio_permanente"].values, has_permanent_loss_transf)
    has_temporally_loss_list, has_temporally_loss_transf = __process_has_x(raw_data_df["extravio_temporario"].values, has_temporally_loss_transf)
    interval_loss_list = __process_loss(raw_data_df["intevalo_extravio"].values)
    has_luggage_violation_list, has_luggage_violation_transf = __process_has_x(raw_data_df["tem_violacao_bagagem"].values, has_luggage_violation_transf)
    has_flight_delay_list, has_flight_delay_transf = __process_has_x(raw_data_df["tem_atraso_voo"].values, has_flight_delay_transf)
    has_flight_cancellation_list, has_flight_cancellation_transf = __process_has_x(raw_data_df["tem_cancelamento_voo"].values, has_flight_cancellation_transf)
    flight_delay_list = __process_time_delay(raw_data_df["qtd_atraso_voo"].values)
    # is_consumers_fault_list, is_consumers_fault_transf = __process_has_x(raw_data_df["culpa_consumidor"].values, is_consumers_fault_transf)
    has_adverse_flight_conditions_list, has_adverse_flight_conditions_transf = __process_has_x(raw_data_df["tem_condicao_adversa_voo"].values,
                                                                                               has_adverse_flight_conditions_transf)
    has_no_show_list, has_no_show_transf = __process_has_x(raw_data_df["tem_no_show"].values, has_no_show_transf)
    has_overbooking_list, has_overbooking_transf = __process_has_x(raw_data_df["tem_overbooking"].values, has_overbooking_transf)
    has_cancel_refunding_problem_list, has_cancel_refunding_transf = __process_has_x(raw_data_df["tem_cancelamento_usuario_ressarcimento"].values,
                                                                                     has_cancel_refunding_transf)
    has_offer_disagreement_list, has_offer_disagreement_transf = __process_has_x(raw_data_df["tem_desacordo_oferta"].values, has_offer_disagreement_transf)

    logging.info("Transform back to dicts")
    list_keys_attrib = dict_attrib.keys()
    dict_attrib = dict()

    for index_key, key_input in tqdm.tqdm(enumerate(list_keys_attrib)):
        dict_attrib[key_input] = dict()
        dict_attrib[key_input]["dia"] = days_list[index_key]
        dict_attrib[key_input]["mes"] = months_list[index_key]
        dict_attrib[key_input]["ano"] = years_list[index_key]
        dict_attrib[key_input]["dia_semana"] = day_week_list[index_key]
        dict_attrib[key_input]["juiz"] = judges[index_key]
        dict_attrib[key_input]["tipo_juiz"] = type_judges[index_key]
        dict_attrib[key_input]["juiz"] = judges[index_key]
        dict_attrib[key_input]["extravio_permanente"] = has_permanent_loss_list[index_key]
        dict_attrib[key_input]["extravio_temporario"] = has_temporally_loss_list[index_key]
        dict_attrib[key_input]["intevalo_extravio"] = interval_loss_list[index_key]
        dict_attrib[key_input]["tem_violacao_bagagem"] = has_luggage_violation_list[index_key]
        dict_attrib[key_input]["tem_atraso_voo"] = has_flight_delay_list[index_key]
        dict_attrib[key_input]["tem_cancelamento_voo"] = has_flight_cancellation_list[index_key]
        dict_attrib[key_input]["qtd_atraso_voo"] = flight_delay_list[index_key]
        # dict_attrib[key_input]["culpa_consumidor"] = is_consumers_fault_list[index_key]
        dict_attrib[key_input]["tem_condicao_adversa_voo"] = has_adverse_flight_conditions_list[index_key]
        dict_attrib[key_input]["tem_no_show"] = has_no_show_list[index_key]
        dict_attrib[key_input]["tem_overbooking"] = has_overbooking_list[index_key]
        dict_attrib[key_input]["tem_cancelamento_usuario_ressarcimento"] = has_cancel_refunding_problem_list[index_key]
        dict_attrib[key_input]["tem_desacordo_oferta"] = has_offer_disagreement_list[index_key]

    dict_transfs = dict()

    dict_transfs["extravio_permanente"] = has_permanent_loss_transf
    dict_transfs["juiz"] = transf_judges
    dict_transfs["tipo_juiz"] = transf_type_judges
    dict_transfs["extravio_temporario"] = has_temporally_loss_transf
    dict_transfs["tem_violacao_bagagem"] = has_luggage_violation_transf
    dict_transfs["tem_atraso_voo"] = has_flight_delay_transf
    dict_transfs["tem_cancelamento_voo"] = has_flight_cancellation_transf
    # dict_transfs["culpa_consumidor"] = is_consumers_fault_transf
    dict_transfs["tem_condicao_adversa_voo"] = has_adverse_flight_conditions_transf
    dict_transfs["tem_no_show"] = has_no_show_transf
    dict_transfs["tem_overbooking"] = has_overbooking_transf
    dict_transfs["tem_cancelamento_usuario_ressarcimento"] = has_cancel_refunding_transf
    dict_transfs["tem_desacordo_oferta"] = has_offer_disagreement_transf

    logging.info("Finished processing attributes")

    return dict_attrib, dict_transfs


def save_attributes(dict_info, train=True):
    dict_attributes = dict_info["attributes"]
    dict_attributes_transf = dict_info["attrib_transformer"]

    df = pd.DataFrame.from_dict(dict_attributes, orient="index").reset_index()
    df.drop('juiz', axis=1, inplace=True)
    df.drop('tipo_juiz', axis=1, inplace=True)
    df.drop('index', axis=1, inplace=True)

    if train:
        df.to_csv("data/jec/jec_attributes_proc_train.csv", index=False)
    else:
        df.to_csv("data/jec/jec_attributes_proc_test.csv", index=False)


def adjust_judge(dict_attrib, attrib_transf):
    logging.info("Adjusting judge")
    attrib_judges = attrib_transf["juiz"]
    attrib_type_judges = attrib_transf["tipo_juiz"]
    for key in dict_attrib.keys():
        #     for judge_name, judge_value in zip(attrib_judges, dict_attrib[key]["juiz"]):
        #         # print(key, judge_name, judge_value)
        #         dict_attrib[key][judge_name] = judge_value

        for type_judge_name, type_judge_value in zip(attrib_type_judges, dict_attrib[key]["tipo_juiz"]):
            # print(key, judge_name, judge_value)
            dict_attrib[key][type_judge_name] = type_judge_value

    return dict_attrib, attrib_transf


def process_attrib():
    outputs = dict()
    logging.info("Processing attributes")

    df = pd.read_csv("data/jec/train_labels.csv")
    ids_docs = list(df["id"])

    dict_attributes = load_attributes(ids_docs)

    dict_label = {}
    for key in dict_attributes.keys():
        valor_indenizacao = dict_attributes[key]["indenizacao"]
        dict_xx = {}
        if valor_indenizacao > 1:
            dict_xx["label"] = 1

        else:
            dict_xx["label"] = 0
        dict_xx["valor"] = valor_indenizacao
        dict_label[key] = dict_xx

    dict_attributes, dict_attributes_transf = transform_attributes(dict_attributes)

    adjust_judge(dict_attributes, dict_attributes_transf)
    outputs["attributes"] = dict_attributes
    outputs["attrib_transformer"] = dict_attributes_transf
    save_attributes(outputs)
    df = pd.DataFrame.from_dict(dict_label, orient="index").reset_index()
    df.to_csv("data/jec/jec_attributes_train_labels.csv", index=False)

    df = pd.read_csv("data/jec/test_labels.csv")
    ids_docs = list(df["id"])

    dict_attributes = load_attributes(ids_docs)

    dict_label = {}
    for key in dict_attributes.keys():
        valor_indenizacao = dict_attributes[key]["indenizacao"]
        dict_xx = {}
        if valor_indenizacao > 1:
            dict_xx["label"] = 1

        else:
            dict_xx["label"] = 0
        dict_xx["valor"] = valor_indenizacao
        dict_label[key] = dict_xx

    dict_attributes, dict_attributes_transf = transform_attributes(dict_attrib=dict_attributes, transformer=dict_attributes_transf)
    adjust_judge(dict_attributes, dict_attributes_transf)
    outputs["attributes"] = dict_attributes
    outputs["attrib_transformer"] = dict_attributes_transf
    save_attributes(outputs, False)

    df = pd.DataFrame.from_dict(dict_label, orient="index").reset_index()
    df.to_csv("data/jec/jec_attributes_test_labels.csv", index=False)

    return outputs
    # Get values here and save to two csv's. One w/ attr and another with labels
