"""
Tabular XAI example using LIME applied to JEC judgments
"""
import pandas as pd
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
from matplotlib import pyplot as plt

from text_lime.text_classifier import process_text


def load_jec_data():

    # load attributes
    df = pd.read_excel("data/attributes.xlsx", sheet_name=0)
    count = {1: 0, 0: 0}


def tabular_lime():
    pass
