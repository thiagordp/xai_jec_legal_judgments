"""

"""
import logging

from data_processing import process_attrib
from tabular_dice.jec_tabular_dice import jec_dice, example_dice, jec_dice_2
from tabular_lime.jec_tabular_regression import jec_lime_regression
from tabular_shap.shap_example import shap_test
from text_lime.text_classifier import legal_text_lime


def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename='lime_example.log',
                        level=logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def main():
    setup_logging()
    # legal_text_lime()
    # process_attrib()
    # jec_lime_regression()
    # shap_test()
    #example_dice()
    jec_dice_2()


if __name__ == '__main__':
    main()
