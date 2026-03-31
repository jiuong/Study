"""
预测下一根K线 OCHL
"""
import argparse
import pickle
import numpy as np
from keras.models import load_model
from config import FEATURES
from data_loader import process_full
from args import get_random_train_args, get_args

def predict_next(period: str = "15min") -> tuple[float, float, float, float]:
    return


if __name__ == "__main__":
    args = get_args()
    predict_next(args.period)