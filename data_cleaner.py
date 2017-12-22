# -*- coding: utf-8 -*-

import os
import pandas as pd

from tools.utilities import list_remove

root_dir = os.path.split(os.path.abspath(__file__))[0].replace("\\", "/") + "/"
data_dir = root_dir + "data/"
train_file = "训练.xlsx"
test_file1 = "测试A.xlsx"
test_file2 = "测试B.xlsx"

id_cols = ["ID"]
y_cols = ["Y"]

if __name__ == "__main__":
    train_data = pd.read_excel(data_dir + train_file)
    test_data1 = pd.read_excel(data_dir + test_file1)
    test_data2 = pd.read_excel(data_dir + test_file2)

    cols_total = train_data.columns.tolist()
    cols_features = list_remove(cols_total, id_cols+y_cols)
