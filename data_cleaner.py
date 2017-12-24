# -*- coding: utf-8 -*-

import os
import pandas as pd

from tools.data import find_miss_columns
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
    cols_tools = [t for t in cols_total if 'tool' in t.lower()]
    cols_number = [t for t in cols_features if 'tool' not in t.lower()]

    # 列名修改
    for i, coln in enumerate(cols_tools):
        t_index = cols_total.index(coln)
        t_name = cols_total[t_index+1].split("X")[0] + "TOOL"
        cols_total[t_index] = t_name
        cols_tools[i] = t_name

    train_data.columns = cols_total
    train_data.train_data
    pd.DataFrame().isnull()
    a = 1