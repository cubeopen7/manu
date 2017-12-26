# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

from tools.data import find_missing
from tools.utilities import list_remove

root_dir = os.path.split(os.path.abspath(__file__))[0].replace("\\", "/") + "/"
data_dir = root_dir + "data/"
train_file = "训练.xlsx"
test_file1 = "测试A.xlsx"
test_file2 = "测试B.xlsx"

id_cols = ["ID"]
y_cols = ["Y"]

if __name__ == "__main__":
    # train_data = pd.read_excel(data_dir + train_file)
    # test_data1 = pd.read_excel(data_dir + test_file1)
    # test_data2 = pd.read_excel(data_dir + test_file2)
    # train_num, test_num1, test_num2 = train_data.shape[0], test_data1.shape[0], test_data2.shape[0]
    #
    # cols_total = train_data.columns.tolist()
    # cols_features = list_remove(cols_total, id_cols+y_cols)
    # cols_tools = [t for t in cols_total if 'tool' in t.lower()]
    # cols_number = [t for t in cols_features if 'tool' not in t.lower()]
    #
    # # 列名修改
    # for i, coln in enumerate(cols_tools):
    #     t_index = cols_total.index(coln)
    #     t_name = cols_total[t_index+1].split("X")[0] + "TOOL"
    #     cols_total[t_index] = t_name
    #     cols_tools[i] = t_name
    #     t_index2 = cols_features.index(coln)
    #     cols_features[t_index2] = t_name
    # train_data.columns = cols_total
    # test_data1.columns = list_remove(cols_total, y_cols)
    # test_data2.columns = list_remove(cols_total, y_cols)
    #
    # # 缺失值填补
    # cols_miss = find_missing(train_data)
    # miss_count = train_data[cols_miss].isnull().sum(axis=0)
    # drop1 = miss_count[miss_count >= train_num // 2].index.tolist()
    # # 1. 移除缺失过多的列
    # train_data = train_data.drop(drop1, axis=1)
    # test_data1 = test_data1.drop(drop1, axis=1)
    # test_data2 = test_data2.drop(drop1, axis=1)
    # cols_total = list_remove(cols_total, drop1)
    # cols_features = list_remove(cols_features, drop1)
    # cols_tools = list_remove(cols_tools, drop1)
    # cols_number = list_remove(cols_number, drop1)
    # print(train_data.shape)
    # # 2. 去除只有单一值或近似单一值的列
    # def _value_count(_data):
    #     t1 = _data.value_counts().sort_values(ascending=False)
    #     return t1.iloc[0] / t1.sum()
    # num_count = train_data[cols_number].apply(_value_count)
    # drop2 = num_count[num_count >= 0.9].index.tolist()
    # train_data = train_data.drop(drop2, axis=1)
    # test_data1 = test_data1.drop(drop2, axis=1)
    # test_data2 = test_data2.drop(drop2, axis=1)
    # cols_total = list_remove(cols_total, drop2)
    # cols_features = list_remove(cols_features, drop2)
    # cols_tools = list_remove(cols_tools, drop2)
    # cols_number = list_remove(cols_number, drop2)
    # print(train_data.shape)
    # # 3. 用对应过程的中位数填补缺失值(TOOL列没有缺失)
    # def _fill(_data, col, tlcol, fill_data):
    #     if not np.isnan(_data[col]):
    #         return _data[col]
    #     t = fill_data.loc[_data[tlcol], col]
    #     return t
    # total_data = pd.concat([train_data[id_cols + cols_features],
    #                         test_data1[id_cols + cols_features],
    #                         test_data2[id_cols + cols_features]], axis=0)
    # total_data = total_data.reset_index(drop=True)
    # for coln in cols_tools:
    #     tid = coln.replace("TOOL", "")
    #     tlist = [t for t in cols_total if tid in t]
    #     tdata = total_data[tlist]
    #     tmean = tdata.groupby(coln).median()
    #     for coln2 in tlist:
    #         if coln2 == coln:
    #             continue
    #         t_data = total_data[[coln, coln2]]
    #         for i in range(total_data.shape[0]):
    #             t_val = total_data[coln2].iloc[i]
    #             if np.isnan(t_val):
    #                 t_type = total_data[coln].iloc[i]
    #                 new_val = tmean.loc[t_type, coln2]
    #                 total_data.ix[i, coln2] = new_val
    # total_data.to_csv(data_dir + "total.csv", index=False)
    # # 4. 特殊Missing的处理, 直接使用中位数填补
    # total_data = pd.read_csv(data_dir + "total.csv")
    # for coln in find_missing(total_data):
    #     t_val = total_data[coln].median()
    #     total_data[coln].fillna(t_val, inplace=True)
    # total_data.to_csv(data_dir + "total.csv", index=False)

    # total_data = pd.read_csv(data_dir + "total.csv")
    # X_cols = [coln for coln in total_data.columns if coln not in id_cols+y_cols and "tool" not in coln.lower()]
    # tool_cols = [coln for coln in total_data.columns if "tool" in coln.lower()]
    # df_id = total_data[id_cols]
    # df_X = total_data[X_cols]
    # df_one = pd.get_dummies(total_data[tool_cols])
    # # 去除日期时间列
    # date_cols = df_X.columns[np.nonzero(df_X.mean() > 1e10)]
    # df_X.drop(date_cols, axis=1, inplace=True)
    # # 去除协方差过大的列
    # X = pd.concat([df_one, df_X], axis=1).iloc[:, 3:]
    # X_cov = np.cov(X.T)
    # t_index = np.sum(np.abs(X_cov) >= 0.8, axis=0) < 1000
    # X = X.loc[:, t_index].copy()
    #
    # # 标准化
    # from sklearn.preprocessing import StandardScaler
    # ss = StandardScaler()
    # X_norm = ss.fit_transform(X)
    # # 降维
    # from sklearn.decomposition import PCA
    # pca1 = PCA(n_components = X_norm.shape[1])
    # pca1.fit(X_norm)
    # k = np.nonzero(np.cumsum(pca1.explained_variance_ratio_) >= 0.9)[0][0] + 1
    # pca = PCA(n_components=k)
    # X_real = pca.fit_transform(X_norm)
    #
    # X_train, X_test = X_real[:500], X_real[500:]
    # id_train, id_test = df_id[:500], df_id[500:]
    # X_cols = ["X"+str(i+1) for i in range(X_train.shape[1])]
    # data_train = pd.concat([id_train.reset_index(drop=True), pd.DataFrame(X_train, columns=X_cols)], axis=1)
    # data_test = pd.concat([id_test.reset_index(drop=True), pd.DataFrame(X_test, columns=X_cols)], axis=1)
    #
    # y_train = pd.read_excel(data_dir + train_file).iloc[:, [-1]]
    # data_train = pd.concat([data_train, y_train], axis=1)
    # data_train.to_csv(data_dir + "train.csv", index=False)
    # data_test.to_csv(data_dir + "test.csv", index=False)

    # 训练模型
    train = pd.read_csv(data_dir + "train.csv")
    test = pd.read_csv(data_dir + "test.csv")
    X_train = train.iloc[:, 1:-1]
    X_test = test.iloc[:, 1:]

    from sklearn.linear_model import LassoCV
    seed = 777