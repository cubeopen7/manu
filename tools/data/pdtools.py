# -*- coding: utf-8 -*-

__all__ = [
    "trans_numerical",      # 转为数字类型
    "fill_missing",         # 填补缺失值
    "find_missing",         # 发现缺失列, 返回列名
]

import copy
import pandas as pd

def find_missing(_data):
    '''
    发现缺失列, 返回列名
    :param _data: 原数据
    :return: 缺失的列的名称列表
    '''
    if not isinstance(_data, pd.DataFrame):
        raise ValueError("input must be Series or DataFrame, got %s" % (type(_data)))

    t_miss = _data.isnull().any(axis=0)
    return t_miss[t_miss].index.tolist()

def fill_missing(_data, _type="all", methon="mean", fill=None):
    '''
    填补缺失值, 可以针对不同的列, 使用不同的方法
    :param _data: 原数据
    :param _type: 对何种类型的数据进行填补
                    1. "all": 全部数据
                    2. "num": 数字列数据
                    3. "obj": 字符串列数据
    :param methon: 填补方法
                    1. "mean": 均值, 只能在数字列中使用
                    2. "linear": 线性回归布置, 只能在数字列中使用
                    3. "mode": 众数模式, 可以在任何数据中使用, 通用
                    4. "zero": 补0, 只能在数字列中使用
                    5. "manual": 使用fill指定的值进行补充
    :return: 填补后的数据
    '''
    data = copy.deepcopy(_data)


def trans_numerical(_data):
    '''
    将Series或DataFrame中数据列的类型转换为数字型, 只对可转换的数字列进行操作, 非数字列原样返回
    :param _data: 待转换的Series或DataFrame
    :return: 转换后的数据
    '''
    data = _data.copy()
    if isinstance(data, pd.Series):
        return pd.to_numeric(data, errors="ignore")

    if isinstance(data, pd.DataFrame):
        for coln in data.columns:
            data[coln] = pd.to_numeric(data[coln], errors="ignore")
        return data

    raise ValueError("input must be Series or DataFrame, got %s" % (type(data)))