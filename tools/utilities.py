# -*- coding: utf-8 -*-

import copy
from collections import Iterable

def list_remove(_list, value):
    '''
    将列表中的一些元素移除. 如果传入的是一个待移除元素列表, 则每个都会被移除
    :param _list: 原list
    :param value: 待移除的元素, 可以是单个元素, 也可以是元素的列表
    :return: 移除后的列表, 在新list上进行.
    '''
    new_list = copy.deepcopy(_list)
    if isinstance(value, Iterable):
        for t in value:
            new_list.remove(t)
    else:
        new_list.remove(value)
    return new_list
