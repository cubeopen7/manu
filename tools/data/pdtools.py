# -*- coding: utf-8 -*-

def find_miss_columns(_data, _type="all"):
    if isinstance(_type, str):
        if "num" in _type.lower():
            _data = _data
