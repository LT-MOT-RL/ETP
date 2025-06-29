#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import importlib
import os
import sys


def get_exp_by_file(exp_file):
    sys.path.append(os.path.dirname(exp_file))
    # print("os.path.dirname(exp_file) : ",os.path.dirname(exp_file))
    # print("os.path.basename(exp_file) : ",os.path.basename(exp_file))
    # print("os.path.basename(exp_file).split('.')[0] : ", os.path.basename(exp_file).split(".")[0])

    # os.path.dirname(exp_file) :  exps/default
    # os.path.basename(exp_file) :  unicorn_track_tiny_sot_only
    # os.path.basename(exp_file).split('.')[0] :  unicorn_track_tiny_sot_only

    # input("??")
    current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
    exp = current_exp.Exp()
    return exp


def get_exp_by_name(exp_name):
    import yolox

    yolox_path = os.path.dirname(os.path.dirname(yolox.__file__))
    filedict = {
        "yolox-s": "yolox_s.py",
        "yolox-m": "yolox_m.py",
        "yolox-l": "yolox_l.py",
        "yolox-x": "yolox_x.py",
        "yolox-tiny": "yolox_tiny.py",
        "yolox-nano": "nano.py",
        "yolov3": "yolov3.py",
    }
    filename = filedict[exp_name]
    exp_path = os.path.join(yolox_path, "exps", "default", filename)
    return get_exp_by_file(exp_path)


def get_exp(exp_file, exp_name):
    """
    get Exp object by file or name. If exp_file and exp_name
    are both provided, get Exp by exp_file.

    Args:
        exp_file (str): file path of experiment.
        exp_name (str): name of experiment. "yolo-s",
    """
    assert (
        exp_file is not None or exp_name is not None
    ), "plz provide exp file or exp name."
    if exp_file is not None:
        return get_exp_by_file(exp_file)
    else:
        return get_exp_by_name(exp_name)
