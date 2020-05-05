# -*- coding: utf-8 -*-
# @Time    : 2020/4/28 11:02 下午
# @Author  : lizhen
# @FileName: utils.py
# @Description:


def load_word_dict(dict_path):
    word_dict = {}
    with open(dict_path,'r') as f:
        for line in f:
            word,freq,tag = line.strip().split(' ')
            word_dict[word] = (freq,tag)

    return word_dict



if __name__ == '__main__':
    pass
