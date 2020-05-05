# -*- coding: utf-8 -*-
# @Time    : 2020/4/28 11:02 下午
# @Author  : lizhen
# @FileName: maximum_match.py
# @Description:

from utils import load_word_dict


def forward_max_match(sentence, window_size, word_dict):
    """
    前向最大匹配算法：
    （1）扫描字典，测试读入的子串是否在字典中
    （2）如果存在，则从输入中删除掉该子串，重新按照规则取子串，重复（1）
    （3）如果不存在于字典中，则从右向左减少子串长度，重复（1）

    :param sentence: 待分词的句子
    :param window_size: 词的最大长度
    :param word_dict: 词典
    :return:
    """
    seg_words = []  # 存放分词结果

    while sentence:
        for word_len in range(window_size, 0, -1):
            if sentence[:word_len] in word_dict:
                # 如果该词再字典中，将该词保存到分词结果中,并将其从句中切出来
                seg_words.append(sentence[:word_len])
                sentence = sentence[word_len:]
                break
        else:
            # 如果窗口中的词不在字典中，将第一个字切分出来
            seg_words.append(sentence[:word_len])
            sentence = sentence[word_len:]

    return '/'.join(seg_words)


def backward_max_match(sentence, window_size, word_dict):
    """
    后向最大匹配算法：
    （1）从后向前扫描字典，测试读入的子串是否在字典中
    （2）如果存在，则从输入中删除掉该子串，重新按照规则取子串，重复（1）
    （3）如果不存在于字典中，则从右向左减少子串长度，重复（1）

    :param sentence: 待分词的句子
    :param window_size: 词的最大长度
    :param word_dict: 词典
    :return:
    """
    seg_words = []
    while sentence:
        for word_len in range(window_size, 0, -1):  # 每次去掉匹配字段最前面的一个字

            if sentence[len(sentence) - word_len:] in word_dict:
                # 如果该词再字典中，将该词保存到分词结果中,并将其从句中切出来
                seg_words.append(sentence[len(sentence) - word_len:])
                sentence = sentence[:len(sentence) - word_len]
                break
        else:
            # 如果窗口中的词不在字典中，将最后一个字切分出来
            seg_words.append(sentence[-1:])
            sentence = sentence[:-1]
    seg_words.reverse()
    return '/'.join(seg_words)

def fb_max_match(sentence,window_size,word_dict):
    """
    1. 比较正向最大匹配和逆向最大匹配结果
    2. 如果分词数量结果不同，那么取分词数量较少的那个
    3. 如果分词数量结果相同
       * 分词结果相同，可以返回任何一个
       * 分词结果不同，返回单字数比较少的那个
    :param sentence: 待分词的句子
    :param window_size: 词的最大长度
    :param word_dict: 词典
    :return:
    """
    forward_seg = forward_max_match(sentence,window_size,word_dict)
    backward_seg = backward_max_match(sentence,window_size,word_dict)

    # 如果分词结果不同，返回词数较小的分词结果
    if len(forward_seg) != len(backward_seg):
        return forward_seg if len(forward_seg) < len(backward_seg) else backward_seg
    else:
        # 如果分词结果词数相同，优先考虑返回包含单个字符最少的分词结果
        forward_single_word_count = len([filter(lambda x:len(x) == 1,forward_seg)])
        backward_single_word_count = len([filter(lambda x:len(x) == 1,backward_seg)])
        if forward_single_word_count != backward_single_word_count:
            return forward_seg if forward_single_word_count < backward_single_word_count else backward_seg
        else:
            # 否则，返回任意结果
            return forward_seg




if __name__ == '__main__':
    word_dict_path = 'data/chinese_word_dict.txt'
    word_dict = load_word_dict(word_dict_path)
    sentence = '前向最大匹配算法，是从待分词句子的左边向右边搜索，寻找词的最大匹配。规定一个词的最大长度，每次扫描的时候寻找当前开始的这个长度的词来和字典中的词匹配，如果没有找到，就缩短长度继续寻找，直到找到字典中的词或者成为单字。'

    print(forward_max_match(sentence, window_size=7, word_dict=word_dict))
    print(backward_max_match(sentence, window_size=7, word_dict=word_dict))
    print(fb_max_match(sentence, window_size=7, word_dict=word_dict))
