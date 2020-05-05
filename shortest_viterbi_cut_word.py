# -*- coding: utf-8 -*-
# @Time    : 2020/4/29 7:36 下午
# @Author  : lizhen
# @FileName: shortest_viterbi_cut_word.py
# @Description:
'''
维特算法分词应用
'''
# 使用viterbi算法-最短路径法，来对文本进行分词
import numpy as np
import pickle


class ViterbiSegment():

    def __init__(self, corpus_path, model_path, mode="train"):
        self.corpus_path = corpus_path
        self.model_path = model_path
        if mode == "work":  # 如果是"work"模式，需要加载已经训练好的参数
            self.vocab, self.word_distance, self.max_word_len = pickle.load(open("model.pkl", 'rb'))

    # 加载人民日报语料,形成每个句子的词语列表，用于后面统计词语频数
    def load_corpus(self, default_corpus_size=None):
        words_list = []
        with open(self.corpus_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            lines = list(filter(lambda x: len(x) > 0, lines))  # 删除空行
            if default_corpus_size != None: lines = lines[:default_corpus_size]  # 测试阶段截取较小的语料
            print("文档总行数是", len(lines))
            for line in lines:
                line = line.replace('\n', '').split("  ")[1:]
                words = list(map(lambda x: x.split('/')[0], line))
                words = list(filter(lambda x: len(x) > 0, words))
                words_list.append(words)
        return words_list

    # 基于标注语料，训练一份词语的概率分布，以及条件概率分布————当然最终目的，是得到两个词语之间的连接权重(也可以理解为转移概率)
    # 转移概率越大，说明两个词语前后相邻的概率越大，那么，从前一个词转移到后一个词语花费的代价就越小。
    def train_simple(self, default_corpus_size=None):  # 简单的边权重计算方式，与hanlp只是权重计算方式的不同
        self.word_num = {}
        self.word_pair_num = {}
        for words in self.load_corpus(default_corpus_size=default_corpus_size):
            words = ["<start>"] + words + ["<end>"]  # 首尾添加标记
            for word in words:
                self.word_num[word] = self.word_num.get(word, 0) + 1  # 词语频数

            for i in range(len(words) - 1):
                word_pair = (words[i], words[i + 1])  # 由于要计算的是条件概率，词语先后是需要考虑的
                self.word_pair_num[word_pair] = self.word_pair_num.get(word_pair, 0) + 1  # 词语对的频数
        # p(AB)=p(A)*p(B|A)=(num_A/num_all)*(num_AB/num_A)=num_AB/num_all。
        # 这个权重计算公式的优点是计算效率快；缺点是丢失了num_A带来的信息
        # 这个训练算法的效率不太重要；权重包含的信息量尽量大，或者说更精准地刻画词语对的分布，是最重要的事情。
        num_all = np.sum(list(self.word_num.values()))  # 语料中词语的总数
        word_pair_prob = {}
        for word_pair in self.word_pair_num:
            word_pair_prob[word_pair] = self.word_pair_num[word_pair] / num_all  # 词语对，也就是边出现的概率

        # 由于我们最终要做的是求最短路径，要求图的边权重是一个表示“代价”或者距离的量，即权重越大，两个节点之间的距离就越远。而前面得到的条件概率与这个距离是负相关的
        # 我们需要对条件概率求倒数，来获得符合场景要求的权重
        # 另外，由于条件概率可能是一个非常小的数，比如0.000001，倒数会很大。我们在运行维特比的时候，需要把多条边的权重加起来——可能遇到上溢出的情况。
        # 常用的避免上溢出的策略是去自然对数。
        self.word_distance = {}
        for word_pair in self.word_pair_num:
            self.word_distance[word_pair] = np.log(1 / word_pair_prob[word_pair])

        self.vocab = set(list(self.word_num.keys()))
        self.max_word_len = 0
        for word in self.vocab:
            if len(word) > self.max_word_len:
                self.max_word_len = len(word)

        model = (self.vocab, self.word_distance, self.max_word_len)
        pickle.dump(model, open(self.model_path, 'wb'))  # 保存参数

    def train_hanlp(self, default_corpus_size=None):
        """
        hanlp里使用的连接器权重计算方式稍微复杂一点，综合考虑了前词出现的概率，以及后词出现的条件规律，有点像全概率p(A)*p(B|A)=p(AB)
        dSmoothingPara 平滑参数0.1, frequency A出现的频率, MAX_FREQUENCY 总词频
        dTemp 平滑因子 1 / MAX_FREQUENCY + 0.00001, nTwoWordsFreq AB共现频次
        -Math.log(dSmoothingPara * frequency / (MAX_FREQUENCY)
        + (1 - dSmoothingPara) * ((1 - dTemp) * nTwoWordsFreq / frequency + dTemp));
        """

        self.word_num = {}
        self.word_pair_num = {}
        for words in self.load_corpus(default_corpus_size=default_corpus_size):
            words = ["<start>"] + words + ["<end>"]
            for word in words:
                self.word_num[word] = self.word_num.get(word, 0) + 1

            for i in range(len(words) - 1):
                word_pair = (words[i], words[i + 1])  # 由于要计算的是条件概率，词语先后是需要考虑的
                self.word_pair_num[word_pair] = self.word_pair_num.get(word_pair, 0) + 1

        num_all = np.sum(list(self.word_num.values()))
        dSmoothingPara = 0.1
        dTemp = 1 / num_all + 0.00001
        word_pair_prob = {}
        for word_pair in self.word_pair_num:
            word_A, word_B = word_pair
            # hanlp里的权重计算公式比较复杂，在查不到设计思路的情况下，我们默认hanlp作者是辛苦研制之后，凑出来的~
            word_pair_prob[word_pair] = dSmoothingPara * self.word_num.get(word_A) / num_all + \
                                        (1 - dSmoothingPara) * ((1 - dTemp) * self.word_pair_num[word_pair] / (
                    self.word_num.get(word_A) + dTemp))

        # 由于我们最终要做的是求最短路径，要求图的边权重是一个表示“代价”或者距离的量，即权重越大，两个节点之间的距离就越远。而前面得到的条件概率与这个距离是负相关的
        # 我们需要对条件概率求倒数，来获得符合场景要求的权重
        # 另外，由于条件概率可能是一个非常小的数，比如0.000001，倒数会很大。我们在运行维特比的时候，需要把多条边的权重加起来——可能遇到上溢出的情况。
        # 常用的避免上溢出的策略是去自然对数。
        self.word_distance = {}
        for word_pair in self.word_pair_num:
            word_A, _ = word_pair
            self.word_distance[word_pair] = np.log(1 / word_pair_prob[word_pair])
        #         print(self.word_distance)
        self.vocab = set(list(self.word_num.keys()))
        self.max_word_len = 0
        for word in self.vocab:
            if len(word) > self.max_word_len:
                self.max_word_len = len(word)

        model = (self.vocab, self.word_distance, self.max_word_len)
        pickle.dump(model, open(self.model_path, 'wb'))

    # 使用改版前向最大匹配法生成词图
    def generate_word_graph(self, text):
        word_graph = []
        for i in range(len(text)):
            cand_words = []
            window_len = self.max_word_len
            # 当索引快到文本右边界时，需要控制窗口长度，以免超出索引
            if i + self.max_word_len >= len(text):
                window_len = len(text) - i + 1
            for j in range(1, window_len):  # 遍历这个窗口内的子字符串，查看是否有词表中的词语
                cand_word = text[i: i + j]
                next_index = i + len(cand_word) + 1
                if cand_word in self.vocab:
                    cand_words.append([cand_word, next_index])
            cand_words.append([text[i], i + 1 + 1])  # 单字必须保留
            word_graph.append(cand_words)
        return word_graph

    # 使用维特比算法求词图的最短路径
    def viterbi(self, word_graph):
        path_length_map = {}  # 用于存储所有的路径，后面的邻接词语所在位置，以及对应的长度
        word_graph = [[["<start>", 1]]] + word_graph + [[["<end>", -1]]]
        # 这是一种比较简单的数据结构
        path_length_map[("<start>",)] = [1, 0]  # start处，后面的临接词语在列表的1处，路径长度是0,。

        for i in range(1, len(word_graph)):
            distance_from_start2current = {}
            if len(word_graph[i]) == 0:
                continue
            for former_path in list(path_length_map.keys()):  # path_length_map内容一直在变，需要深拷贝key,也就是已经积累的所有路径
                # 取出已经积累的路径，后面的临接词语位置，以及路径的长度。
                [next_index_4_former_path, former_distance] = path_length_map[former_path]
                former_word = former_path[-1]
                later_path = list(former_path)
                if next_index_4_former_path == i:  # 如果这条路径的临接词语的位置就是当前索引
                    for current_word in word_graph[i]:  # 遍历词图数据中，这个位置上的所有换选词语，然后与former_path拼接新路径
                        current_word, next_index = current_word
                        new_path = tuple(later_path + [current_word])  # 只有int, string, tuple这种不可修改的数据类型可以hash，
                        # 也就是成为dict的key
                        # 计算新路径的长度
                        new_path_len = former_distance + self.word_distance.get((former_word, current_word), 100)

                        path_length_map[new_path] = [next_index, new_path_len]  # 存储新路径后面的临接词语，以及路径长度

                        # 维特比的部分。选择到达当前节点的路径中，最短的那一条
                        if current_word in distance_from_start2current:  # 如果已经有到达当前词语的路径，需要择优
                            if distance_from_start2current[current_word][1] > new_path_len:  # 如果当前新路径比已有的更短
                                distance_from_start2current[current_word] = [new_path, new_path_len]  # 用更短的路径数据覆盖原来的
                        else:
                            distance_from_start2current[current_word] = [new_path, new_path_len]  # 如果还没有这条路径，就记录它
        shortest_path = distance_from_start2current["<end>"][0]
        shortest_path = shortest_path[1:-1]
        return shortest_path

    # 对文本分词
    def segment(self, text):
        word_graph = self.generate_word_graph(text)  # 对文本进行全切分
        shortest_path = self.viterbi(word_graph)
        return shortest_path

    # 基于标注语料，对模型进行评价
    def evaluation(self):
        pass


if __name__ == '__main__':
    # hanlp 与 simple 的区别只是在计算权重的方式上有些不同
    corpus_path = 'data/199801.txt'
    model_path = 'data/model.pkl'
    V = ViterbiSegment(corpus_path, model_path, mode="train")
    V.train_simple(default_corpus_size=None)
    print(V.segment("体操小将王惠莹艰苦拚搏。"))
    print(V.segment("我们一定要战胜敌人，我们认为它们都是纸老虎。"))
