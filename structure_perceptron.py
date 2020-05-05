# -*- coding: utf-8 -*-
# @Time    : 2020/5/4 10:44 下午
# @Author  : lizhen
# @FileName: structure_perceptron.py
# @Description:
import os
import time
import random
import pickle



class CPTTrain:
    def __init__(self, segment, train):
        self.__char_type = {}
        data_path = "data"
        for ind, name in enumerate(["punc", "alph", "date", "num"]):
            fn = data_path + "/" + name
            if os.path.isfile(fn):
                for line in open(fn, "r"):
                    self.__char_type[line.strip()] = ind
            else:
                print("can't open", fn)
                exit()

        self.__train_insts = None  # all instances for training.
        self.__feats_weight = None  # ["b", "m", "e", "s"][all the features] --> weight.
        self.__words_num = None  # total words num in all the instances.
        self.__insts_num = None  # namley the sentences' num.
        self.__cur_ite_ID = None  # current iteration index.
        self.__cur_inst_ID = None  # current index_th instance.
        self.__real_inst_ID = None  # the accurate index in training instances after randimizing.
        self.__last_update = None  # ["b".."s"][feature] --> [last_update_ite_ID, last_update_inst_ID]
        self.__feats_weight_sum = None  # sum of ["b".."s"][feature] from begin to end.

        if segment and train or not segment and not train:
            print("there is only a True and False in segment and train")
            exit()
        elif train:
            self.Train = self.__Train
        else:
            self.__LoadModel()
            self.Segment = self.__Segment

    def __LoadModel(self):
        model = "data/avgmodel"
        print("load", model, "...")
        self.__feats_weight = {}
        if os.path.isfile(model):
            start = time.clock()
            self.__feats_weight = pickle.load(open(model, "rb"))
            end = time.clock()
            print("It takes %d seconds" % (end - start))
        else:
            print("can't open", model)

    def __Train(self, corp_file_name, max_train_num, max_ite_num):
        if not self.__LoadCorp(corp_file_name, max_train_num):
            return False

        starttime = time.clock()

        self.__feats_weight = {}
        self.__last_update = {}
        self.__feats_weight_sum = {}

        for self.__cur_ite_ID in range(max_ite_num):
            if self.__Iterate():
                break

        self.__SaveModel()
        endtime = time.clock()
        print("total iteration times is %d seconds" % (endtime - starttime))

        return True

    def __GenerateFeats(self, inst):
        inst_feat = []
        for ind, [c, tag, t] in enumerate(inst):
            inst_feat.append([])
            if t == -1:
                continue
            # Cn
            for n in range(-2, 3):
                inst_feat[-1].append("C%d==%s" % (n, inst[ind + n][0]))
            # CnCn+1
            for n in range(-2, 2):
                inst_feat[-1].append("C%dC%d==%s%s" % (n, n + 1, inst[ind + n][0], inst[ind + n + 1][0]))
            # C-1C1
            inst_feat[-1].append("C-1C1==%s%s" % (inst[ind - 1][0], inst[ind + 1][0]))
            # Pu(C0)
            inst_feat[-1].append("Pu(%s)==%d" % (c, int(t == 0)))
            # T(C-2)T(C-1)T(C0)T(C1)T(C2)
            inst_feat[-1].append("T-2...2=%d%d%d%d%d" % (
            inst[ind - 2][2], inst[ind - 1][2], inst[ind][2], inst[ind + 1][2], inst[ind + 2][2]))

        return inst_feat

    def __SaveModel(self):
        # the last time to sum all the features.
        norm = float(self.__cur_ite_ID + 1) * self.__insts_num
        for feat in self.__feats_weight_sum:
            last_ite_ID = self.__last_update[feat][0]
            last_inst_ID = self.__last_update[feat][1]
            c = (self.__cur_ite_ID - last_ite_ID) * self.__insts_num + self.__cur_inst_ID - last_inst_ID
            self.__feats_weight_sum[feat] += self.__feats_weight[feat] * c
            self.__feats_weight_sum[feat] = self.__feats_weight_sum[feat] / norm

        pickle.dump(self.__feats_weight_sum, open("data/avgmodel", "wb"))
        self.__train_insts = None

    def __LoadCorp(self, corp_file_name, max_train_num):
        if not os.path.isfile(corp_file_name):
            print("can't open", corp_file_name)
            return False

        self.__train_insts = []
        self.__words_num = 0
        for ind, line in enumerate(open(corp_file_name, "r")):
            if max_train_num > 0 and ind >= max_train_num:
                break
            self.__train_insts.append(self.__PreProcess(line.strip()))
            self.__words_num += len(self.__train_insts[-1]) - 4
        self.__insts_num = len(self.__train_insts)

        print("number of total insts is", self.__insts_num)
        print("number of total characters is", self.__words_num)

        return True

    def __PreProcess(self, sent):
        inst = []
        for i in range(2):
            inst.append(["<s>", "s", -1])
        for word in sent.split():
            rt = word.rpartition("/")
            t = self.__char_type.get(rt[0], 4)
            inst.append([rt[0], rt[2], t])  # [c, tag, t]
        for i in range(2):
            inst.append(["<s>", "s", -1])

        return inst

    def __Segment(self, src):
        """suppose there is one sentence once."""
        inst = []
        for i in range(2):
            inst.append(["<s>", "s", -1])
        for c in src:
            inst.append([c, "", self.__char_type.get(c, 4)])
        for i in range(2):
            inst.append(["<s>", "s", -1])

        feats = self.__GenerateFeats(inst)
        tags = self.__DPSegment(inst, feats)

        rst = []
        for i in range(2, len(tags) - 2):
            if tags[i] in ["s", "b"]:
                rst.append(inst[i][0])
            else:
                rst[-1] += inst[i][0]

        return " ".join(rst)

    def __Iterate(self):
        start = time.clock()
        print("%d th iteration" % self.__cur_ite_ID)

        train_list = random.sample(range(self.__insts_num), self.__insts_num)
        error_sents_num = 0
        error_words_num = 0

        for self.__cur_inst_ID, self.__real_inst_ID in enumerate(train_list):
            num = self.__TrainInstance()
            error_sents_num += 1 if num > 0 else 0
            error_words_num += num

        st = 1 - float(error_sents_num) / self.__insts_num
        wt = 1 - float(error_words_num) / self.__words_num

        end = time.clock()
        print("sents accuracy = %f%%, words accuracy = %f%%, it takes %d seconds" % (st * 100, wt * 100, end - start))

        return error_sents_num == 0 and error_words_num == 0

    def __TrainInstance(self):
        cur_inst = self.__train_insts[self.__real_inst_ID]
        feats = self.__GenerateFeats(cur_inst)

        seg = self.__DPSegment(cur_inst, feats)
        return self.__Correct(seg, feats)

    def __DPSegment(self, inst, feats):
        num = len(inst)

        # get all position's score.
        value = [{} for i in range(num)]
        for i in range(2, num - 2):
            for t in ["b", "m", "e", "s"]:
                value[i][t] = self.__GetScore(i, t, feats)

        # find optimal path.
        tags = [None for i in range(num)]
        best = [-1 for i in range(num)]  # best[i]: [i, i + length(i)) is optimal segment.
        length = [None for i in range(num)]

        for i in range(num - 2 - 1, 1, -1):
            for dis in range(1, 11):
                if i + dis > num - 2:
                    break
                cur_score = best[i + dis]
                self.__Tag(i, i + dis, tags)
                for k in range(i, i + dis):
                    cur_score += value[k][tags[k]]
                if length[i] is None or cur_score > best[i]:
                    best[i] = cur_score
                    length[i] = dis

        i = 2
        while i < num - 2:
            self.__Tag(i, i + length[i], tags)
            i += length[i]

        return tags

    def __GetScore(self, pos, t, feats):
        pos_feats = feats[pos]
        score = 0.0
        for feat in pos_feats:
            score += self.__feats_weight.get(feat + "=>" + t, 0)

        return score

    def __Tag(self, f, t, tags):
        """tag the sequence tags in the xrange of [f, t)"""
        if t - f == 1:
            tags[f] = "s"
        elif t - f >= 2:
            tags[f], tags[t - 1] = "b", "e"
            for i in range(f + 1, t - 1):
                tags[i] = "m"

    def __Correct(self, tags, feats):
        updates = {}
        cur_inst = self.__train_insts[self.__real_inst_ID]
        error_words_num = 0
        for i in range(2, len(cur_inst) - 2):
            if tags[i] == cur_inst[i][1]:
                continue
            error_words_num += 1
            pos_feats = feats[i]
            target = cur_inst[i][1]
            mine = tags[i]
            for feat in pos_feats:
                updates[feat + "=>" + target] = updates.get(feat + "=>" + target, 0.0) + 1
                updates[feat + "=>" + mine] = updates.get(feat + "=>" + mine, 0.0) - 1

        self.__Update(updates)

        return error_words_num

    def __Update(self, updates):
        # update the features weight.
        for feat in updates:
            pair = self.__last_update.get(feat, [0, 0])
            last_ite_ID = pair[0]
            last_inst_ID = pair[1]

            c = (self.__cur_ite_ID - last_ite_ID) * self.__insts_num + self.__cur_inst_ID - last_inst_ID
            self.__feats_weight_sum[feat] = self.__feats_weight_sum.get(feat, 0) + c * self.__feats_weight.get(feat, 0)

            self.__feats_weight[feat] = self.__feats_weight.get(feat, 0) + updates[feat]
            self.__last_update[feat] = [self.__cur_ite_ID, self.__cur_inst_ID]


if __name__ == "__main__":

    train = CPTTrain(train=True, segment=False)
    train.Train("data/msr_train.txt", max_train_num=1000000, max_ite_num=20)

    srcs = ["夏天的清晨",
            "“人们常说生活是一部教科书，而血与火的战争更是不可多得的教科书，她确实是名副其实的‘我的大学’。",
            "夏天的清晨夏天看见猪八戒和嫦娥了。",
            "海运业雄踞全球之首，按吨位计占世界总数的１７％。"]

    print("avg")
    seg = CPTTrain(train=False, segment=True)
    for src in srcs:
        print(seg.Segment(src))
