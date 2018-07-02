#coding=utf-8
import caffe
import numpy as np
import math


class EuclideanLossLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num


class DLMCLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        self._batch_size = bottom[1].num
        self._p = 0.2
        self._alpha = 0.01

    def reshape(self, bottom, top):
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(1)

    def forward(self, bottom, top):
        loss = 0
        #求batch中类别的个数
        all_label_value = []
        for i in range(self._batch_size):
            all_label_value.append(bottom[1].data[i])
        set_label = set(all_label_value)
        list_label = list(set_label)
        num_class_batch = len(list_label)
        
        for i in range(self._batch_size):
            #当前样本的label,属于该类的概率
            label_i = bottom[1].data[i]
            prob_i = bottom[0].data[i][int(label_i)]
            #求batch中类别的个数
            inter_i = 0
            #记录当前样本属于其他类别的概率
            i_index_cos = {}
            for j in range(num_class_batch):
                i_index_cos[str(int(list_label[j]))] = bottom[0].data[i][int(list_label[j])]
            #求其它类中参与计算的样本个数
            p_inter_i = int(math.floor(self._p * num_class_batch))

            #排序
            sum_exp_inter = 0
            index_inter = []   #id of xj
            costh_inter = []   # exp(xj/p)/p
            if p_inter_i > 0:
                sort_cos = sorted(i_index_cos.items(),key=lambda e:e[1],reverse=True)
                for index, iterm in enumerate(sort_cos):
                    cos_t_i = iterm[1] / p_inter_i
                    index_inter.append(int(iterm[0]))
                    exp_cos_t_i = np.exp(cos_t_i)
                    costh_inter.append(exp_cos_t_i / p_inter_i)
                    sum_exp_inter += exp_cos_t_i
                    if index == p_inter_i-1:
                        break
            #求对数
            ln_inter = 0
            i_loss = 0
            if p_inter_i > 0:
                ln_inter = math.log(sum_exp_inter)
                #计算loss
                i_loss = ln_inter - (prob_i - self._alpha)

            # loss累加正数部分
            if i_loss > 0:
                #当前样本第label_i维度的梯度
                self.diff[i][int(label_i)] = -1.0
                #参与loss计算的当前样本第index_inter[]维度的梯度
                for k in range(p_inter_i):
                    self.diff[i][index_inter[k]] += costh_inter[k] / sum_exp_inter
                loss += i_loss

        top[0].data[...] = loss / self._batch_size

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = self.diff  / self._batch_size


class CosinLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        self._batch_size = bottom[1].num

    def reshape(self, bottom, top):
        self.flag = np.zeros((self._batch_size, 1), dtype=np.float32)
        self.diff = np.zeros((self._batch_size, 1), dtype=np.float32)
        top[0].reshape(1)

    def find_same_batch(self, bottom):
        self.allpair = []
        Choice = np.zeros((self._batch_size, 1), dtype=np.float32)
        for i in range(self._batch_size):
            if Choice[i] == 1:
                continue
            pair = []
            for j in range(i + 1, self._batch_size):
                if bottom[1].data[i] == bottom[1].data[j] and Choice[j] == 0:
                    pair.append(i)
                    pair.append(j)
                    Choice[i] = 1
                    Choice[j] = 1
                    self.allpair.append(pair)
                    break
        for i in range(self._batch_size):
            if Choice[i] == 1:
                continue
            pair = []
            for j in range(i + 1, self._batch_size):
                if Choice[j] == 0:
                    pair.append(i)
                    pair.append(j)
                    Choice[i] = 1
                    Choice[j] = 1
                    self.allpair.append(pair)
                    break


    def forward(self, bottom, top):
        self.find_same_batch(bottom)
        for pair in self.allpair:
            dot_product = 0.0;
            for j in range(512):
                dot_product += bottom[0].data[pair[0]][j] * bottom[0].data[pair[1]][j]

            if bottom[1].data[pair[0]] == bottom[1].data[pair[1]]:
                self.flag[pair[0]] = 0
                self.diff[pair[0]] = 1 - dot_product
            else:
                self.flag[pair[0]] = 1
                if dot_product > 0:
                    self.diff[pair[0]] = dot_product
                else:
                    self.diff[pair[0]] = 0

        top[0].data[...] = np.sum(self.diff) / (self._batch_size * 0.5)


    def backward(self, top, propagate_down, bottom):
        for pair in self.allpair:
            if self.flag[pair[0]] == 0:
                bottom[0].diff[pair[0]] = -1 * bottom[0].data[pair[1]]
                bottom[0].diff[pair[1]] = -1 * bottom[0].data[pair[0]]

            if self.flag[pair[0]] == 1:
                if self.diff[pair[0]] != 0:
                    bottom[0].diff[pair[0]] = bottom[0].data[pair[1]]
                    bottom[0].diff[pair[1]] = bottom[0].data[pair[0]]

                else:
                    bottom[0].diff[pair[0]] = np.zeros_like(bottom[0].diff[pair[0]], dtype=np.float32)
                    bottom[0].diff[pair[1]] = np.zeros_like(bottom[0].diff[pair[1]], dtype=np.float32)
