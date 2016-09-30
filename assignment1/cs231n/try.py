#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 16/9/15 下午3:01
# @Author  : Tianyu Liu


from data_utils import *

Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar-10/') # a magic function we provide
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072

num_test = Xte_rows.shape[0]

print(np.abs(Xtr_rows - Xte_rows[4, :]))

tDot = np.multiply(np.dot(Xte_rows, Xtr_rows.T), -2)
t1 = np.sum(np.square(Xte_rows), axis=1, keepdims=True)
t2 = np.sum(np.square(Xtr_rows), axis=1, keepdims=True).T
print(tDot.shape)
print(t1.shape)
print(t2.shape)
tDot = np.add(t1, tDot)
print(tDot.shape)
tDot = np.add(tDot, t2)
dists = np.sqrt(tDot)

