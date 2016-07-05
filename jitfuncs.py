#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import numpy as np
import scipy.stats as st
from numba import jit

@jit("void(f4[:,:])", nopython=True)
def normalizeAfterNoise(a):
	"""
	開始、高値、低値、終値に乱数を加えた後の正規化処理.
	"""
	n = a.shape[0]
	for i in range(n):
		a[i, 1] = a[i].max()
		a[i, 2] = a[i].min()

#@jit("f8[:](i8, f8)", nopython=True)
def gaussianKernel(maSize, sigma):
	"""
	移動平均用ガウシアンカーネルの計算.

	Args:
		maSize: 移動平均サイズ.
		sigma: σ値.

	Returns:
		ガウシアンカーネル.
	"""
	maSize = (maSize // 2) * 2 + 1
	interval = sigma + (sigma + 0.5) / maSize
	k = np.diff(st.norm.cdf(np.linspace(-interval, interval, maSize + 1)))
	k /= k.sum()
	return k

@jit("i4[:](f4[:,:], i8, i8, f8, i8, f4[:,:])", nopython=True)
def makeConvolutedPredData(trainDataset, frameSize, clsNum, clsSpan, predLen, predMeanK):
	"""
	指定された学習用データセットから未来データの畳み込みをし、教師データセットを作成する.

	Args:
		trainDataset: 学習用データセット.
		frameSize: １回の処理で使用するデータ数.
		clsNum: 片側分類分け数.
		clsSpan: clsNum に対応する pips.
		predLen: 予測長.
		predMeanK: 畳み込み時の係数.

	Returns:
		教師データセット.
	"""
	n = trainDataset.shape[1] - frameSize - predLen + 1
	dataset = np.empty(n, dtype=np.int32)
	rate = 100.0 * clsNum / clsSpan

	for i in range(n):
		frameEnd = i + frameSize

		# 教師値取得
		# 既知の終値と未来の分足データの開始値との差を教師とする
		d = (trainDataset[0, frameEnd : frameEnd + predLen] * predMeanK).sum() - trainDataset[3, frameEnd - 1]
		t = int(round(d * rate, 0))
		if t < -clsNum:
			t = -clsNum
		elif clsNum < t:
			t = clsNum
		t += clsNum
		dataset[i] = t

	return dataset

@jit("i4[:](f4[:,:], i8, i8, f8, i8)", nopython=True)
def makePredData(trainDataset, frameSize, clsNum, clsSpan, predLen):
	"""
	指定された学習用データセットから未来データを取得し教師データセットを作成する.

	Args:
		trainDataset: 学習用データセット.
		frameSize: １回の処理で使用するデータ数.
		clsNum: 片側分類分け数.
		clsSpan: clsNum に対応する pips.
		predLen: 予測長.

	Returns:
		教師データセット.
	"""
	n = trainDataset.shape[1] - frameSize - predLen + 1
	dataset = np.empty(n, dtype=np.int32)
	rate = 100.0 * clsNum / clsSpan

	for i in range(n):
		frameEnd = i + frameSize

		# 教師値取得
		# 既知の終値と未来の分足データの開始値との差を教師とする
		d = trainDataset[0, frameEnd + predLen - 1] - trainDataset[3, frameEnd - 1]
		t = int(round(d * rate, 0))
		if t < -clsNum:
			t = -clsNum
		elif clsNum < t:
			t = clsNum
		t += clsNum
		dataset[i] = t

	return dataset

@jit("f4[:,:,:](i8, f4[:,:], i4[:])", nopython=True)
def buildMiniBatchCloseHighLow(inCount, trainDataset, batchIndices):
	"""
	学習データセットの終値、高値、低値を使い指定位置から全ミニバッチデータを作成する.

	Args:
		inCount: ニューラルネットの入力次元数.
		trainDataset: 学習データセット.
		batchIndices: 各バッチの学習データセット開始インデックス番号.

	Returns:
		全バッチ分の学習データセット.
	"""
	batchSize = batchIndices.shape[0]
	x = np.empty(shape=(3, batchSize, inCount), dtype=np.float32)
	for i, p in enumerate(batchIndices):
		pe = p + inCount
		x[0,i,:] = trainDataset[3,p:pe]
		x[1,i,:] = trainDataset[1,p:pe]
		x[2,i,:] = trainDataset[2,p:pe]
	return x
