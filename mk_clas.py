#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import sys
import time
import math
import numpy as np
import scipy.stats as st
from numba import jit
import chainer
import chainer.cuda as cuda
import matplotlib.pyplot as plt
import chainer.functions as F
import os.path as path
import ini
import fxreader
import share as s
import funcs as f

clsNum = 0 # クラス分け方でのクラス数、＋片側の数、－側も同じ数だけあるので実際にクラス数は clsNum * 2 + 1 となる
clsSpan = 0 # クラス分け方でのスパン(pips)
dropoutRatio = 0.5 # ドロップアウト率
offsetMode = None # 切り出した学習エリア値に対するオフセットモード
datasetExtract = None # データセットの抽出方法
fxRetMaSize = 0 # クライアントへ返す第二値の移動平均サイズ
fxRetMaSig = 0 # クライアントへ返す第二値の移動平均（ガウシアン）の標準偏差
fxRetMaSizeK = None

subPlot1 = None
subPlot2 = None
gxIn = None
gyIn = None
gxOut = None
gyOut = None
glIn1 = None
glIn2 = None
glIn3 = None
glIn4 = None
glOut = None
glOutV = None
glTeach = None
glTeachV = None
glErr = None

class Dnn(object):
	"""クラス分類用のモデルとオプティマイザへルパクラス"""
	model = None
	optimizer = None

	def __init__(self, model = None, optimizer = None):
		self.model = model
		self.optimizer = optimizer

	def forward(self, x, volatile=chainer.flag.OFF):
		return self.model(x, volatile)

	def evaluate(self, x, t, volatile=chainer.flag.OFF):
		y = self.model(x, volatile)
		return (y, F.softmax_cross_entropy(y, chainer.Variable(t, volatile=volatile)))

	def update(self, loss):
		self.model.zerograds()
		loss.backward()
		self.optimizer.update()


def readDataset(filename, inMA, noise):
	"""指定された分足為替CSVからロウソク足データを作成する
	Args:
		filename: 読み込むCSVファイル名.
		Returns: 開始値配列、高値配列、低値配列、終値配列の2次元データ
	"""
	return fxreader.readDataset(filename, inMA, noise)

def init(iniFileName):
	"""クラス分類用の初期化を行う"""
	global clsNum
	global clsSpan
	global dropoutRatio
	global offsetMode
	global datasetExtract
	global fxRetMaSize
	global fxRetMaSig
	global fxRetMaSizeK

	configIni = ini.file(iniFileName, "CLAS")
	clsNum = configIni.getInt("clsNum", "3") # クラス分け方でのクラス数、＋片側の数、－側も同じ数だけあるので実際にクラス数は clsNum * 2 + 1 となる
	clsSpan = configIni.getFloat("clsSpan", "3") # クラス分け方でのスパン(pips)
	dropoutRatio = configIni.getFloat("dropoutRatio", "0.5") # ドロップアウト率
	offsetMode = configIni.getStr("offsetMode", "average") # 切り出した学習エリア値に対するオフセットモード
	datasetExtract = configIni.getStr("datasetExtract", "all") # データセットの抽出方法
	fxRetMaSize = configIni.getInt("fxRetMaSize", "5") # クライアントへ返す第二値の移動平均サイズ
	fxRetMaSig = configIni.getInt("fxRetMaSig", "3") # クライアントへ返す第二値の移動平均（ガウシアン）の標準偏差

	# 移動平均（ガウシアン）のカーネル計算
	fxRetMaSize = (fxRetMaSize // 2) * 2 + 1
	interval = fxRetMaSig + (fxRetMaSig + 0.5) / fxRetMaSize
	fxRetMaSizeK = np.diff(st.norm.cdf(np.linspace(-interval, interval, fxRetMaSize + 1)))
	fxRetMaSizeK /= fxRetMaSizeK.sum()

	s.minPredLen = s.frameSize # ドル円未来予測に必要な最小データ数
	s.minEvalLen = s.minPredLen + s.predLen # 学習結果の評価に必要な最小データ数

	# ニューラルネットの入力次元数
	s.dnnIn = s.frameSize 
	# ニューラルネットの出力次元数
	s.dnnOut = clsNum * 2 + 1

	s.fxRetLen = 3 # クライアントに返す結果データ長
	s.fxInitialYenDataLen = s.frameSize # 初期化時にMT4から送る必要がある円データ数

def initGraph(windowCaption):
	global subPlot1
	global subPlot2
	global gxIn
	global gyIn
	global gxOut
	global gyOut
	global glIn1
	global glIn2
	global glIn3
	global glIn4
	global glOut
	global glOutV
	global glTeach
	global glTeachV
	global glErr

	# グラフ描画用の初期化
	if s.grEnable:
		plt.ion() # 対話モードON
		fig = plt.figure() # 何も描画されていない新しいウィンドウを描画
		plt.xlabel("min") # x軸ラベル
		plt.ylabel("yen") # y軸ラベル
		plt.grid() # グリッド表示
		plt.gcf().canvas.set_window_title(windowCaption)

		# 窓は２枠
		subPlot1 = fig.add_subplot(2, 1, 1)
		subPlot2 = fig.add_subplot(2, 1, 2)
		subPlot2.set_xlim([-clsNum, clsNum])
		subPlot2.axhline(y=0, color='black')

		gxOut = np.arange(-clsNum, clsNum + 1, 1)
		gyOut = np.zeros(s.dnnOut)

		if s.mode == "server":
			subPlot1.set_xlim([0, s.minPredLen])
			gxIn = np.arange(0, s.minPredLen, 1)
			gyIn = np.zeros(s.minPredLen)
			glOutV = subPlot2.axvline(x=0, color='red')
		else:
			subPlot1.set_xlim([0, s.minEvalLen])
			subPlot1.axvline(x=s.frameSize, color='black')
			gxIn = np.arange(0, s.minEvalLen, 1)
			gyIn = np.zeros(s.minEvalLen)
			glTeachV = subPlot2.axvline(x=0, color='red')
			glOutV = subPlot2.axvline(x=0, color='orange')

		glIn1, = subPlot1.plot(gxIn, gyIn, label="open")
		glIn2, = subPlot1.plot(gxIn, gyIn, label="high")
		glIn3, = subPlot1.plot(gxIn, gyIn, label="low")
		glIn4, = subPlot1.plot(gxIn, gyIn, label="close")
		glOut, = subPlot2.plot(gxOut, gyOut, label="y")
		
		if s.mode == "testhr" or s.mode == "trainhr":
			glErr, = subPlot2.plot(gxOut, gyOut, label="err", color='red')
			glTeach, = subPlot2.plot(gxOut, gyOut, label="t", color='green')

		subPlot1.legend(loc='lower left') # 凡例表示
		subPlot2.legend(loc='lower left') # 凡例表示

def getTestFileName(testFileName):
	return testFileName + "c" + str(clsNum) + "s" + str(clsSpan) + datasetExtract

#@jit
def trainGetT(dataset, i):
	"""教師データ取得"""
	frameEnd = i + s.frameSize

	# 教師値取得
	# 既知の終値と未来の分足データの開始値との差を教師とする
	last = dataset[3, frameEnd - 1]
	predData = dataset[0, frameEnd : frameEnd + s.predLen]
	if s.predAve:
		p = (predData * s.predMeanK).sum()
	else:
		p = predData[-1, 0]
	d = p - last
	t = int(round(100.0 * d * clsNum / clsSpan, 0))
	if t < -clsNum:
		t = -clsNum
	elif clsNum < t:
		t = clsNum
	t += clsNum

	return t

#@jit
def trainGetXBatchs(model, dataset, batchIndices, toGpu=True):
	"""学習データと教師データをミニバッチで取得"""
	x = model.buildMiniBatchData(dataset, batchIndices)
	return s.toGpu(x) if toGpu else x

#@jit
def trainGetTBatchs(dataset, batchIndices, toGpu=True):
	"""学習データと教師データをミニバッチで取得"""
	t = np.zeros(shape=batchIndices.shape, dtype=np.int32)
	for i, p in enumerate(batchIndices):
		t[i] = trainGetT(dataset, p)
	return s.toGpu(t) if toGpu else t

#@jit
def trainBatch(dataset, itr):
	"""ミニバッチで学習する"""

	# 学習実行
	x = trainGetXBatchs(s.dnn.model, dataset, s.batchStartIndices)
	t = trainGetTBatchs(dataset, s.batchStartIndices)
	y, loss = s.dnn.evaluate(x, t)

	# ユーザー入力による流れ制御
	s.forceEval = False
	f.trainFlowControl()

	# 評価処理
	if (itr % s.evalInterval == 0) or s.forceEval:
		print('evaluate')
		perp = trainEvaluate(dataset, s.evalIndex)
		print('epoch {} validation perplexity: {}'.format(s.curEpoch, perp))
		#if 1 <= itr and s.optm == "Adam":
		#	print('learning rate =', s.dnn.optimizer.lr)

	return loss

#@jit
def trainEvaluate(dataset, index):
	"""現在のニューラルネットワーク評価処理"""

	# モデルに影響を与えないようにコピーする
	model = s.dnn.model.copy()  # to use different state
	model.reset_state()  # initialize state
	model.train = False  # dropout does nothing
	evdnn = Dnn(model, None)

	# 学習データ取得
	batchIndices = np.asarray([index])
	x = trainGetXBatchs(model, dataset, batchIndices)
	t = trainGetTBatchs(dataset, batchIndices)

	# ニューラルネットを通す
	y, loss = evdnn.evaluate(x, t, chainer.flag.ON)

	# 必要ならグラフ表示を行う
	if s.grEnable:
		# グラフにデータを描画する
		plt.title(s.trainDataFile + " : " + str(index)) # グラフタイトル
		xvals = dataset[:, index : index + s.minEvalLen]
		tx = int(t[0])
		ox = y.data.argmax(1)[0]
		if s.xp is np:
			yvals = y.data[0]
		else:
			yvals = cuda.to_cpu(y.data[0])
		glIn1.set_ydata(xvals[0])
		glIn2.set_ydata(xvals[1])
		glIn3.set_ydata(xvals[2])
		glIn4.set_ydata(xvals[3])
		glTeachV.set_xdata([gxOut[tx], gxOut[tx]])
		glOut.set_ydata(yvals)
		glOutV.set_xdata([gxOut[ox], gxOut[ox]])

		subPlot1.set_ylim(f.npMaxMin(xvals))
		subPlot2.set_ylim(f.npMaxMin([yvals]))
		plt.draw()
		plt.pause(0.001)

	try:
		return math.exp(float(loss.data))
	except Exception as e:
		print("evaluate overflow")
		return 0.0

#@jit
def testhr():
	"""指定データを現在のニューラルネットワークを使用し予測値部分の的中率を計測する"""

	print('Hit rate test mode')

	# 学習データ読み込み
	dataset = f.loadDataset()

	## モデルに影響を与えないようにコピーする
	#evaluator = s.dnn.model.copy()  # to use different state
	#evaluator.reset_state()  # initialize state
	#evaluator.train = False  # dropout does nothing

	# モデルを非学習モードにしてそのまま使用する
	model = s.dnn.model
	model.reset_state()  # initialize state
	model.train = False  # dropout does nothing

	testLen = dataset.shape[1] - s.minEvalLen
	hitcount = 0 # 教師と出力が一致した回数
	hitnzcount = 0 # 教師と出力が0以外の時に一致した回数
	sdcount = 0 # 教師と出力が同じ極性だった回数
	distance = 0.0 # 教師値との差

	xvals = dataset
	tvals = np.zeros(testLen, dtype=np.int32)
	yvals = np.zeros(testLen, dtype=np.int32)
	evals = np.zeros(testLen, dtype=np.int32)

	if s.grEnable:
		gxIn = np.arange(0, testLen, 1)
		gxOut = np.arange(0, testLen, 1)
		glTeach.set_xdata(gxOut)
		glTeach.set_ydata(tvals)
		glOut.set_xdata(gxOut)
		glOut.set_ydata(yvals)
		glErr.set_xdata(gxOut)
		glErr.set_ydata(evals)
		glIn1.set_xdata(gxIn)
		glIn1.set_ydata(xvals[0,:testLen])
		glIn2.set_xdata(gxIn)
		glIn2.set_ydata(xvals[1,:testLen])
		glIn3.set_xdata(gxIn)
		glIn3.set_ydata(xvals[2,:testLen])
		glIn4.set_xdata(gxIn)
		glIn4.set_ydata(xvals[3,:testLen])
		subPlot1.set_xlim(0, testLen)
		subPlot1.set_ylim(f.npMaxMin([xvals[0,:testLen], xvals[1,:testLen], xvals[2,:testLen], xvals[3,:testLen]]))
		subPlot2.set_xlim(0, testLen)
		subPlot1.legend(loc='lower left') # 凡例表示
		subPlot2.legend(loc='lower left') # 凡例表示
		plt.title("testhr: " + s.trainDataFile) # グラフタイトル
		plt.draw()
		plt.pause(0.001)

	i = 0
	loop = 0
	while i < testLen:
		# 学習データと教師データ取得
		# バッチ数分まとめて取得する
		n = testLen - i
		if s.batchSize < n:
			n = s.batchSize
		batchIndices = np.arange(i, i + n, 1)
		x = trainGetXBatchs(model, dataset, batchIndices)
		t = trainGetTBatchs(dataset, batchIndices, False)
		# ニューラルネットを通す
		y = model(x, chainer.flag.ON)
		y = s.toCpu(y.data) # 最大値検索処理はCPUのが早い？

		# 的中率計算用＆描画用データにセット
		tvals[i : i + n] = tval = t - clsNum
		yvals[i : i + n] = yval = y.argmax(1) - clsNum
		tyval = tval * yval
		diff = tval - yval
		evals[i : i + n] = np.less(tyval, 0) * diff # 極性が逆の場合のみ誤差波形になるようにする

		# 的中率更新
		i += n
		eqs = np.equal(tval, yval)
		nzs = eqs * np.not_equal(yval, 0)
		hitcount += eqs.sum()
		hitnzcount += nzs.sum()
		sdcount += np.greater(tyval, 0).sum()
		distance += float((diff ** 2).sum())

		print("{0}: {1:.2f}%, {2:.2f}%, {3:.2f}%, rms err {4:.2f}".format(i, 100.0 * hitcount / i, 100.0 * hitnzcount / i, 100.0 * sdcount / i, math.sqrt(distance / i)))

		if loop % 100 == 0 or testLen <= i:
			# 指定間隔または最終データ完了後に
			# グラフにデータを描画する

			if testLen <= i:
				# 最終データ完了後なら
				# xvals の平均値にt、yが近づくよう調整してCSVファイルに吐き出す
				xvalsAverage = np.average(xvals)
				scale = clsSpan / (clsNum * 100.0)
				tvals = tvals * scale
				yvals = yvals * scale
				evals = evals * scale
				tvals += xvalsAverage
				yvals += xvalsAverage
				evals += xvalsAverage
				f.writeTestHrGraphCsv(xvals, tvals, yvals)

			if s.grEnable:
				glTeach.set_ydata(tvals)
				glOut.set_ydata(yvals)
				glErr.set_ydata(evals)
				subPlot2.set_ylim(f.npMaxMin([tvals[:i], yvals[:i]]))
				plt.draw()
				plt.pause(0.001)

		loop += 1

	hitRate = 100.0 * hitcount / testLen
	nzhitRate = 100.0 * hitnzcount / testLen
	sdRate = 100.0 * sdcount / testLen
	distance = math.sqrt(distance / testLen)
	print("{0:.2f}%, {1:.2f}%, {2:.2f}%, rms err {3:.2f}".format(hitRate, nzhitRate, sdRate, distance))

	f.writeTestHrStatCsv(s.curEpoch, hitRate, nzhitRate, sdRate, distance)

	if s.grEnable:
		plt.ioff() # 対話モードOFF
		plt.show()

#@jit
def fxPrediction():
	"""現在の円データから予測する"""

	## 必要があるなら学習を行う
	#if s.serverTrainCount != 0:
	#	# 学習用変数初期化
	#	f.serverTrainInit(s.fxYenData.shape[0])
	#	s.dnn.model.train = True

	#	# 指定回数分移動させながら学習させる
	#	for i in range(s.serverTrainCount):
	#		# バッチ位置初期化
	#		s.batchStartIndices = np.asarray(s.batchIndices, dtype=np.integer)
	#		s.batchStartIndices += s.batchOffset
	#		# 学習実行
	#		x, t = trainGetBatchs(s.fxYenData)
	#		y, loss = s.dnn.forward(x, t, True)
	#		# ニューラルネットワーク更新
	#		s.dnn.update(loss)
	#		s.batchOffset -= 1

	# モデル取得
	model = s.dnn.model
	model.train = False
	# 予測元データ取得してニューラルネットを通す
	dataset = s.fxYenData[-s.dnnIn:,0]
	x = trainGetXBatchs(model, dataset, np.asarray([0]))
	y = model(x, chainer.flag.ON)
	y = s.toCpu(y.data) # 最大値検索処理はCPUのが早い？
	ox = y.argmax(1)[0]

	# 必要ならグラフ表示を行う
	if s.grEnable:
		# グラフにデータを描画する
		xvals = s.fxYenData[:, -s.frameSize:]
		glIn1.set_ydata(xvals[0])
		glIn2.set_ydata(xvals[1])
		glIn3.set_ydata(xvals[2])
		glIn4.set_ydata(xvals[3])
		glOut.set_ydata(yvals)
		glOutV.set_xdata([gxOut[ox], gxOut[ox]])

		subPlot1.set_ylim(f.npMaxMin([xvals]))
		subPlot2.set_ylim(f.npMaxMin([yvals]))
		plt.draw()
		plt.pause(0.001)

	# 戻り値配列作成
	# AI予測値
	# 移動平均の差分値
	# 移動平均の差分値の差分値
	deltaPips = float(clsSpan * (ox - clsNum) / clsNum)
	dataForMa = s.fxYenData[3, -fxRetMaSize - 3:]
	ma = np.asarray(np.convolve(np.asarray(dataForMa), fxRetMaSizeK, 'valid'), dtype=np.float32)
	diff1 = np.diff(ma)
	diff2 = np.diff(diff1)
	return np.asarray([deltaPips, diff1[-1] * 100.0, diff2[-1] * 100.0], dtype=np.float32)
