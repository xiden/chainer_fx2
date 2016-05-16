#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import time
import math
import csv
import codecs
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
fxRetMaSize = 0 # クライアントへ返す第二値の移動平均サイズ
fxRetMaSig = 0 # クライアントへ返す第二値の移動平均（ガウシアン）の標準偏差
fxRetMaSizeK = None

subPlot1 = None
subPlot2 = None
gxIn = None
gyIn = None
gxOut = None
gyOut = None
glIn = None
glOut = None
glOutV = None
glTeachV = None
glErr = None

class Dnn(object):
	"""クラス分類用のモデルとオプティマイザへルパクラス"""
	model = None
	optimizer = None

	def __init__(self, model = None, optimizer = None):
		self.model = model
		self.optimizer = optimizer

	def forward(self, x, t, calcLoss):
		y = self.model(x)
		if calcLoss:
			return (y, F.softmax_cross_entropy(y, t))
		else:
			return (y, None)

	def update(self, loss):
		self.model.zerograds()
		loss.backward()
		self.optimizer.update()


def read(filename, inMA):
	"""指定された分足為替CSVからロウソク足データを作成する
	Args:
		filename: 読み込むCSVファイル名.
		Returns: [int]
	"""
	return fxreader.read(filename, inMA)

def init(iniFileName):
	"""クラス分類用の初期化を行う"""
	global clsNum
	global clsSpan
	global dropoutRatio
	global fxRetMaSize
	global fxRetMaSig
	global fxRetMaSizeK

	configIni = ini.file(iniFileName, "CLAS")
	clsNum = configIni.getInt("clsNum", "3") # クラス分け方でのクラス数、＋片側の数、－側も同じ数だけあるので実際にクラス数は clsNum * 2 + 1 となる
	clsSpan = configIni.getFloat("clsSpan", "3") # クラス分け方でのスパン(pips)
	dropoutRatio = configIni.getFloat("dropoutRatio", "0.5") # ドロップアウト率
	fxRetMaSize = configIni.getInt("fxRetMaSize", "5") # クライアントへ返す第二値の移動平均サイズ
	fxRetMaSig = configIni.getInt("fxRetMaSig", "3") # クライアントへ返す第二値の移動平均（ガウシアン）の標準偏差

	# 移動平均（ガウシアン）のカーネル計算
	fxRetMaSize = (fxRetMaSize // 2) * 2 + 1
	interval = fxRetMaSig + (fxRetMaSig + 0.5) / fxRetMaSize
	fxRetMaSizeK = np.diff(st.norm.cdf(np.linspace(-interval, interval, fxRetMaSize + 1)))
	fxRetMaSizeK /= fxRetMaSizeK.sum()

	s.minPredLen = s.frameSize # ドル円未来予測に必要な最小データ数
	s.minEvalLen = s.minPredLen + s.predLen # 学習結果の評価に必要な最小データ数
	s.n_in = s.frameSize # ニューラルネットの入力次元数
	s.n_out = clsNum * 2 + 1 # ニューラルネットの出力次元数
	s.fxRetLen = 3 # クライアントに返す結果データ長
	s.fxInitialYenDataLen = s.frameSize # 初期化時にMT4から送る必要がある円データ数

def initGraph(windowCaption):
	global subPlot1
	global subPlot2
	global gxIn
	global gyIn
	global gxOut
	global gyOut
	global glIn
	global glOut
	global glOutV
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
		gyOut = np.zeros(s.n_out)

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

		glIn, = subPlot1.plot(gxIn, gyIn, label="in")
		glOut, = subPlot2.plot(gxOut, gyOut, label="out")
		
		if s.mode == "testhr":
			glErr, = subPlot2.plot(gxOut, gyOut, label="err", color='red')

def getTestFileName(testFileName):
	return testFileName + "c" + str(clsNum) + "s" + str(clsSpan)

#@jit
def trainGetDataAndT(dataset, i):
	"""学習データと教師データ取得"""
	frameEnd = i + s.frameSize

	# 教師値取得
	if s.predAve:
		t = (dataset[frameEnd : frameEnd + s.predLen] * s.predMeanK).sum()
	else:
		t = dataset[frameEnd + s.predLen - 1]
	t = int(round(100.0 * float(t - dataset[frameEnd - 1]) * clsNum / clsSpan, 0))
	if t < -clsNum:
		t = -clsNum
	elif clsNum < t:
		t = clsNum
	t += clsNum

	# フレーム取得
	# フレーム内の平均値または中央値が0になるようシフトする
	x = dataset[i : frameEnd]
	x = x - np.median(x)
	#x = x - (x.sum() / s.frameSize)
	return x, t

#@jit
def trainGetData(dataset, i):
	"""学習データのみ取得"""
	# フレーム取得
	# フレーム内の平均値または中央値が0になるようシフトする
	x = dataset[i : i + s.frameSize]
	x = x - np.median(x)
	#x = x - (x.sum() / s.frameSize)
	return x

#@jit
def trainGetBatchs(dataset):
	# 学習データと教師データ取得
	xa_cpu = np.zeros(shape=(s.batchSize, s.n_in), dtype=np.float32)
	ta_cpu = np.zeros(shape=(s.batchSize,), dtype=np.int32)
	for bi in range(s.batchSize):
		xa_cpu[bi][...], ta_cpu[bi] = trainGetDataAndT(dataset, s.batchStartIndices[bi])
	return chainer.Variable(cuda.to_gpu(xa_cpu)), chainer.Variable(cuda.to_gpu(ta_cpu))

#@jit
def trainBatch(dataset, itr):
	"""ミニバッチで学習する"""

	# 学習実行
	x, t = trainGetBatchs(dataset)
	y, loss = s.dnn.forward(x, t, True)

	# ユーザー入力による流れ制御
	s.forceEval = False
	f.trainFlowControl()

	# 評価処理
	if (itr % s.evalInterval == 0) or s.forceEval:
		print('evaluate')
		now = time.time()
		perp = trainEvaluate(dataset, s.evalIndex)
		print('epoch {} validation perplexity: {}'.format(s.curEpoch, perp))
		#if 1 <= itr and s.optm == "Adam":
		#	print('learning rate =', s.dnn.optimizer.lr)

	return loss

#@jit
def trainEvaluate(dataset, index):
	"""現在のニューラルネットワーク評価処理"""

	# モデルに影響を与えないようにコピーする
	evaluator = s.dnn.model.copy()  # to use different state
	evaluator.reset_state()  # initialize state
	evaluator.train = False  # dropout does nothing
	evdnn = Dnn(evaluator, None)

	# 学習データ取得
	xa, ta = trainGetDataAndT(dataset, index)
	xa = s.xp.asarray([xa], dtype=np.float32)
	ta = s.xp.asarray([ta], dtype=np.int32)
	x = chainer.Variable(xa, volatile='on')
	t = chainer.Variable(ta, volatile='on')

	# ニューラルネットを通す
	y, loss = evdnn.forward(x, t, True)

	# 必要ならグラフ表示を行う
	if s.grEnable:
		# グラフにデータを描画する
		plt.title(s.trainDataFile + " : " + str(index)) # グラフタイトル
		xvals = cuda.to_cpu(dataset[index : index + s.minEvalLen])
		tx = int(t.data[0])
		ox = y.data.argmax(1)[0]
		yvals = np.asarray(y.data[0].tolist(), dtype=np.float32)
		glIn.set_ydata(xvals)
		glTeachV.set_xdata([gxOut[tx], gxOut[tx]])
		glOut.set_ydata(yvals)
		glOutV.set_xdata([gxOut[ox], gxOut[ox]])

		subPlot1.set_ylim(f.npMaxMin(xvals))
		subPlot2.set_ylim(f.npMaxMin(yvals))
		plt.draw()
		plt.pause(0.001)

	try:
		return math.exp(float(loss.data))
	except Exception as e:
		print("evaluate overflow")
		return 0.0

def writeTestHrCsv(xvals, tvals, yvals):
	"""テスト結果CSVファイルに書き込む"""
	fname = path.join(s.resultDir, f.getTestHrFileBase() + str(s.curEpoch) + ".csv")
	with codecs.open(fname, 'w', "shift_jis") as file:
		writer = csv.writer(file)
		for i in range(xvals.shape[0]):
			writer.writerow([xvals[i], tvals[i], yvals[i]])

#@jit
def testhr():
	"""指定データを現在のニューラルネットワークを使用し予測値部分の的中率を計測する"""

	print('Hit rate test mode')
	print("Loading data from  " + s.trainDataFile)
	dataset = s.mk.read(s.trainDataFile, s.inMA)
	index = 0

	# モデルに影響を与えないようにコピーする
	evaluator = s.dnn.model.copy()  # to use different state
	evaluator.reset_state()  # initialize state
	evaluator.train = False  # dropout does nothing

	testPos = 0
	testLen = dataset.shape[0] - s.minEvalLen
	count = 0
	hitcount = 0
	zero = np.zeros(s.batchSize, dtype=np.float32)

	if s.grEnable:
		xvals = np.zeros(testLen, dtype=np.float32)
		tvals = np.zeros(testLen, dtype=np.int32)
		yvals = np.zeros(testLen, dtype=np.int32)
		evals = np.zeros(testLen, dtype=np.int32)

		gxIn = np.arange(0, xvals.shape[0], 1)
		gxOut = np.arange(0, testLen, 1)
		glIn.set_xdata(gxIn)
		glOut.set_xdata(gxOut)
		glErr.set_xdata(gxOut)
		subPlot1.set_xlim(0, xvals.shape[0])
		subPlot2.set_xlim(0, testLen)

	i = 0
	loop = 0
	while i < testLen:
		# 学習データと教師データ取得
		# バッチ数分まとめて取得する
		n = testLen - i
		if s.batchSize < n:
			n = s.batchSize
		xa_cpu = np.zeros(shape=(n, s.n_in), dtype=np.float32)
		ta_cpu = np.zeros(shape=(n,), dtype=np.int32)
		for bi in range(n):
			xa_cpu[bi][...], ta_cpu[bi] = trainGetDataAndT(dataset, i + bi)
		# ニューラルネットを通す
		y = evaluator(chainer.Variable(cuda.to_gpu(xa_cpu), volatile='on'))
		y = cuda.to_cpu(y.data)

		# 描画用データにセット
		xvals[i : i + n] = dataset[i : i + n]
		tvals[i : i + n] = tval = ta_cpu - clsNum
		yvals[i : i + n] = yval = y.argmax(1) - clsNum
		evals[i : i + n] = np.less(tval * yval, 0) * (tval - yval)

		# 的中率更新
		i += n
		hitcount += np.equal(tval, yval).sum()
		print(i, ": ", 100.0 * hitcount / i, "%")

		if loop % 10 == 0 or testLen <= i:
			# 指定間隔または最終データ完了後に
			# グラフにデータを描画する
			plt.title("testhr: " + s.trainDataFile) # グラフタイトル

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
				writeTestHrCsv(xvals, tvals, yvals)

			glIn.set_ydata(xvals)
			glOut.set_ydata(yvals)
			glErr.set_ydata(evals)

			subPlot1.set_ylim(f.npMaxMin([xvals[:i]]))
			subPlot2.set_ylim(f.npMaxMin([tvals[:i], yvals[:i]]))
			plt.draw()
			plt.pause(0.001)

		loop += 1

	result = 100.0 * hitcount / testLen
	print("result: ", result, "%")

	section = s.trainDataFile
	testFileIni = ini.file(s.testFilePath + ".ini", section)
	testFileIni.set("hitRate" + str(s.curEpoch), result)
	testFileIni.setSection(section + "_DEFAULT" + str(s.curEpoch), s.configIni.getSectionCommentRemove("DEFAULT"))
	testFileIni.setSection(section + "_CLAS" + str(s.curEpoch), s.configIni.getSectionCommentRemove("CLAS"))

	if s.grEnable:
		plt.ioff() # 対話モードOFF
		plt.show()

#@jit
def fxGetData(dataset):
	"""予測元データ取得"""
	# フレーム取得
	x = dataset[-s.frameSize:]
	# フレーム内の平均値を0になるようシフトする
	x = x - (x.sum() / s.frameSize)
	return s.xp.asarray([x], dtype=np.float32)

##@jit
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
	pred = s.dnn.model
	pred.train = False
	# 予測元データ取得してニューラルネットを通す
	y = pred(chainer.Variable(fxGetData(s.fxYenData), volatile='on'))
	yvals = cuda.to_cpu(y.data[0]) # np.asarray(y.data[0].tolist(), dtype=np.float32)
	ox = y.data.argmax(1)[0]

	# 必要ならグラフ表示を行う
	if s.grEnable:
		# グラフにデータを描画する
		xvals = s.fxYenData[-s.frameSize:]
		glIn.set_ydata(xvals)
		glOut.set_ydata(yvals)
		glOutV.set_xdata([gxOut[ox], gxOut[ox]])

		subPlot1.set_ylim(f.npMaxMin(xvals))
		subPlot2.set_ylim(f.npMaxMin(yvals))
		plt.draw()
		plt.pause(0.001)

	# 戻り値配列作成
	# AI予測値
	# 移動平均の差分値
	# 移動平均の差分値の差分値
	deltaPips = float(clsSpan * (ox - clsNum) / clsNum)
	ma = np.asarray(np.convolve(np.asarray(s.fxYenData[-fxRetMaSize - 3:]), fxRetMaSizeK, 'valid'), dtype=np.float32)
	diff1 = np.diff(ma)
	diff2 = np.diff(diff1)
	return np.asarray([deltaPips, diff1[-1] * 100.0, diff2[-1] * 100.0], dtype=np.float32)
