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

clsNum = 1 # クラス分け方でのクラス数、＋片側の数、－側も同じ数だけあるので実際にクラス数は clsNum * 2 + 1 となる
zigzagTolerance = 0.02 # ジグザグ判定用許容誤差
zigzagExpand = 5 # 教師データジグザグ位置前後にも同じ値を設定する範囲（片範囲数）
zigzagMark = 0 # 学習データの高値と低値にジグザグをマーキングする
zigzagTrain = 0 # 学習データにジグザグデータ行を追加する

zigzagCache = None

subPlot1 = None
subPlot2 = None
gxIn = None
gyIn = None
gxZigzag = None
gyZigzag = None
gxOut = None
gyOut = None
glIn1 = None
glIn2 = None
glIn3 = None
glIn4 = None
glZigzag = None
glOut = None
glOutV = None
glTeach = None
glTeachV = None
glErr = None

class Dnn(object):
	"""ジグザグ用のモデルとオプティマイザへルパクラス"""
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

@jit("Tuple((i4[:],f4[:]))(f4[:,:],f4)", nopython=True)
def makeZigzag(dataset, tolerance):
	"""
	指定データから山谷位置リストを作成する.

	Args:
		dataset: 開始値、高値、低値、終値の２次元配列.
		tolerance: 山谷判定許容誤差.

	Returns:
		山谷位置リスト[位置, 値].
	"""

	n = dataset.shape[1]
	cvals = dataset[3]
	positions = []
	values = []

	# 最大最小位置を初期化する
	pos = 0
	hval = lval = cvals[0]
	dir = 0

	# ループで最大最小位置探しながら山谷を設定していく
	for i in range(1, n):
		c = cvals[i]

		if hval < c:
			# 最大値更新
			hval = c
			if tolerance < hval - lval:
				lval = hval - tolerance
				if dir <= 0:
					positions.append(pos)
					values.append(cvals[pos])
					dir = 1
			pos = i
		elif c < lval:
			# 最小値更新
			lval = c
			if tolerance < hval - lval:
				hval = lval + tolerance
				if 0 <= dir:
					positions.append(pos)
					values.append(cvals[pos])
					dir = -1
			pos = i

	return (np.array(positions, dtype=np.int32), np.array(values, dtype=np.float32))

@jit("i4[:](i4[:], f4[:], i8, i8, i8)", nopython=True)
def makeTeachDataFromZigzag(positions, values, length, minEvalLen, expand):
	"""
	ジグザグポジションから教師データを作成する.

	Args:
		positions: ジグザグ位置インデックス.
		values: ジグザグ位置の値.
		length: 学習データ長.
		minEvalLen: １回の学習に必要な最小データ数.
		expand: ジグザグの値を前後に広げる数.

	Returns:
		教師データ.
	"""
	n = length - minEvalLen + 1
	result = np.ones(n, dtype=np.int32)
	count = len(positions)

	for i in range(1, count):
		p = positions[i] - minEvalLen
		t = 2 if values[i] < values[i - 1] else 0
		for i in range(p - expand, p + expand + 1):
			if 0 <= i and i < n:
				result[i] = t

	return result

@jit("void(i4[:], f4[:], f4[:,:])", nopython=True)
def markZigzagToTrainData(positions, values, trainDataset):
	"""
	学習データにジグザグをマーキングする、山部分は高値を1円上げ、谷部分は低値を1円下げる.

	Args:
		positions: ジグザグ位置インデックス.
		values: ジグザグ位置の値.
		trainData: 学習データ.
	"""
	count = len(positions)

	for i in range(1, count):
		p = positions[i]
		if values[i] < values[i - 1]:
			trainDataset[2, p] -= 1
		else:
			trainDataset[1, p] += 1

@jit("f4[:](i4[:], f4[:], i8)", nopython=True)
def getZigzagTrainData(positions, values, length):
	"""
	学習データ追加するジグザグデータ行を取得する.

	Args:
		positions: ジグザグ位置インデックス.
		values: ジグザグ位置の値.
		length: 学習データ長.

	Returns:
		学習データに追加するジグザグデータ.
	"""
	count = len(positions)
	zigzag = np.zeros(length, dtype=np.float32)

	for i in range(1, count):
		zigzag[positions[i]] = 1 if values[i] < values[i - 1] else -1

	return zigzag

def addZigzagLine():
	"""
	入力側グラフ領域にジグザグラインを追加する.
	"""
	global glZigzag
	if glZigzag is None:
		glZigzag, = subPlot1.plot(gxZigzag, gyZigzag, label="zigzag", marker="o", markersize=8)

def readDataset(filename, inMA, noise):
	"""
	指定された分足為替CSVからロウソク足データを作成する

	Args:
		filename: 読み込むCSVファイル名.
		inMA: 移動平均サイズ.
		noise: 加えるノイズ量.

	Returns:
		開始値配列、高値配列、低値配列、終値配列の2次元データ.
	"""
	global zigzagCache
	global gxZigzag
	global gyZigzag

	trainDataset = fxreader.readDataset(filename, inMA, noise)

	# ジグザグ取得
	if zigzagCache is None:
		zigzagCache = makeZigzag(trainDataset, zigzagTolerance)

	# グラフ描画用として保持しておく
	if s.grEnable:
		gxZigzag = zigzagCache[0]
		gyZigzag = zigzagCache[1]

	# 指定されていたら学習データにジグザグデータを追加する
	if zigzagTrain:
		zigzag = getZigzagTrainData(zigzagCache[0], zigzagCache[1], trainDataset.shape[1])
		trainDataset = np.vstack((trainDataset, zigzag))

	return trainDataset


def makeTeachDataset(trainDataset):
	"""
	指定された学習用データセットから教師データセットを作成する.

	Args:
		trainDataset: 学習用データセット.

	Returns:
		教師データセット.
	"""
	# 指定されていたら学習データにジグザグをマーキングする
	if zigzagMark:
		markZigzagToTrainData(zigzagCache[0], zigzagCache[1], trainDataset)

	return makeTeachDataFromZigzag(zigzagCache[0], zigzagCache[1], trainDataset.shape[1], s.minEvalLen, zigzagExpand)

def init(iniFileName):
	"""クラス分類用の初期化を行う"""
	global clsNum
	global zigzagTolerance
	global zigzagExpand
	global zigzagMark
	global zigzagTrain

	configIni = ini.file(iniFileName, "ZIGZAG")
	clsNum = configIni.getInt("clsNum", "1") # クラス分け方でのクラス数、＋片側の数、－側も同じ数だけあるので実際にクラス数は clsNum * 2 + 1 となる
	zigzagTolerance = configIni.getFloat("zigzagTolerance", 0.02) # ジグザグ判定用許容誤差
	zigzagExpand = configIni.getFloat("zigzagExpand", 5) # 教師データジグザグ位置前後にも同じ値を設定する範囲（片範囲数）
	zigzagMark = configIni.getInt("zigzagMark", 0) # 学習データの高値と低値にジグザグをマーキングする
	zigzagTrain = configIni.getInt("zigzagTrain", 0) # 学習データにジグザグデータ行を追加する

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
		subPlot2.set_xlim([-clsNum - 1, clsNum + 1])
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
	return testFileName + "tole" + str(zigzagTolerance) + "exp" + str(zigzagExpand)

#@jit
def trainGetXBatchs(model, dataset, batchIndices, toGpu=True):
	"""
	学習データと教師データをミニバッチで取得
	"""
	x = model.buildMiniBatchData(dataset, batchIndices)
	return s.toGpu(x) if toGpu else x

#@jit
def trainGetTBatchs(dataset, batchIndices, toGpu=True):
	"""
	学習データと教師データをミニバッチで取得
	"""
	t = np.take(dataset, batchIndices)
	return s.toGpu(t) if toGpu else t

#@jit
def trainBatch(trainDataset, teachDataset, itr):
	"""
	ミニバッチで学習する
	"""

	# 学習実行
	x = trainGetXBatchs(s.dnn.model, trainDataset, s.batchStartIndices)
	t = trainGetTBatchs(teachDataset, s.batchStartIndices)
	y, loss = s.dnn.evaluate(x, t)

	# ユーザー入力による流れ制御
	s.forceEval = False
	f.trainFlowControl()

	# 評価処理
	if (itr % s.evalInterval == 0) or s.forceEval:
		print('evaluate')
		perp = trainEvaluate(trainDataset, teachDataset, s.evalIndex)
		print('epoch {} validation perplexity: {}'.format(s.curEpoch, perp))

	return loss

#@jit
def trainEvaluate(trainDataset, teachDataset, index):
	"""
	現在のニューラルネットワーク評価処理
	"""

	# モデルに影響を与えないようにコピーする
	model = s.dnn.model.copy()  # to use different state
	model.reset_state()  # initialize state
	model.train = False  # dropout does nothing
	evdnn = Dnn(model, None)

	# 学習データ取得
	batchIndices = np.asarray([index])
	x = trainGetXBatchs(model, trainDataset, batchIndices)
	t = trainGetTBatchs(teachDataset, batchIndices)

	# ニューラルネットを通す
	y, loss = evdnn.evaluate(x, t, chainer.flag.ON)

	# 必要ならグラフ表示を行う
	if s.grEnable:
		# グラフにデータを描画する
		addZigzagLine()

		plt.title(s.trainDataFile + " : " + str(index)) # グラフタイトル
		xvals = trainDataset[:, index : index + s.minEvalLen]
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
		glZigzag.set_xdata(gxZigzag - index)
		glTeachV.set_xdata([gxOut[tx], gxOut[tx]])
		glOut.set_ydata(yvals)
		glOutV.set_xdata([gxOut[ox], gxOut[ox]])

		subPlot1.set_ylim(f.npMaxMin(xvals[0:4]))
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
	"""
	指定データを現在のニューラルネットワークを使用し予測値部分の的中率を計測する
	"""
	print('Hit rate test mode')

	# 学習データ読み込み
	trainDataset = f.loadTrainDataset()
	# 教師データ作成
	teachDataset = f.makeTeachDataset(trainDataset)

	## モデルに影響を与えないようにコピーする
	#evaluator = s.dnn.model.copy()  # to use different state
	#evaluator.reset_state()  # initialize state
	#evaluator.train = False  # dropout does nothing

	# モデルを非学習モードにしてそのまま使用する
	model = s.dnn.model
	model.reset_state()  # initialize state
	model.train = False  # dropout does nothing

	testLen = trainDataset.shape[1] - s.minEvalLen
	ynzcount = 0 # 出力が0以外だった回数
	hitcount = 0 # 教師と出力が一致した回数
	hitnzcount = 0 # 教師と出力が0以外の時に一致した回数
	sdcount = 0 # 教師と出力が同じ極性だった回数
	distance = 0.0 # 教師値との差

	xvals = trainDataset
	tvals = np.zeros(testLen, dtype=np.int32)
	yvals = np.zeros(testLen, dtype=np.int32)
	evals = np.zeros(testLen, dtype=np.int32)

	if s.grEnable:
		addZigzagLine()

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
		glZigzag.set_xdata(gxZigzag)
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
		x = trainGetXBatchs(model, trainDataset, batchIndices)
		t = trainGetTBatchs(teachDataset, batchIndices, False)
		# ニューラルネットを通す
		y = model(x, chainer.flag.ON)
		y = s.toCpu(y.data) # 最大値検索処理はCPUのが早い？

		# 的中率計算用＆描画用データにセット
		tvals[i : i + n] = tval = t - clsNum
		yvals[i : i + n] = yval = y.argmax(1) - clsNum
		tyval = tval * yval
		diff = yval - tval
		evals[i : i + n] = diff

		# 的中率更新
		i += n
		eqs = np.equal(tval, yval)
		ynzs = np.not_equal(yval, 0)
		nzs = eqs * ynzs
		ynzcount += ynzs.sum()
		hitcount += eqs.sum()
		hitnzcount += nzs.sum()
		sdcount += np.greater(tyval, 0).sum()
		distance += float((diff ** 2).sum())

		if loop % 100 == 0 or testLen <= i:
			print(
				"{0}: {1:.2f}%, {2:.2f}%, {3:.2f}%, rms err {4:.2f}".format(
					i,
					100.0 * hitcount / i,
					100.0 * hitnzcount / ynzcount if ynzcount != 0 else 0.0,
					100.0 * sdcount / ynzcount if ynzcount != 0 else 0.0,
					math.sqrt(distance / i)))

			# 指定間隔または最終データ完了後に
			# グラフにデータを描画する

			#if testLen <= i:
			#	# 最終データ完了後なら
			#	# xvals の平均値にt、yが近づくよう調整してCSVファイルに吐き出す
			#	xvalsAverage = np.average(xvals)
			#	scale = clsSpan / (clsNum * 100.0)
			#	tvals = tvals * scale
			#	yvals = yvals * scale
			#	evals = evals * scale
			#	tvals += xvalsAverage
			#	yvals += xvalsAverage
			#	evals += xvalsAverage
			#	f.writeTestHrGraphCsv(xvals, tvals, yvals)

			if s.grEnable:
				glTeach.set_ydata(tvals)
				glOut.set_ydata(yvals)
				glErr.set_ydata(evals)
				subPlot2.set_ylim(f.npMaxMin([tvals[:i], yvals[:i]]))
				plt.draw()
				plt.pause(0.001)

		loop += 1

	hitRate = 100.0 * hitcount / testLen
	nzhitRate = 100.0 * hitnzcount / ynzcount if ynzcount != 0 else 0.0
	sdRate = 100.0 * sdcount / ynzcount if ynzcount != 0 else 0.0
	distance = math.sqrt(distance / testLen)
	print("{0:.2f}%, {1:.2f}%, {2:.2f}%, rms err {3:.2f}".format(hitRate, nzhitRate, sdRate, distance))

	f.writeTestHrStatCsv(s.curEpoch, hitRate, nzhitRate, sdRate, distance)

	if s.grEnable:
		plt.ioff() # 対話モードOFF
		plt.show()

#@jit
def fxPrediction():
	"""
	現在の円データから予測する
	"""

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

	## モデル取得
	#model = s.dnn.model
	#model.train = False
	## 予測元データ取得してニューラルネットを通す
	#dataset = s.fxYenData[-s.dnnIn:,0]
	#x = trainGetXBatchs(model, dataset, np.asarray([0]))
	#y = model(x, chainer.flag.ON)
	#y = s.toCpu(y.data) # 最大値検索処理はCPUのが早い？
	#ox = y.argmax(1)[0]

	## 必要ならグラフ表示を行う
	#if s.grEnable:
	#	# グラフにデータを描画する
	#	xvals = s.fxYenData[:, -s.frameSize:]
	#	glIn1.set_ydata(xvals[0])
	#	glIn2.set_ydata(xvals[1])
	#	glIn3.set_ydata(xvals[2])
	#	glIn4.set_ydata(xvals[3])
	#	glOut.set_ydata(yvals)
	#	glOutV.set_xdata([gxOut[ox], gxOut[ox]])

	#	subPlot1.set_ylim(f.npMaxMin([xvals]))
	#	subPlot2.set_ylim(f.npMaxMin([yvals]))
	#	plt.draw()
	#	plt.pause(0.001)

	## 戻り値配列作成
	## AI予測値
	## 移動平均の差分値
	## 移動平均の差分値の差分値
	#deltaPips = float(clsSpan * (ox - clsNum) / clsNum)
	#dataForMa = s.fxYenData[3, -fxRetMaSize - 3:]
	#ma = np.asarray(np.convolve(np.asarray(dataForMa), fxRetMaSizeK, 'valid'), dtype=np.float32)
	#diff1 = np.diff(ma)
	#diff2 = np.diff(diff1)
	#return np.asarray([deltaPips, diff1[-1] * 100.0, diff2[-1] * 100.0], dtype=np.float32)
