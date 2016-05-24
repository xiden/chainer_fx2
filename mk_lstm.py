#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import time
import math
import csv
import codecs
import numpy as np
from numba import jit
import chainer
import chainer.cuda as cuda
import matplotlib.pyplot as plt
import win32api
import win32con
import chainer.functions as F
import os.path as path
import ini
import fxreader
import share as s
import funcs as f

rnnLen = 0 # LSTM用連続学習回数
rnnStep = 0 # LSTM用連続学習移動ステップ
trainDataExtract = None
datasetExtract = None # データセットの抽出方法

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
glTeach = None
glErr = None

class Dnn(object):
	"""LSTM用のモデルとオプティマイザへルパクラス"""
	model = None
	optimizer = None

	def __init__(self, model = None, optimizer = None):
		self.model = model
		self.optimizer = optimizer

	def forward(self, x, t, calcLoss):
		y = self.model(x)
		if calcLoss:
			return (y, F.mean_squared_error(y, t))
		else:
			return (y, None)

	def update(self, loss):
		self.model.zerograds()
		loss.backward()
		loss.unchain_backward()  # truncate
		self.optimizer.update()

#encodedValueToPips = 100.0 # エンコード後の円値からpips単位に換算する係数
##@jit('f8(f8)')
#def encode(v):
#	return v
##@jit
#def encodeArray(v):
#	return v
##@jit('f8(f8)')
#def decode(v):
#	return v
##@jit
#def decodeArray(v):
#	return v

encodedValueToPips = 1000.0 # エンコード後の円値からpips単位に換算する係数
#@jit('f8(f8)')
def encode(v):
	return (v - 110.0) / 10.0
#@jit
def encodeArray(v):
	return (v - 110.0) / 10.0
#@jit('f8(f8)')
def decode(v):
	return v * 10.0 + 110.0
#@jit
def decodeArray(v):
	return v * 10.0 + 110.0

def readDataset(filename, inMA, noise):
	"""指定された分足為替CSVからロウソク足データを作成する
	Args:
		filename: 読み込むCSVファイル名.
		Returns: 開始値、高値、低値、終値が縦に並んでるイメージの2次元データ
	"""
	return encodeArray(fxreader.readDataset(filename, inMA, noise))

def init(iniFileName):
	"""LSTM用の初期化を行う"""
	global rnnLen
	global rnnStep
	global trainDataExtract
	global datasetExtract

	configIni = ini.file(iniFileName, "LSTM")
	rnnLen = configIni.getInt("rnnLen", "30") # LSTM用連続学習回数
	rnnStep = configIni.getInt("rnnStep", "1") # LSTM用連続学習移動ステップ
	datasetExtract = configIni.getStr("datasetExtract", "all") # データセットの抽出方法

	# INIにより関数振り分け
	if datasetExtract == "all":
		trainDataExtract = trainDataExtractAll
	elif datasetExtract == "open":
		trainDataExtract = trainDataExtractOpen
	else:
		print("Unknown trainDataExtract :", trainDataExtract)
		sys.exit()

	s.minPredLen = s.frameSize + (rnnLen - 1) * rnnStep # ドル円未来予測に必要な最小データ数
	s.minEvalLen = s.minPredLen + s.predLen # 学習結果の評価に必要な最小データ数

	# ニューラルネットの入力次元数
	if trainDataExtract == trainDataExtractOpen:
		s.dnnIn = s.frameSize
	else:
		s.dnnIn = s.frameSize * 4

	# ニューラルネットの出力次元数
	s.dnnOut = 1

	s.fxRetLen = 3 # クライアントに返す結果データ長
	s.fxInitialYenDataLen = s.minEvalLen * 3 # 初期化時にMT4から送る必要がある円データ数

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
		subPlot2.set_xlim([0, rnnLen])
		subPlot2.axhline(y=0, color='black')

		gxOut = np.arange(0, rnnLen, 1)
		gyOut = np.zeros(rnnLen)

		if s.mode == "server":
			subPlot1.set_xlim([0, s.minPredLen])
			gxIn = np.arange(0, s.minPredLen, 1)
			gyIn = np.zeros(s.minPredLen)
		else:
			subPlot1.set_xlim([0, s.minEvalLen])
			subPlot1.axvline(x=s.minPredLen, color='black')
			gxIn = np.arange(0, s.minEvalLen, 1)
			gyIn = np.zeros(s.minEvalLen)

		glIn1, = subPlot1.plot(gxIn, gyIn, label="open")
		glIn2, = subPlot1.plot(gxIn, gyIn, label="high")
		glIn3, = subPlot1.plot(gxIn, gyIn, label="low")
		glIn4, = subPlot1.plot(gxIn, gyIn, label="close")
		glOut, = subPlot2.plot(gxOut, gyOut, label="y")
		if s.mode != "server":
			glTeach, = subPlot2.plot(gxOut, gyOut, label="t", color='green')
		if s.mode == "testhr" or s.mode == "trainhr":
			glErr, = subPlot2.plot(gxOut, gyOut, label="err", color='red')

		subPlot1.legend(loc='lower left') # 凡例表示
		subPlot2.legend(loc='lower left') # 凡例表示

def getTestFileName(testFileName):
	return testFileName + "rnn" + str(rnnLen) + "step" + str(rnnStep)

#@jit
def trainDataExtractAll(x, n):
	return np.reshape(x[:n], (n * 4,))

#@jit("f4[:](f4[:],i4)", nopython=True)
def trainDataExtractOpen(x, n):
	return x[:n,0]

#@jit
def trainGetX(dataset, i):
	"""学習データ取得"""
	return trainDataExtract(dataset[i:], s.frameSize)

#@jit
def trainGetT(dataset, i):
	"""教師データ取得"""
	frameEnd = i + s.frameSize

	# 教師値取得
	# 既知の終値と未来の分足データの開始値との差を教師とする
	last = dataset[frameEnd - 1, 3]
	predData = dataset[frameEnd : frameEnd + s.predLen]
	if s.predAve:
		p = (predData * s.predMeanK).sum()
	else:
		p = predData[-1, 0]
	return p - last

#@jit(nopython=True)
def trainBatch(dataset, itr):
	"""ミニバッチで学習する"""

	# 全ミニバッチ分のメモリ領域確保して学習＆教師データ取得
	xa_cpu = np.zeros(shape=(rnnLen, s.batchSize, s.dnnIn), dtype=np.float32)
	ta_cpu = np.zeros(shape=(rnnLen, s.batchSize, s.dnnOut), dtype=np.float32)
	for i in range(rnnLen):
		offset = i * rnnStep
		for bi in range(s.batchSize):
			index = s.batchStartIndices[bi] + offset
			xa_cpu[i,bi,:] = trainGetX(dataset, index)
			ta_cpu[i,bi,:] = trainGetT(dataset, index)
	if s.xp == np:
		xa_gpu = xa_cpu
		ta_gpu = ta_cpu
	else:
		xa_gpu = cuda.to_gpu(xa_cpu)
		ta_gpu = cuda.to_gpu(ta_cpu)

	# LSTMによる一連の学習
	accumLoss = 0
	for i in range(rnnLen):
		# 学習実行
		y, loss = s.dnn.forward(chainer.Variable(xa_gpu[i]), chainer.Variable(ta_gpu[i]), True)
		accumLoss += loss # 誤差逆伝播時に辿れる様にグラフ追加していく

		# ユーザー入力による流れ制御
		s.forceEval = False
		f.trainFlowControl()

		# 評価処理
		if (i == 0 and itr % s.evalInterval == 0) or s.forceEval:
			print('evaluate')
			perp = trainEvaluate(dataset, s.evalIndex)
			print('epoch {} validation perplexity: {}'.format(s.curEpoch, perp))
			#if 1 <= itr and s.optm == "Adam":
			#	print('learning rate =', s.dnn.optimizer.lr)

	return accumLoss

#@jit
def trainEvaluate(dataset, index):
	"""現在のニューラルネットワーク評価処理"""

	# モデルに影響を与えないようにコピーする
	evaluator = s.dnn.model.copy()  # to use different state
	evaluator.reset_state()  # initialize state
	evaluator.train = False  # dropout does nothing
	evdnn = Dnn(evaluator, None)

	# メモリ領域確保して学習＆教師データ取得
	xa_cpu = np.zeros(shape=(rnnLen, 1, s.dnnIn), dtype=np.float32)
	ta_cpu = np.zeros(shape=(rnnLen, 1, s.dnnOut), dtype=np.float32)
	for i in range(rnnLen):
		offset = index + i * rnnStep
		xa_cpu[i,0,:] = trainGetX(dataset, offset)
		ta_cpu[i,0,:] = trainGetT(dataset, offset)
	if s.xp == np:
		xa_gpu = xa_cpu
		ta_gpu = ta_cpu
		losses = np.zeros((rnnLen,), dtype=np.float32)
	else:
		xa_gpu = cuda.to_gpu(xa_cpu)
		ta_gpu = cuda.to_gpu(ta_cpu)
		losses = cuda.zeros((rnnLen,), dtype=np.float32)

	# 必要ならグラフ表示初期化
	if s.grEnable:
		if s.xp == np:
			ya_gpu = np.zeros((rnnLen,), dtype=np.float32)
		else:
			ya_gpu = cuda.zeros((rnnLen,), dtype=np.float32)

		# グラフにデータを描画する
		plt.title(s.trainDataFile + " : " + str(index)) # グラフタイトル
		xvals = dataset[index : index + s.minEvalLen].transpose()
		tvals = ta_cpu.reshape((rnnLen,))
		glIn1.set_ydata(xvals[0])
		glIn2.set_ydata(xvals[1])
		glIn3.set_ydata(xvals[2])
		glIn4.set_ydata(xvals[3])
		glTeach.set_ydata(tvals)
		subPlot1.set_ylim(f.npMaxMin(xvals))

	# RNNを評価
	for i in range(rnnLen):
		y, loss = evdnn.forward(chainer.Variable(xa_gpu[i], volatile='on'), chainer.Variable(ta_gpu[i], volatile='on'), True)
		losses[i : i + 1] = loss.data
		if s.grEnable:
			ya_gpu[i : i + 1] = y.data[0, 0 : 1]

	# 必要ならグラフ表示
	if s.grEnable:
		if s.xp == np:
			yvals = ya_gpu
		else:
			yvals = cuda.to_cpu(ya_gpu)
		glOut.set_ydata(yvals)
		subPlot2.set_ylim(f.npMaxMin([tvals, yvals]))
		plt.draw()
		plt.pause(0.001)

	return math.exp(float(losses.sum()) / rnnLen)

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
	evaluator = s.dnn.model
	evaluator.reset_state()  # initialize state
	evaluator.train = False  # dropout does nothing

	# 的中率計算に必要な変数など初期化
	testPos = 0
	testEnd = dataset.shape[0] - s.minEvalLen
	testLen = testEnd // rnnStep + 1
	hitcount = 0 # 教師と出力が一致した回数
	hitnzcount = 0 # 教師と出力が0以外の時に一致した回数
	sdcount = 0 # 教師と出力が同じ極性だった回数
	distance = 0.0 # 教師値との差

	# メモリ領域確保して学習＆教師データ取得
	xa_cpu = trainDataExtract(dataset, dataset.shape[0])
	ta_cpu = np.zeros(shape=(testLen,), dtype=np.float32)
	ya_cpu = np.zeros(shape=(testLen,), dtype=np.float32)
	for i in range(testLen):
		ta_cpu[i] = trainGetT(dataset, i * rnnStep)
	if s.xp == np:
		xa_gpu = xa_cpu
		ya_gpu = ya_cpu
	else:
		xa_gpu = cuda.to_gpu(xa_cpu)
		ya_gpu = cuda.to_gpu(ya_cpu)

	xvals = dataset.transpose()
	tvals = ta_cpu * encodedValueToPips
	yvals = np.zeros(testLen, dtype=np.float32)

	# 必要ならグラフ表示初期化
	if s.grEnable:
		evals = np.zeros(testLen, dtype=np.float32)
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

	# RNNを評価する
	i = 0
	loop = 0
	lastLen = 0
	while i < testLen:
		# ニューラルネットを通す
		y = evaluator(chainer.Variable(xa_gpu[testPos : testPos + s.dnnIn].reshape((1, s.dnnIn)), volatile='on'))
		ya_gpu[i : i + 1] = y.data[0 : 1]
		i += 1

		if loop % 1000 == 0 or testLen <= i:
			# 指定間隔または最終データ完了後に
			# グラフにデータを描画する

			# 的中率計算用＆描画用データにセット
			tval = ta_cpu[lastLen : i]
			if s.xp == np:
				yval = ya_gpu[lastLen : i] * encodedValueToPips
			else:
				yval = cuda.to_cpu(ya_gpu[lastLen : i]) * encodedValueToPips
			tyval = tval * yval
			diff = tval - yval
			if s.grEnable:
				yvals[lastLen : i] = yval
				evals[lastLen : i] = np.less(tyval, 0) * diff # 極性が逆の場合のみ誤差波形になるようにする
			lastLen = i

			# 的中率更新
			eqs = np.less(np.abs(diff), 0.1)
			nzs = eqs * np.not_equal(yval, 0)
			hitcount += eqs.sum()
			hitnzcount += nzs.sum()
			sdcount += np.greater(tyval, 0).sum()
			distance += float((diff ** 2).sum())

			print("{0}: {1:.2f}%, {2:.2f}%, {3:.2f}%, rms err {4:.2f}".format(i, 100.0 * hitcount / i, 100.0 * hitnzcount / i, 100.0 * sdcount / i, math.sqrt(distance / i)))

			if testLen <= i:
				# 最終データ完了後なら
				# xvals の平均値にt、yが近づくよう調整してCSVファイルに吐き出す
				xvalsAverage = np.average(xvals)
				tvals += xvalsAverage
				yvals += xvalsAverage
				evals += xvalsAverage
				f.writeTestHrGraphCsv(xvals, tvals, yvals)

			if s.grEnable and (loop % 5000 == 0 or testLen <= i):
				glTeach.set_ydata(tvals)
				glOut.set_ydata(yvals)
				glErr.set_ydata(evals)
				subPlot2.set_ylim(f.npMaxMin([tvals[:i], yvals[:i]]))
				plt.draw()
				plt.pause(0.001)

		loop += 1
		testPos += rnnStep

	hitRate = 100.0 * hitcount / testLen
	nzhitRate = 100.0 * hitnzcount / testLen
	sdRate = 100.0 * sdcount / testLen
	distance = math.sqrt(distance / testLen)
	print("{0:.2f}%, {1:.2f}%, {2:.2f}%, rms err {3:.2f}".format(hitRate, nzhitRate, sdRate, distance))

	f.writeTestHrStatCsv(s.curEpoch, hitRate, nzhitRate, sdRate, distance)

	if s.grEnable:
		plt.ioff() # 対話モードOFF
		plt.show()
