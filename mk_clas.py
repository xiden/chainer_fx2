#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import time
import math
import csv
import codecs
import numpy as np
from numba import jit
import chainer
import matplotlib.pyplot as plt
import win32api
import win32con
import chainer.functions as F
import ini
import fxreader
import share as s
import funcs as f

clsNum = 0 # クラス分け方でのクラス数、＋片側の数、－側も同じ数だけあるので実際にクラス数は clsNum * 2 + 1 となる
clsSpan = 0 # クラス分け方でのスパン(pips)

subPlot1 = None
subPlot2 = None
gxIn = None
gyIn = None
gxOut = None
gyOut = None
glIn = None
glOut = None
glTeach = None

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

	configIni = ini.file(iniFileName, "CLAS")
	clsNum = configIni.getInt("clsNum", "3") # クラス分け方でのクラス数、＋片側の数、－側も同じ数だけあるので実際にクラス数は clsNum * 2 + 1 となる
	clsSpan = configIni.getFloat("clsSpan", "3") # クラス分け方でのスパン(pips)

	s.minPredLen = s.frameSize # ドル円未来予測に必要な最小データ数
	s.minEvalLen = s.minPredLen + s.predLen # 学習結果の評価に必要な最小データ数
	s.n_in = s.frameSize # ニューラルネットの入力次元数
	s.n_out = clsNum * 2 + 1 # ニューラルネットの出力次元数
	s.retLen = s.n_out # クライアントに返す結果データ長

def initGraph():
	global subPlot1
	global subPlot2
	global gxIn
	global gyIn
	global gxOut
	global gyOut
	global glIn
	global glOut
	global glTeach

	# グラフ描画用の初期化
	if s.grEnable:
		plt.ion() # 対話モードON
		fig = plt.figure() # 何も描画されていない新しいウィンドウを描画
		plt.xlabel("min") # x軸ラベル
		plt.ylabel("yen") # y軸ラベル
		plt.grid() # グリッド表示
		plt.gcf().canvas.set_window_title(s.testFileName)

		subPlot1 = fig.add_subplot(2, 1, 1)
		subPlot2 = fig.add_subplot(2, 1, 2)
		subPlot1.set_xlim([0, s.minEvalLen])
		subPlot2.set_xlim([-clsNum, clsNum])

		subPlot1.axvline(x=s.frameSize, color='black')
		subPlot2.axhline(y=0, color='black')

		gxIn = np.arange(0, s.minEvalLen, 1)
		gyIn = np.zeros(s.minEvalLen)
		gxOut = np.arange(-clsNum, clsNum + 1, 1)
		gyOut = np.zeros(s.n_out)

		glIn, = subPlot1.plot(gxIn, gyIn, label="in")
		glOut, = subPlot2.plot(gxOut, gyOut, label="out")
		glTeach, = subPlot2.plot(gxOut, gyOut, label="trg")

def getTestFileName(testFileName):
	return testFileName + "c" + str(clsNum) + "s" + str(clsSpan)

@jit
def getTrainData(dataset, i):
	"""学習データと教師データ取得"""
	# フレーム取得
	x = dataset[i : i + s.frameSize]

	# 教師値取得
	if s.predAve:
		t = (dataset[i + s.frameSize : i + s.frameSize + s.predLen] * s.predMeanK).sum()
	else:
		t = dataset[i + s.frameSize + s.predLen - 1]

	# フレームの最終値から教師値への変化量を教師ベクトルにする
	i = int(round(100.0 * float(t - x[-1]) * clsNum / clsSpan, 0))
	if i < -clsNum:
		i = -clsNum
	elif clsNum < i:
		i = clsNum
	i += clsNum

	# フレーム内の最低値を0になるようシフトする
	x = x - x.min()

	return s.xp.asarray([x], dtype=np.float32), s.xp.asarray([i], dtype=np.int32)

def trainBatch(dataset, itr):
	"""ミニバッチで学習する"""

	# 学習データと教師データ取得
	xa = s.xp.zeros(shape=(s.batchSize, s.n_in), dtype=np.float32)
	ta = s.xp.zeros(shape=(s.batchSize,), dtype=np.int32)
	for bi in range(s.batchSize):
		xa[bi][:], ta[bi] = getTrainData(dataset, s.batchStartIndices[bi])
	x = chainer.Variable(xa)
	t = chainer.Variable(ta)

	# 学習実行
	y, loss = s.dnn.forward(x, t, True)

	# ユーザー入力による流れ制御
	s.forceEval = False
	f.trainFlowControl()

	# 評価処理
	if (itr % s.evalInterval == 0) or s.forceEval:
		print('evaluate')
		now = time.time()
		perp = evaluate(dataset, s.evalIndex)
		print('epoch {} validation perplexity: {}'.format(s.curEpoch, perp))
		if 1 <= itr and s.optm == "Adam":
			print('learning rate =', s.dnn.optimizer.lr)

	return loss

#@jit
def evaluate(dataset, index):
	"""現在のニューラルネットワーク評価処理"""

	# モデルに影響を与えないようにコピーする
	evaluator = s.dnn.model.copy()  # to use different state
	evaluator.reset_state()  # initialize state
	evaluator.train = False  # dropout does nothing
	evdnn = Dnn(evaluator, None)

	# 学習データ取得
	x, t = getTrainData(dataset, index)
	x = chainer.Variable(x, volatile='on')
	t = chainer.Variable(t, volatile='on')

	# ニューラルネットを通す
	y, loss = evdnn.forward(x, t, True)

	# 必要ならグラフ表示を行う
	if s.grEnable:
		# グラフにデータを描画する
		plt.title(s.trainDataFile + " : " + str(index)) # グラフタイトル
		xvals = dataset[index : index + s.minEvalLen]
		tvals = np.zeros(s.n_out, dtype=np.float32)
		tvals[t.data[0]] = 1.0
		yvals = np.asarray(y.data[0].tolist(), dtype=np.float32)
		glIn.set_ydata(xvals)
		glTeach.set_ydata(tvals)
		glOut.set_ydata(yvals)

		subPlot1.set_ylim(f.npMaxMin(xvals))
		subPlot2.set_ylim(f.npMaxMin([tvals, yvals]))
		plt.draw()
		plt.pause(0.001)

	return math.exp(float(loss.data))

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

	if s.grEnable:
		xvals = np.zeros(testLen, dtype=np.float32)
		tvals = np.zeros(testLen, dtype=np.int32)
		yvals = np.zeros(testLen, dtype=np.int32)

		gxIn = np.arange(0, xvals.shape[0], 1)
		gxOut = np.arange(0, testLen, 1)
		glIn.set_xdata(gxIn)
		glTeach.set_xdata(gxOut)
		glOut.set_xdata(gxOut)
		subPlot1.set_xlim(0, xvals.shape[0])
		subPlot2.set_xlim(0, testLen)

	for i in range(testLen):
		# 学習データ取得
		x, t = getTrainData(dataset, i)
		# ニューラルネットを通す
		y = evaluator(chainer.Variable(x, volatile='on'))

		xvals[i] = dataset[i + s.frameSize - 1]
		tvals[i] = int(t[0]) - clsNum
		yvals[i] = y.data.argmax(1)[0] - clsNum

		count += 1
		if tvals[i] == yvals[i]:
			hitcount += 1

		if count % 100 == 0:
			print(i, ": ", 100.0 * hitcount / count, "%")

		if (count % 1000 == 0) or (i == testLen - 1):
			# 指定間隔または最終データ完了後に
			# グラフにデータを描画する
			plt.title("testhr: " + s.trainDataFile) # グラフタイトル

			if i == testLen - 1:
				# 最終データ完了後なら
				# xvals の平均値にt、yが近づくよう調整してCSVファイルに吐き出す
				xvalsMedian = np.median(xvals)
				scale = clsSpan / (clsNum * 100.0)
				tvals = tvals * scale
				yvals = yvals * scale
				tvals += xvalsMedian
				yvals += xvalsMedian

				with codecs.open(f.getTestHrFileBase() + str(s.curEpoch) + ".csv", 'w', "shift_jis") as file:
					writer = csv.writer(file)
					for i in range(testLen):
						writer.writerow([xvals[i], tvals[i], yvals[i]])

			glIn.set_ydata(xvals)
			glTeach.set_ydata(tvals)
			glOut.set_ydata(yvals)

			gi = i + 1
			subPlot1.set_ylim(f.npMaxMin([xvals[:gi]]))
			subPlot2.set_ylim(f.npMaxMin([tvals[:gi], yvals[:gi]]))
			plt.draw()
			plt.pause(0.001)

	result = 100.0 * hitcount / count
	print("result: ", result, "%")

	testFileIni = ini.file(s.testFileName + ".ini", f.getTestHrFileBase() + str(s.curEpoch))
	testFileIni.set("hitRate" + str(s.curEpoch), result)

	if s.grEnable:
		plt.ioff() # 対話モードOFF
		plt.show()
