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

encodedValueToPips = 2000.0 # エンコード後の円値からpips単位に換算する係数
outLen = 0 # 出力データ長
outDeltaMA = 0 # 出力値微分値の移動平サイズを奇数にする
outDeltaMAK = 0 # 出力値微分値の移動平均係数
outDeltaLen = 0 # 出力値微分値の移動平均後の長さ
bpropLen = 0 # LSTM用連続学習回数
bpropStep = 0 # LSTM用連続学習移動ステップ
bpropHeadLossCut = 0 # LSTM用連続学習時の先頭lossを無視する数
outRangeUniform = 0 # 出力値範囲を教師データ範囲と同一なるよう換算するかどうか
outHeadCut = 0 # 出力値の先頭を切り捨てるかどうか、どうも先頭側が不安定になりやすいようだ
fxAveYenKs = np.zeros(4) # ドル円未来予測用データの合成係数
fxAveYenK = 0.0 # １階微分と２階微分の混合比率
fxLastD1YVals = None # 前回の１階微分値
fxLastD2YVals = None # 前回の２階微分値

subPlot1 = None
subPlot2 = None
subPlot3 = None
gxIn = None
gyIn = None
gxOut = None
gyOut = None
gxTeachDelta = None
gyTeachDelta = None
gxOutDelta = None
gyOutDelta = None
gxOutDelta2 = None
gyOutDelta2 = None
glIn = None
glInLast = None
glTeach = None
glOut = None
glTeachDelta = None
glOutDelta = None
glOutDelta2 = None
glRet = None

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

#@jit('f8(f8)')
def encode(v):
	return (v - 110.0) / 20.0
#@jit
def encodeArray(v):
	return (v - 110.0) / 20.0
#@jit('f8(f8)')
def decode(v):
	return v * 20.0 + 110.0
#@jit
def decodeArray(v):
	return v * 20.0 + 110.0

def read(filename, inMA):
	"""指定された分足為替CSVからロウソク足データを作成する
	Args:
		filename: 読み込むCSVファイル名.
		Returns: [int]
	"""
	return encodeArray(fxreader.read(filename, inMA))

def initAveYenKs(k):
	"""未来予測データの１階～４階微分の合成係数の初期化、指定可能範囲は 0.0 < startK < 1.0、数値が大きい程未来の値の割合が大きくなる"""
	global fxAveYenKs
	global fxAveYenK
	fxAveYenK = k
	fxAveYenKs = [1.0 - k, k]

def init(iniFileName):
	"""LSTM用の初期化を行う"""
	global outLen # 出力データ長
	global outDeltaMA # 出力値微分値の移動平サイズを奇数にする
	global outDeltaMAK # 出力値微分値の移動平均係数
	global outDeltaLen # 出力値微分値の移動平均後の長さ
	global bpropLen # LSTM用連続学習回数
	global bpropStep # LSTM用連続学習移動ステップ
	global bpropHeadLossCut # LSTM用連続学習時の先頭lossを無視する数
	global outRangeUniform # 出力値範囲を教師データ範囲と同一なるよう換算するかどうか
	global outHeadCut # 出力値の先頭を切り捨てるかどうか、どうも先頭側が不安定になりやすいようだ

	configIni = ini.file(iniFileName, "LSTM")
	bpropLen = configIni.getInt("bpropLen", "60") # LSTM用連続学習回数
	bpropStep = configIni.getInt("bpropStep", "5") # LSTM用連続学習移動ステップ
	bpropHeadLossCut = configIni.getInt("bpropHeadLossCut", "10") # LSTM用連続学習時の先頭lossを無視する数
	outRangeUniform = configIni.getInt("outRangeUniform", "0") # 出力値範囲を教師データ範囲と同一なるよう換算するかどうか
	outHeadCut = configIni.getInt("outHeadCut", "0") # 出力値の先頭を切り捨てるかどうか、どうも先頭側が不安定になりやすいようだ
	outDeltaMA = configIni.getInt("outDeltaMA", "5") # 出力値の微分値の移動平均サイズ

	outLen = bpropLen - outHeadCut # 出力データ長
	outDeltaMA = (outDeltaMA // 2) * 2 + 1 # 出力値微分値の移動平サイズを奇数にする
	outDeltaMAK = np.ones(outDeltaMA) / outDeltaMA # 出力値微分値の移動平均係数
	outDeltaLen = outLen - outDeltaMA # 出力値微分値の移動平均後の長さ
	s.minPredLen = s.frameSize + (bpropLen - 1) * bpropStep # ドル円未来予測に必要な最小データ数
	s.minEvalLen = s.minPredLen + s.predLen # 学習結果の評価に必要な最小データ数
	s.n_in = s.frameSize # ニューラルネットの入力次元数
	s.n_out = 1 # ニューラルネットの出力次元数
	s.fxRetLen = outDeltaLen - 1 # クライアントに返す結果データ長
	s.fxInitialYenDataLen = s.minEvalLen * 3 # 初期化時にMT4から送る必要がある円データ数

	# 未来予測データの合成係数初期化
	initAveYenKs(fxAveYenK)

def initGraph():
	global subPlot1
	global subPlot2
	global subPlot3
	global gxIn
	global gyIn
	global gxOut
	global gyOut
	global gxTeachDelta
	global gyTeachDelta
	global gxOutDelta
	global gyOutDelta
	global gxOutDelta2
	global gyOutDelta2
	global glIn
	global glInLast
	global glTeach
	global glOut
	global glTeachDelta
	global glOutDelta
	global glOutDelta2
	global glRet

	# グラフ描画用の初期化
	if s.grEnable:
		plt.ion() # 対話モードON
		fig = plt.figure() # 何も描画されていない新しいウィンドウを描画
		plt.xlabel("min") # x軸ラベル
		plt.ylabel("yen") # y軸ラベル
		plt.grid() # グリッド表示
		plt.gcf().canvas.set_window_title(s.testFileName)

		subPlot1 = fig.add_subplot(3, 1, 1)
		subPlot2 = fig.add_subplot(3, 1, 2)
		subPlot3 = fig.add_subplot(3, 1, 3)

		subPlot2.axvline(x=(outLen - 2) * bpropStep, color='black')
		subPlot3.axhline(y=0, color='black')

		if s.mode == "server":
			gxIn = np.arange(0, s.minPredLen, 1)
			gyIn = np.zeros(s.minPredLen)
		elif s.mode == "train" or s.mode == "test":
			for i in range(bpropLen):
				subPlot1.axvline(x=i * bpropStep + s.frameSize, color='black')
			subPlot1.axvline(x=s.minPredLen, color='blue')
			gxIn = np.arange(0, s.minEvalLen, 1)
			gyIn = np.zeros(s.minEvalLen)
		elif s.mode == "testhr":
			gxIn = np.arange(0, 2, 1)
			gyIn = np.zeros(2)

		gxOut = np.arange(0, outLen * bpropStep, bpropStep)
		gyOut = np.zeros(outLen)
		gxTeachDelta = np.arange(bpropStep, outLen * bpropStep, bpropStep)
		gyTeachDelta = np.zeros(outLen - 1)
		gxOutDelta = np.arange((outLen - outDeltaLen) * bpropStep, outLen * bpropStep, bpropStep)
		gyOutDelta = np.zeros(outDeltaLen)
		gxOutDelta2 = np.arange((outLen - outDeltaLen + 1) * bpropStep, outLen * bpropStep, bpropStep)
		gyOutDelta2 = np.zeros(outDeltaLen - 1)

		glIn, = subPlot1.plot(gxIn, gyIn, label="in")
		glInLast, = subPlot2.plot(gxOut, gyOut, label="in")
		glTeach, = subPlot2.plot(gxOut, gyOut, label="trg")
		glOut, = subPlot2.plot(gxOut, gyOut, label="out")
		glTeachDelta, = subPlot3.plot(gxTeachDelta, gyTeachDelta, label="trg")
		glOutDelta, = subPlot3.plot(gxOutDelta, gyOutDelta, label="out", color='#5555ff')
		if s.mode != "testhr":
			glOutDelta2, = subPlot3.plot(gxOutDelta2, gyOutDelta2, label="out", color='#ff55ff')
		glRet, = subPlot3.plot(gxOutDelta2, gyOutDelta2, label="outave", color='#ff0000')

		#subPlot1.legend(loc='lower right') # 凡例表示
		#subPlot2.legend(loc='lower right') # 凡例表示
		#subPlot3.legend(loc='lower right') # 凡例表示

def getTestFileName(testFileName):
	return testFileName + "b" + str(bpropLen) + "bs" + str(bpropStep)

#@jit
def trainGetDataAndT(dataset, i):
	"""学習データと教師データ取得"""
	if s.predAve:
		t = (dataset[i + s.frameSize : i + s.frameSize + s.predLen] * s.predMeanK).sum()
	else:
		t = dataset[i + s.frameSize + s.predLen - 1]
	return s.xp.asarray([dataset[i : i + s.frameSize]], dtype=np.float32), s.xp.asarray([[t]], dtype=np.float32)

def trainBatch(dataset, itr):
	"""ミニバッチで学習する"""

	# LSTMによる一連の学習
	accumLoss = 0
	for i in range(bpropLen):
		# 学習データと教師データ取得
		xa = s.xp.zeros(shape=(s.batchSize, s.n_in), dtype=np.float32)
		ta = s.xp.zeros(shape=(s.batchSize, s.n_out), dtype=np.float32)
		offset = i * bpropStep
		for bi in range(s.batchSize):
			xa[bi][:], ta[bi][:] = trainGetDataAndT(dataset, s.batchStartIndices[bi] + offset)
		x = chainer.Variable(xa)
		t = chainer.Variable(ta)

		# 学習実行
		y, loss = s.dnn.forward(x, t, True)
		if bpropHeadLossCut <= i:
			accumLoss += loss

		s.forceEval = False
		onlyAveDYVals = False

		# 予測データ合成係数変更
		if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD8) & 0x8000) != 0:
			initAveYenKs(fxAveYenK + 0.05)
			s.forceEval = True
			onlyAveDYVals = True
		if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD5) & 0x8000) != 0:
			initAveYenKs(fxAveYenK - 0.05)
			s.forceEval = True
			onlyAveDYVals = True

		# ユーザー入力による流れ制御
		f.trainFlowControl()

		# 評価処理
		if (i == 0 and itr % s.evalInterval == 0) or s.forceEval:
			print('evaluate')
			now = time.time()
			perp = evaluate(dataset, s.evalIndex, onlyAveDYVals)
			print('epoch {} validation perplexity: {}'.format(s.curEpoch, perp))
			#if 1 <= itr and s.optm == "Adam":
			#	print('learning rate =', s.dnn.optimizer.lr)

	return accumLoss

#@jit
def evaluate(dataset, index, onlyAveDYVals = False):
	"""現在のニューラルネットワーク評価処理"""

	global fxLastD1YVals
	global fxLastD2YVals

	if onlyAveDYVals:
		# 予測値のみ更新する
		avedyvals = fxAveYenKs[0] * fxLastD1YVals[1:] + fxAveYenKs[1] * fxLastD2YVals
		plt.title(s.trainDataFile + " : " + str(index) + " : " + str(fxAveYenK)) # グラフタイトル
		glRet.set_ydata(avedyvals)
		plt.draw()
		plt.pause(0.001)
		return -1.0

	# モデルに影響を与えないようにコピーする
	evaluator = s.dnn.model.copy()  # to use different state
	evaluator.reset_state()  # initialize state
	evaluator.train = False  # dropout does nothing
	evdnn = Dnn(evaluator, None)

	accumLoss = 0

	if s.grEnable:
		xvals = dataset[index : index + s.minEvalLen]
		ivals = np.zeros(outLen, dtype=np.float32)
		tvals = np.zeros(outLen, dtype=np.float32)
		yvals = np.zeros(outLen, dtype=np.float32)

	for i in range(bpropLen):
		# 学習データ取得
		x, t = trainGetDataAndT(dataset, index + i * bpropStep)
		ival = x[0][-1]
		x = chainer.Variable(x, volatile='on')
		t = chainer.Variable(t, volatile='on')

		# ニューラルネットを通す
		y, loss = evdnn.forward(x, t, True)
		if bpropHeadLossCut <= i:
			accumLoss += loss.data

		if s.grEnable and outHeadCut <= i:
			ivals[i - outHeadCut] = ival
			tvals[i - outHeadCut] = t.data[0][0]
			yvals[i - outHeadCut] = y.data[0][0]

	# 必要ならグラフ表示を行う
	if s.grEnable:
		# 指定されていたら出力値のレンジを教師データレンジに合わせる
		if outRangeUniform:
			tvalsSpan = tvals.ptp()
			tvalsMin = tvals.min()
			yvalsSpan = yvals.ptp()
			yvalsMin = yvals.min()
			yvalsScale = 0.0 if yvalsSpan == 0 else tvalsSpan / yvalsSpan
			yvals *= yvalsScale
			yvals += tvalsMin - yvalsMin * yvalsScale

		# 予測値を微分し移動平均する、それをさらに微分したものを作成する
		d1yvals = yvals[1:] - yvals[:-1]
		d1yvals = np.asarray(np.convolve(np.asarray(d1yvals), outDeltaMAK, 'valid'), dtype=np.float32)
		d2yvals = d1yvals[1:] - d1yvals[:-1]
		avedyvals = fxAveYenKs[0] * d1yvals[1:] + fxAveYenKs[1] * d2yvals
		# 教師データを微分する
		dtvals = tvals[1:] - tvals[:-1]
		# 微分後データをpips単位に変換する
		d1yvals *= encodedValueToPips
		d2yvals *= encodedValueToPips
		avedyvals *= encodedValueToPips
		dtvals *= encodedValueToPips

		# グラフにデータを描画する
		plt.title(s.trainDataFile + " : " + str(index) + " : " + str(fxAveYenK)) # グラフタイトル
		glIn.set_ydata(xvals)
		glInLast.set_ydata(ivals)
		glTeach.set_ydata(tvals)
		glOut.set_ydata(yvals)
		glTeachDelta.set_ydata(dtvals)
		glOutDelta.set_ydata(d1yvals)
		glOutDelta2.set_ydata(d2yvals)
		glRet.set_ydata(avedyvals)
		fxLastD1YVals = d1yvals
		fxLastD2YVals = d2yvals

		subPlot1.set_xlim(0, s.minEvalLen)
		subPlot1.set_ylim(f.npMaxMin(xvals))
		subPlot2.set_xlim([0, outLen * bpropStep])
		subPlot2.set_ylim(f.npMaxMin([tvals, yvals]))
		subPlot3.set_xlim([0, outLen * bpropStep])
		subPlot3.set_ylim(f.npMaxMin([dtvals, d1yvals, d2yvals, avedyvals]))
		plt.draw()
		plt.pause(0.001)

	return math.exp(float(accumLoss) / (bpropLen - bpropHeadLossCut))

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
	testStart = bpropLen * bpropStep
	testEnd = dataset.shape[0] - s.minEvalLen
	testLen = (testEnd - testStart) // bpropStep + 1
	count = 0
	hitcount = 0
	lastT = 0
	lastY = 0

	if s.grEnable:
		tvals = np.zeros(testLen, dtype=np.float32)
		ivals = np.zeros(testLen, dtype=np.float32)
		yvals = np.zeros(testLen, dtype=np.float32)
		dtvals = np.zeros(testLen, dtype=np.float32)
		dyvals = np.zeros(testLen, dtype=np.float32)
		devals = np.zeros(testLen, dtype=np.float32)
		gi = 0

		gxIn = np.arange(0, dataset.shape[0], 1)
		gxOut = np.arange(testStart, testStart + testLen * bpropStep, bpropStep)
		gxTeachDelta = gxOut
		gxOutDelta = gxOut
		glIn.set_xdata(gxIn)
		glInLast.set_xdata(gxOut)
		glTeach.set_xdata(gxOut)
		glOut.set_xdata(gxOut)
		glTeachDelta.set_xdata(gxOut)
		glOutDelta.set_xdata(gxOut)
		glRet.set_xdata(gxOut)
		subPlot1.set_xlim(0, dataset.shape[0])
		subPlot2.set_xlim([0, dataset.shape[0]])
		subPlot3.set_xlim([0, dataset.shape[0]])
		subPlot1.set_ylim(f.npMaxMin(dataset))

	while testPos <= testEnd:
		# 学習データ取得
		x, t = trainGetDataAndT(dataset, testPos)
		inLast = float(x[0][-1])
		x = chainer.Variable(x, volatile='on')

		# ニューラルネットを通す
		y = evaluator(x)

		if testStart <= testPos:
			count += 1
			dt = float(t - lastT)
			dy = float(y.data[0][0] - lastY.data[0][0])
			if 0.0 < dt * dy:
				hitcount += 1
			if count % 100 == 0:
				print(testPos, ": ", 100.0 * hitcount / count, "%")
			if s.grEnable:
				tvals[gi] = float(t)
				ivals[gi] = inLast
				yvals[gi] = float(y.data[0][0])
				dtvals[gi] = dt
				dyvals[gi] = dy
				devals[gi] = dt - dy
				gi += 1

				if (count % 1000 == 0 or testEnd < testPos + bpropStep):
					# 指定間隔または最終データ完了後に
					# グラフにデータを描画する
					plt.title("testhr: " + s.trainDataFile) # グラフタイトル

					if testEnd < testPos + bpropStep:
						# 最終データ完了後なら
						# 円単位に直す
						ivals = decodeArray(ivals)
						tvals = decodeArray(tvals)
						yvals = decodeArray(yvals)
						# CSVファイルに吐き出す
						with codecs.open(f.getTestHrFileBase() + str(s.curEpoch) + ".csv", 'w', "shift_jis") as file:
							writer = csv.writer(file)
							for i in range(ivals.shape[0]):
								writer.writerow([ivals[i], tvals[i], yvals[i]])
					#	tvalsSpan = tvals.ptp()
					#	tvalsMin = tvals.min()
					#	yvalsSpan = yvals.ptp()
					#	yvalsMin = yvals.min()
					#	yvalsScale = tvalsSpan / yvalsSpan if yvalsSpan != 0.0 else 0.0
					#	yvals *= yvalsScale
					#	yvals += tvalsMin - yvalsScale * yvalsMin

					glIn.set_ydata(dataset)
					glInLast.set_ydata(ivals)
					glTeach.set_ydata(tvals)
					glOut.set_ydata(yvals)
					glTeachDelta.set_ydata(dtvals)
					glOutDelta.set_ydata(dyvals)
					glRet.set_ydata(devals)

					subPlot2.set_ylim(f.npMaxMin([tvals[:gi], yvals[:gi]]))
					subPlot3.set_ylim(f.npMaxMin([dtvals[:gi], dyvals[:gi]]))
					plt.draw()
					plt.pause(0.001)

		lastT = t
		lastY = y
		testPos += bpropStep

	result = 100.0 * hitcount / count
	print("result: ", result, "%")

	section = s.trainDataFile
	testFileIni = ini.file(s.testFileName + ".ini", section)
	testFileIni.set("hitRate" + str(s.curEpoch), result)
	testFileIni.setSection(section + "_DEFAULT" + str(s.curEpoch), s.configIni.getSection("DEFAULT"))
	testFileIni.setSection(section + "_LSTM" + str(s.curEpoch), s.configIni.getSection("LSTM"))

	if s.grEnable:
		plt.ioff() # 対話モードOFF
		plt.show()
