#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import argparse
import math
import sys
import time
import csv
from numba import jit
import random
import numpy as np
import matplotlib.pyplot as plt
import six
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda
from chainer import optimizers
from chainer import serializers
import net
import candle
import win32con
import os.path as path
import threading
import ini

class DnnHelper(object):
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
		#self.model.zerograds()
		#loss.backward()
		#self.optimizer.update()

def loadModelAndOptimizer():
	"""モデルとオプティマイザ読み込み"""
	if modelFile and path.isfile(modelFile):
		print('Load model from', netType + "_" + modelFile)
		serializers.load_npz(modelFile, dnn.model)
	if stateFile and path.isfile(stateFile):
		print('Load optimizer state from', stateFile)
		serializers.load_npz(stateFile, dnn.optimizer)

def saveModelAndOptimizer():
	"""モデルとオプティマイザ保存"""
	testFileIni.set("curEpoch", curEpoch) # 現在の実施済みエポック数保存
	with modelLock:
		print('save the model')
		serializers.save_npz(modelFile, dnn.model)
		print('save the optimizer')
		serializers.save_npz(stateFile, dnn.optimizer)

def snapShotPredictionModel():
	"""学習中のモデルからドル円未来予測用のモデルを作成する"""
	global fxYenPredictionModel
	e = dnn.model.copy()  # to use different state
	e.reset_state()  # initialize state
	e.train = False  # dropout does nothing
	fxYenPredictionModel = e

def initAveYenKs(k):
	"""未来予測データの１階～４階微分の合成係数の初期化、指定可能範囲は 0.0 < startK < 1.0、数値が大きい程未来の値の割合が大きくなる"""
	global fxAveYenKs
	global fxAveYenK
	fxAveYenK = k
	fxAveYenKs = [1.0 - k, k]

@jit
def getTrainData(dataset, i):
	"""学習データと教師データ取得"""
	if predAve:
		t = (dataset[i + frameSize : i + frameSize + predLen] * predMeanK).sum()
	else:
		t = dataset[i + frameSize + predLen - 1]
	return xp.asarray([dataset[i : i + frameSize]], dtype=np.float32), xp.asarray([[t]], dtype=np.float32)

@jit
def getFxYenData(dataset, i):
	"""ドル円データ取得"""
	tstart = i + frameSize
	tend = tstart + predLen
	if tend <= dataset.shape[0]:
		t = (dataset[tstart : tend] * predMeanK).sum()
	else:
		t = None
	return xp.asarray([dataset[i : tstart]], dtype=np.float32), t

@jit
def npMaxMin(arrays):
	rmax = float(arrays[0].max())
	rmin = float(arrays[0].min())
	for i in range(1, len(arrays)):
		tmax = float(arrays[i].max())
		tmin = float(arrays[i].min())
		if rmax < tmax: rmax = tmax
		if tmin < rmin: rmin = tmin
	return rmin, rmax

#@jit
def evaluate(dataset, index, onlyAveDYVals = False):
	"""現在のニューラルネットワーク評価処理"""
	global fxLastD1YVals
	global fxLastD2YVals
	global fxLastD3YVals
	global fxLastD4YVals
	global fxLastD5YVals

	if onlyAveDYVals:
		# 予測値のみ更新する
		avedyvals = fxAveYenKs[0] * fxLastD1YVals[1:] + fxAveYenKs[1] * fxLastD2YVals
		plt.title(trainDataFile + " : " + str(index) + " : " + str(fxAveYenK)) # グラフタイトル
		glRet.set_ydata(avedyvals)
		plt.draw()
		plt.pause(0.001)
		return -1.0

	# モデルに影響を与えないようにコピーする
	evaluator = dnn.model.copy()  # to use different state
	evaluator.reset_state()  # initialize state
	evaluator.train = False  # dropout does nothing
	evdnn = DnnHelper(evaluator, None)

	accumLoss = 0

	if grEnable:
		xvals = dataset[index : index + minEvalLen]
		tvals = np.zeros(outLen, dtype=np.float32)
		yvals = np.zeros(outLen, dtype=np.float32)

	for i in range(bpropLen):
		# 学習データ取得
		x, t = getTrainData(dataset, index + i * bpropStep)
		x = chainer.Variable(x, volatile='on')
		t = chainer.Variable(t, volatile='on')

		# ニューラルネットを通す
		y, loss = evdnn.forward(x, t, True)
		if bpropHeadLossCut <= i:
			accumLoss += loss.data

		if grEnable and outHeadCut <= i:
			tvals[i - outHeadCut] = t.data[0][0]
			yvals[i - outHeadCut] = y.data[0][0]

	# 必要ならグラフ表示を行う
	if grEnable:
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
		d1yvals *= candle.encodedValueToPips
		d2yvals *= candle.encodedValueToPips
		avedyvals *= candle.encodedValueToPips
		dtvals *= candle.encodedValueToPips

		# グラフにデータを描画する
		plt.title(trainDataFile + " : " + str(index) + " : " + str(fxAveYenK)) # グラフタイトル
		glIn.set_xdata(gxIn)
		glTeach.set_xdata(gxOut)
		glOut.set_xdata(gxOut)
		glIn.set_ydata(xvals)
		glTeach.set_ydata(tvals)
		glOut.set_ydata(yvals)
		glTeachDelta.set_ydata(dtvals)
		glOutDelta.set_ydata(d1yvals)
		glOutDelta2.set_ydata(d2yvals)
		glRet.set_ydata(avedyvals)
		fxLastD1YVals = d1yvals
		fxLastD2YVals = d2yvals

		subPlot1.set_xlim(0, minEvalLen)
		subPlot1.set_ylim(npMaxMin(xvals))
		subPlot2.set_xlim([0, outLen * bpropStep])
		subPlot2.set_ylim(npMaxMin([tvals, yvals]))
		subPlot3.set_xlim([0, outLen * bpropStep])
		subPlot3.set_ylim(npMaxMin([dtvals, d1yvals, d2yvals, avedyvals]))
		plt.draw()
		plt.pause(0.001)

	return math.exp(float(accumLoss) / (bpropLen - bpropHeadLossCut))

#@jit
def prediction():
	"""現在のニューラルネットワークで未来を予測"""

	# モデルに影響を与えないようにコピーする
	dataset = fxYenData[-minPredLen:]
	evaluator = fxYenPredictionModel  # to use different state

	accumLoss = 0
	n = bpropLen
	yvals = np.zeros(outLen, dtype=np.float32)
	if grEnable:
		xvals = dataset
		tvals = np.zeros(outLen, dtype=np.float32)

	for i in range(n):
		# 学習データ取得
		x, t = getFxYenData(dataset, i * bpropStep)
		x = chainer.Variable(x, volatile='on')

		# ニューラルネットを通す
		y = evaluator(x)

		if outHeadCut <= i:
			ig = i - outHeadCut
			yvals[ig] = y.data[0][0]
			if grEnable:
				tvals[ig] = tvals[ig - 1] if t is None else t

	# 予測値を１～２階微分を行い合成する
	d1yvals = yvals[1:] - yvals[:-1]
	d2yvals = d1yvals[1:] - d1yvals[:-1]
	avedyvals = fxAveYenKs[0] * d1yvals[1:] + fxAveYenKs[1] * d2yvals
	# 微分後データをpips単位に変換する
	d1yvals *= candle.encodedValueToPips
	d2yvals *= candle.encodedValueToPips
	avedyvals *= candle.encodedValueToPips

	# 必要ならグラフ表示を行う
	if grEnable:
		# 教師データを微分する
		dtvals = tvals[1:] - tvals[:-1]
		# 微分後データをpips単位に変換する
		dtvals *= candle.encodedValueToPips

		# グラフにデータを描画する
		plt.title("USD/JPY : " + str(fxAveYenK)) # グラフタイトル
		glIn.set_ydata(xvals)
		glTeach.set_ydata(tvals)
		glOut.set_ydata(yvals)
		glTeachDelta.set_ydata(dtvals)
		glOutDelta.set_ydata(d1yvals)
		glOutDelta2.set_ydata(d2yvals)
		glRet.set_ydata(avedyvals)

		subPlot1.set_xlim(0, minEvalLen)
		subPlot1.set_ylim(npMaxMin(xvals))
		subPlot2.set_xlim([0, outLen])
		subPlot2.set_ylim(npMaxMin([tvals, yvals]))
		subPlot3.set_xlim([0, outLen])
		subPlot3.set_ylim(npMaxMin([dtvals, d1yvals, d2yvals, avedyvals]))
		plt.draw()
		plt.pause(0.01)

	return avedyvals

def trainFxYen():
	"""ドル円学習"""
	global fxYenDataTrain

	# 学習ループ初期化
	train_data = None
	requestQuit = False
	updateCount = 0

	# 学習データの取得と学習のループ
	while True:
		if requestQuit:
			break

		# 学習用データがあるなら取得する
		if not (fxYenDataTrain is None):
			train_data = fxYenDataTrain
			fxYenDataTrain = None
			whole_len = train_data.shape[0]
			batchRangeStart = 0
			batchRangeEnd = whole_len - minEvalLen
			if batchRangeEnd < 0:
				print("Data length not enough")
				sys.exit()
			batchIndices = [0] * batchSize
			for i in range(batchSize):
				batchIndices[i] = batchRangeStart + i * (batchRangeEnd - batchRangeStart) // batchSize
			batchStart = 0
			batchStep = (batchRangeEnd - batchRangeStart) // batchSize
			batchOffsetInitial = batchRangeEnd - batchIndices[len(batchIndices) - 1]
			batchOffset = batchOffsetInitial

		# LSTMによる一連の学習
		if train_data is None:
			time.sleep(0.1)
		else:
			with modelLock:
				accumLoss = 0
				batchStartIndices = np.asarray(batchIndices, dtype=np.integer)
				batchStartIndices += batchOffset
				for j in range(bpropLen):
					# 学習データと教師データ取得
					xa = xp.zeros(shape=(batchSize, n_in), dtype=np.float32)
					ta = xp.zeros(shape=(batchSize, n_out), dtype=np.float32)
					offset = j * bpropStep
					for bi in range(batchSize):
						xa[bi][:], ta[bi][:] = getTrainData(train_data, batchStartIndices[bi] + offset)
					x = chainer.Variable(xa)
					t = chainer.Variable(ta)

					# 学習実行
					y, loss = dnn.forward(x, t, True)
					accumLoss += loss

					## 終了判定処理
					#if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD0) & 0x8000) != 0:
					#	requestQuit = True
					#	break

				# ニューラルネットワーク更新
				print("dnn update, loss: ", math.exp(float(accumLoss.data) / bpropLen))
				dnn.update(accumLoss)

			# ニューラルネットワークをドル円予測処理で使える様にコピー
			updateCount += 1
			if updateCount % 10 == 0:
				snapShotPredictionModel()

			# バッチ位置の移動
			batchOffset -= predLen
			if batchOffset < 0:
				batchOffset = batchOffsetInitial
				print("一周した")

# コマンドライン引数解析
parser = argparse.ArgumentParser()
parser.add_argument('iniFileName', help='設定ファイル')
args = parser.parse_args()

# 指定されたINIファイルからパラメータ取得
configIni = ini.file(args.iniFileName, "DEFAULT")
configIni.getStr("key")

mode = configIni.getStr("mode", "train") # 実行モード
trainDataFile = configIni.getStr("trainDataFile", "") # 学習データファイル
trainDataDummy = configIni.getStr("trainDataDummy", "") # 生成したダミーデータを学習データするするかどうか、 sin/sweep
gpu = configIni.getInt("gpu", "-1") # 使用GPU番号、0以上ならGPU使用
netType = configIni.getStr("netType", "") # ニューラルネットワークモデルタイプ
netInitParamRandom = configIni.getFloat("netInitParamRandom", "0.0") # ニューラルネットワーク重み初期化乱数サイズ
epoch = configIni.getInt("epoch", "1000") # 実行エポック数
numUnits = configIni.getInt("numUnits", "60") # ユニット数
inMA = configIni.getInt("inMA", "5") # 入力値の移動平均サイズ
frameSize = configIni.getInt("frameSize", "300") # 入力フレームサイズ
batchSize = configIni.getInt("batchSize", "20") # バッチ数
batchRandom = configIni.getInt("batchRandom", "1") # バッチ位置をランダムにするかどうか
bpropLen = configIni.getInt("bpropLen", "60") # LSTM用連続学習回数
bpropStep = configIni.getInt("bpropStep", "5") # LSTM用連続学習移動ステップ
bpropHeadLossCut = configIni.getInt("bpropHeadLossCut", "10") # LSTM用連続学習時の先頭lossを無視する数
gradClip = configIni.getFloat("gradClip", "5") # 勾配クリップ
grEnable = configIni.getInt("grEnable", "0") # グラフ表示有効かどうか
evalInterval = configIni.getInt("evalInterval", "20") # 評価（グラフも）間隔エポック数
outRangeUniform = configIni.getInt("outRangeUniform", "0") # 出力値範囲を教師データ範囲と同一なるよう換算するかどうか
outHeadCut = configIni.getInt("outHeadCut", "0") # 出力値の先頭を切り捨てるかどうか、どうも先頭側が不安定になりやすいようだ
outDeltaMA = configIni.getInt("outDeltaMA", "5") # 出力値の微分値の移動平均サイズ
predLen = configIni.getInt("predLen", "1") # 未来予測のサンプル数
predAve = configIni.getInt("predAve", "1") # 未来予測分を平均化するかどうか
lossMag = configIni.getFloat("lossMag", "1") # 学習時のlossに掛ける係数
optm = configIni.getStr("optm", "Adam") # 勾配計算最適化オブジェクトタイプ
adamAlpha = configIni.getFloat("adamAlpha", "0.001") # Adamアルゴリズムのα値
adaDeltaRho = configIni.getFloat("adaDeltaRho", "0.95") # AdaDeltaアルゴリズムのrho値
adaDeltaEps = configIni.getFloat("adaDeltaEps", "0.000001") # AdaDeltaアルゴリズムのeps値

# その他グローバル変数初期化
inMA = (inMA // 2) * 2 + 1 # 入力値移動平均サイズを奇数にする
n_in = frameSize # ニューラルネットの入力次元数
n_out = 1 # ニューラルネットの出力次元数
predMeanK = np.ones(predLen) # 未来教師データの平均化係数
#predictionMeanK = np.arange(1.0 / predLen, 1.0, 1.0 / (predLen + 1))
#predictionMeanK *= predictionMeanK
predMeanK = predMeanK / predMeanK.sum()
outLen = bpropLen - outHeadCut # 出力データ長
outDeltaMA = (outDeltaMA // 2) * 2 + 1 # 出力値微分値の移動平サイズを奇数にする
outDeltaMAK = np.ones(outDeltaMA) / outDeltaMA # 出力値微分値の移動平均係数
outDeltaLen = outLen - outDeltaMA # 出力値微分値の移動平均後の長さ
retLen = outDeltaLen - 1 # クライアントに返す結果データ長
minPredLen = frameSize + (bpropLen - 1) * bpropStep # ドル円未来予測に必要な最小データ数
minEvalLen = minPredLen + predLen # 学習結果の評価に必要な最小データ数
fxYenData = np.zeros(1, dtype=np.float32) # MT4から送られる円データ、添え字は fxMinData と同じ
fxMinData = np.zeros(1, dtype=np.int32) # MT4から送られる分データ、添え字は fxYenData と同じ
fxYenDataTrain = None # 学習用の円データ、学習したいデータが更新されたら None 以外になる
fxYenPredictionModel = None # ドル円未来予測用のネットワークモデル
fxAveYenKs = np.zeros(4) # ドル円未来予測用データの合成係数
fxAveYenK = 0.0
fxLastD1YVals = None
fxLastD2YVals = None
modelLock = threading.Lock() # model を排他処理するためのロック

# ネットタイプと設定ファイル名によりモデルデータファイル名修飾文字列作成
# モデルファイル名に付与される
batchName = "btch" + str(batchSize) + ("rnd" if batchRandom else "")
testFileName = path.splitext(path.basename(args.iniFileName))[0] + "_" + str(netType) + "_" + optm + "_" + batchName + "_u" + str(numUnits) + "f" + str(frameSize) + "p" + str(predLen) + "b" + str(bpropLen) + "bs" + str(bpropStep)
if trainDataDummy:
	testFileName += "_" + trainDataDummy
modelFile = testFileName + ".model"
stateFile = testFileName + ".state"
testFileIni = ini.file(testFileName + ".ini", "DEFAULT")
curEpoch = testFileIni.getInt("curEpoch", 0) # 現在の実施済みエポック数取得

# 未来予測データの合成係数初期化
initAveYenKs(fxAveYenK)

# グラフ描画用の初期化
if grEnable:
	plt.ion() # 対話モードON
	fig = plt.figure() # 何も描画されていない新しいウィンドウを描画
	plt.xlabel("min") # x軸ラベル
	plt.ylabel("yen") # y軸ラベル
	plt.grid() # グリッド表示
	plt.gcf().canvas.set_window_title(testFileName)

	subPlot1 = fig.add_subplot(3, 1, 1)
	subPlot2 = fig.add_subplot(3, 1, 2)
	subPlot3 = fig.add_subplot(3, 1, 3)


	subPlot2.axvline(x=(outLen - 2) * bpropStep, color='black')
	subPlot3.axhline(y=0, color='black')

	if mode == "server":
		gxIn = np.arange(0, minPredLen, 1)
		gyIn = np.zeros(minPredLen)
	else:
		minEvalLen
		for i in range(bpropLen):
			subPlot1.axvline(x=i * bpropStep + frameSize, color='black')
		subPlot1.axvline(x=minPredLen, color='blue')
		gxIn = np.arange(0, minEvalLen, 1)
		gyIn = np.zeros(minEvalLen)
	gxOut = np.arange(0, outLen * bpropStep, bpropStep)
	gyOut = np.zeros(outLen)
	gxTeachDelta = np.arange(bpropStep, outLen * bpropStep, bpropStep)
	gyTeachDelta = np.zeros(outLen - 1)
	gxOutDelta = np.arange((outLen - outDeltaLen) * bpropStep, outLen * bpropStep, bpropStep)
	gyOutDelta = np.zeros(outDeltaLen)
	gxOutDelta2 = np.arange((outLen - outDeltaLen + 1) * bpropStep, outLen * bpropStep, bpropStep)
	gyOutDelta2 = np.zeros(outDeltaLen - 1)

	glIn, = subPlot1.plot(gxIn, gyIn, label="in")
	glTeach, = subPlot2.plot(gxOut, gyOut, label="trg")
	glOut, = subPlot2.plot(gxOut, gyOut, label="out")
	glTeachDelta, = subPlot3.plot(gxTeachDelta, gyTeachDelta, label="trg")
	glOutDelta, = subPlot3.plot(gxOutDelta, gyOutDelta, label="out", color='#5555ff')
	glOutDelta2, = subPlot3.plot(gxOutDelta2, gyOutDelta2, label="out", color='#ff55ff')
	glRet, = subPlot3.plot(gxOutDelta2, gyOutDelta2, label="outave", color='#ff0000')

	#subPlot1.legend(loc='lower right') # 凡例表示
	#subPlot2.legend(loc='lower right') # 凡例表示
	#subPlot3.legend(loc='lower right') # 凡例表示

# GPU使うならそれ用の数値処理ライブラリ取得
xp = cuda.cupy if gpu >= 0 else np

# ネットワークモデルとオプティマイザ初期化
dnn = DnnHelper()

# Prepare RNNLM model, defined in net.py
netClassDef = getattr(net, netType)
dnn.model = netClassDef(n_in, numUnits, n_out, gpu, True)
if netInitParamRandom:
	for param in dnn.model.params():
		data = param.data
		data[:] = np.random.uniform(-netInitParamRandom, netInitParamRandom, data.shape)
if gpu >= 0:
	cuda.get_device(gpu).use()
	dnn.model.to_gpu()

# Setup optimizer
if optm == "Adam":
	dnn.optimizer = optimizers.Adam(adamAlpha)
elif optm == "AdaDelta":
	dnn.optimizer = optimizers.AdaDelta(adaDeltaRho, adaDeltaEps)
else:
	print("Unknown optimizer: ", optm)
	sys.exit()
#dnn.optimizer = optimizers.MomentumSGD(lr=learningRrate, momentum=0.5)
#dnn.optimizer = optimizers.SGD(lr=learningRrate)
dnn.optimizer.setup(dnn.model)
dnn.optimizer.add_hook(chainer.optimizer.GradientClipping(gradClip))

# モデルとオプティマイザをロード
loadModelAndOptimizer()
