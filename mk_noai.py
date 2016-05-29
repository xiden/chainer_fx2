#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import numpy as np
import scipy.stats as st
import scipy.optimize
import matplotlib.pyplot as plt
import chainer
import ini
import fxreader
import share as s
import funcs as f

# グローバル変数
maLevel = 1 # 移動平均レベル、0が3、1で5、2で7・・・
leastSquaresLen=5 # 最小二乗法に使用する頂点数
threshold=0.01 # 売買判定オーバー閾値

lastLabel = None
lastColor = None
maSize = 0 # 移動平均カーネルサイズ
kernel = None # 移動平均用カーネル
maxMaLevel = 30 # 最大移動平均線数
dataset = None # グラフ描画するデータセット
datasetLength = 0 # dataset のデータ長
datasetX = None # dataset に対応するX座標値
dataMaH = None # dataset 内の高値の移動平均後データ
dataMaL = None # dataset 内の低値の移動平均後データ
lines = [] # ライン一覧

class Dnn(object):
	"""
	ダミー
	"""
	model = None
	optimizer = None

	def __init__(self, model = None, optimizer = None):
		self.model = model
		self.optimizer = optimizer

	def forward(self, x, volatile=chainer.flag.OFF):
		return 0.0

	def evaluate(self, x, t, volatile=chainer.flag.OFF):
		return 0.0

	def update(self, loss):
		pass

def gaussianKernel(maSize, sigma):
	"""
	移動平均用ガウシアンカーネルの計算
	"""
	maSize = (maSize // 2) * 2 + 1
	interval = sigma + (sigma + 0.5) / maSize
	k = np.diff(st.norm.cdf(np.linspace(-interval, interval, maSize + 1)))
	k /= k.sum()
	return k

def addLine(data, label=None, color=None):
	"""
	指定データのライン追加、追加されたラインは管理リストにも追加される
	"""
	global lastColor
	global lines

	if color is None:
		color = lastColor
	lastColor = color

	ln, = plt.plot(datasetX[datasetLength - data.shape[0]:], data, label=label, color=color, marker="o", markersize=2)
	lines.append(ln)

def addIntersectPoints(data, madata, madataIsLow, color):
	"""
	高値が低値の移動平均線より下がったまたは、低値が高値の移動平均線より上がったポイントに点を追加
	"""
	ofs = data.shape[0] - madata.shape[0]

	xvals = []
	yvals = []

	for i in range(madata.shape[0]):
		index = ofs + i
		p = data[index]
		m = madata[i]
		if (p <= m - threshold if madataIsLow else m + threshold <= p):
			xvals.append(index)
			yvals.append(p)

	if len(xvals) != 0:
		ln, = plt.plot(xvals, yvals, "o", color=color, markersize=8)
		lines.append(ln)

def addLines(data, madata, label, color):
	"""
	指定データと指定移動平均レベルのラインを追加、追加されたラインは管理リストにも追加される
	"""
	color = np.append(color, [1.0])
	white = np.copy(color)
	white[3] = 0.0

	isLow = madata is dataMaL
	addLine(data, label=label, color=color)
	addLine(madata, label=label, color=(white + color) / 2)
	addIntersectPoints(dataset[1 if isLow else 2], madata, isLow, color)

def readDataset(filename, inMA, noise):
	"""指定された分足為替CSVからロウソク足データを作成する
	Args:
		filename: 読み込むCSVファイル名.
		Returns: 開始値配列、高値配列、低値配列、終値配列の2次元データ
	"""
	return fxreader.readDataset(filename, inMA, noise)

def init(iniFileName):
	"""AI以外での予測用の初期化を行う"""
	global maLevel
	global leastSquaresLen
	global threshold
	global maSize
	global kernel

	configIni = ini.file(iniFileName, "NOAI")
	maLevel = configIni.getInt("maLevel", 1)
	leastSquaresLen = configIni.getInt("leastSquaresLen", 10)
	threshold = configIni.getFloat("threshold", 0.01)

	maSize = 3 + maLevel * 2
	kernel = gaussianKernel(maSize, 3)

	s.minPredLen = s.frameSize # ドル円未来予測に必要な最小データ数
	s.minEvalLen = s.minPredLen # 学習結果の評価に必要な最小データ数

	# ニューラルネットの入力次元数
	s.dnnIn = 0
	# ニューラルネットの出力次元数
	s.dnnOut = 0

	s.fxRetLen = 3 # クライアントに返す結果データ長
	s.fxInitialYenDataLen = s.minEvalLen # 初期化時にMT4から送る必要がある円データ数

def initGraph(windowCaption):
	# グラフ描画用の初期化
	if s.grEnable:
		plt.ion() # 対話モードON
		fig = plt.figure() # 何も描画されていない新しいウィンドウを描画
		plt.xlabel("min") # x軸ラベル
		plt.ylabel("yen") # y軸ラベル
		plt.grid() # グリッド表示
		plt.gcf().canvas.set_window_title(windowCaption)
		#plt.legend(loc='lower left') # 凡例表示

def getTestFileName(testFileName):
	return testFileName + "noai"

#@jit(nopython=True)
def trainBatch(dataset, itr):
	"""ダミー"""
	return 0

def func(x, a, b, c):
	x2 = x * x
	return a + b * x + c * x2

def predByDiff(data, n):
	"""4階微分で指定数のデータを予測し追加する"""
	d1 = np.diff(data[-5:])
	d2 = np.diff(d1)
	d3 = np.diff(d2)
	d4 = np.diff(d3)
	d1 = d1[-1]
	d2 = d2[-1]
	d3 = d3[-1]
	d4 = d4[-1]
	t = data[-1]
	l = data.shape[0]
	data = np.resize(data, (l + n,))
	for i in range(n):
		d1 += d2
		d2 += d3
		d3 += d4
		t += d1
		data[l + i] = t
	return data


def fxPrediction():
	"""
	現在の円データから予測する
	"""
	global dataset
	global datasetLength
	global datasetX
	global dataMaH
	global dataMaL
	global lines

	# 現在のデータ取得
	dataset = s.fxYenData[-s.frameSize:].transpose()
	datasetLength = dataset.shape[1]
	datasetX = np.arange(0, datasetLength, 1)

	# 移動平均
	dataMaH = np.convolve(dataset[1], kernel, 'valid')
	dataMaL = np.convolve(dataset[2], kernel, 'valid')

	# 変化量で減ったデータを予測
	n = maSize // 2
	dataMaH = predByDiff(dataMaH, n)
	dataMaL = predByDiff(dataMaL, n)

	## 最小二乗法で減ったデータを予測
	#xma = datasetX[-leastSquaresLen:]
	#ph, covariance = scipy.optimize.curve_fit(func, xma, dataMaH[-leastSquaresLen:], p0=np.zeros(3, dtype=np.float64))
	#pl, covariance = scipy.optimize.curve_fit(func, xma, dataMaL[-leastSquaresLen:], p0=np.zeros(3, dtype=np.float64))
	#xma = np.arange(datasetLength, datasetLength + maSize // 2, 1)
	#dataMaH = np.append(dataMaH, func(xma, ph[0], ph[1], ph[2]))
	#dataMaL = np.append(dataMaL, func(xma, pl[0], pl[1], pl[2]))

	# 必要ならグラフ表示を行う
	if s.grEnable:
		for ln in lines:
			ln.remove()
		lines = []
		addLine(dataset[0], label="open", color=np.asarray([0, 0.5, 0.5], dtype=np.float64))
		addLines(dataset[1], dataMaH, label="high", color=np.asarray([1, 0, 0], dtype=np.float64))
		addLines(dataset[2], dataMaL, label="low", color=np.asarray([0, 0, 1], dtype=np.float64))
		addLine(dataset[3], label="close", color=np.asarray([0.5, 0.5, 0.5], dtype=np.float64))
		plt.ylim(f.npMaxMin([dataset[1], dataset[2]]))
		plt.legend(loc='lower left') # 凡例表示
		plt.draw()
		plt.pause(0.001)

	value = 0.0
	if dataset[1,-1] <= dataMaL[-1] - threshold:
		# 高値の最終値が低値移動平均を下回ったら下がるんじゃないかな？
		value = -1.0
	elif dataMaH[-1] + threshold <= dataset[2,-1]:
		# 低値の最終値が高値移動平均を上回ったら上がるんじゃないかな？
		value = 1.0

	return np.asarray([value, value, value], dtype=np.float32)
