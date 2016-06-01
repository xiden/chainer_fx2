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
maSize1 = 0 # 移動平均カーネルサイズ
maSize2 = 0 # 移動平均カーネルサイズ
kernel1 = None # 移動平均用カーネル
kernel2 = None # 移動平均用カーネル
maxMaLevel = 30 # 最大移動平均線数
dataset = None # グラフ描画するデータセット
datasetLength = 0 # dataset のデータ長
datasetX = None # dataset に対応するX座標値
dataMaH1 = None # dataset 内の高値の移動平均後データ
dataMaL1 = None # dataset 内の低値の移動平均後データ
dataMaH2 = None # dataset 内の高値の移動平均後データ
dataMaL2 = None # dataset 内の低値の移動平均後データ
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

def addMaLines(isLow, madata, label, color):
	"""
	移動平均ラインと注文または決済シグナルラインを追加、追加されたラインは管理リストにも追加される
	"""
	color = np.append(color, [1.0])
	white = np.copy(color)
	white[3] = 0.0

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
	global maSize1
	global maSize2
	global kernel1
	global kernel2

	configIni = ini.file(iniFileName, "NOAI")
	maLevel = configIni.getInt("maLevel", 0)
	leastSquaresLen = configIni.getInt("leastSquaresLen", 10)
	threshold = configIni.getFloat("threshold", 0.01)

	maSize1 = 3 + maLevel * 2
	maSize2 = 3 + (maLevel + 2) * 2
	kernel1 = gaussianKernel(maSize1, 3)
	kernel2 = gaussianKernel(maSize2, 3)

	s.minPredLen = s.frameSize # ドル円未来予測に必要な最小データ数
	s.minEvalLen = s.minPredLen # 学習結果の評価に必要な最小データ数

	# ニューラルネットの入力次元数
	s.dnnIn = 0
	# ニューラルネットの出力次元数
	s.dnnOut = 0

	s.fxRetLen = 6 # クライアントに返す結果データ長
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

def addDiff(data):
	"""指定されたデータの差分を加算し位相を進めたような効果を期待"""
	d = np.diff(data)
	return data[-d.shape[0]:] + d



def fxPrediction():
	"""
	現在の円データから予測する
	"""
	global dataset
	global datasetLength
	global datasetX
	global dataMaH1
	global dataMaL1
	global dataMaH2
	global dataMaL2
	global lines

	# 現在のデータ取得
	dataset = s.fxYenData[-s.frameSize:].transpose()
	datasetLength = dataset.shape[1]

	# 移動平均
	n1 = datasetLength if s.grEnable else maSize1 + 1
	n2 = datasetLength if s.grEnable else maSize2 + 1
	dataMaH1 = addDiff(np.convolve(dataset[1, :n1], kernel1, 'valid'))
	dataMaL1 = addDiff(np.convolve(dataset[2, :n1], kernel1, 'valid'))
	dataMaH2 = addDiff(np.convolve(dataset[1, :n2], kernel2, 'valid'))
	dataMaL2 = addDiff(np.convolve(dataset[2, :n2], kernel2, 'valid'))

	## 変化量で減ったデータを予測
	#n = maSize // 2
	#dataMaH = predByDiff(dataMaH, n)
	#dataMaL = predByDiff(dataMaL, n)

	## 最小二乗法で減ったデータを予測
	#xma = datasetX[-leastSquaresLen:]
	#ph, covariance = scipy.optimize.curve_fit(func, xma, dataMaH[-leastSquaresLen:], p0=np.zeros(3, dtype=np.float64))
	#pl, covariance = scipy.optimize.curve_fit(func, xma, dataMaL[-leastSquaresLen:], p0=np.zeros(3, dtype=np.float64))
	#xma = np.arange(datasetLength, datasetLength + maSize // 2, 1)
	#dataMaH = np.append(dataMaH, func(xma, ph[0], ph[1], ph[2]))
	#dataMaL = np.append(dataMaL, func(xma, pl[0], pl[1], pl[2]))

	# 必要ならグラフ表示を行う
	if s.grEnable:
		datasetX = np.arange(0, datasetLength, 1)
		for ln in lines:
			ln.remove()
		lines = []
		addLine(dataset[0], label="open", color=np.asarray([0, 0.5, 0.5], dtype=np.float64))
		addLine(dataset[1], label="high", color=np.asarray([1, 0, 0], dtype=np.float64))
		addLine(dataset[2], label="low", color=np.asarray([0, 0, 1], dtype=np.float64))
		addLine(dataset[3], label="close", color=np.asarray([0.5, 0.5, 0.5], dtype=np.float64))
		addMaLines(False, dataMaH1, label="high", color=np.asarray([1, 0, 0], dtype=np.float64))
		addMaLines(True, dataMaL1, label="low", color=np.asarray([0, 0, 1], dtype=np.float64))
		addMaLines(False, dataMaH2, label="high", color=np.asarray([0.5, 0, 0], dtype=np.float64))
		addMaLines(True, dataMaL2, label="low", color=np.asarray([0, 0, 0.5], dtype=np.float64))
		plt.ylim(f.npMaxMin([dataset[1], dataset[2]]))
		plt.legend(loc='lower left') # 凡例表示
		plt.draw()
		plt.pause(0.001)

	# 注文判定用の値を設定
	h = dataset[1,-1]
	l = dataset[2,-1]
	v1 = 0.0
	if h <= dataMaL1[-1] - threshold:
		# 高値の最終値が低値移動平均を下回ったら下がるんじゃないかな？
		v1 = -1.0
	elif dataMaH1[-1] + threshold <= l:
		# 低値の最終値が高値移動平均を上回ったら上がるんじゃないかな？
		v1 = 1.0

	# 決済判定用の値を設定
	v2 = 0.0
	if h <= dataMaL2[-1] - threshold:
		# 高値の最終値が低値移動平均を下回ったら下がるんじゃないかな？
		v2 = -1.0
	elif dataMaH2[-1] + threshold <= l:
		# 低値の最終値が高値移動平均を上回ったら上がるんじゃないかな？
		v2 = 1.0

	return np.asarray([v1, v1, v1, v2, v2, v2], dtype=np.float32)
