#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import sys
import csv
import re
import codecs
import shutil
import os
import os.path as path
import win32api
import win32con
import time
import math
from pathlib import Path
from numba import jit
import numpy as np
import chainer.cuda as cuda
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import share as s

def calcGraphLayout(n):
	"""
	指定数のグラフを表示するのに丁度いい縦横グラフ数の取得.
	"""
	nh = int(math.ceil(math.sqrt(n / (1920 / 1080))))
	nw = int(math.ceil(n / nh))
	return nh, nw

def getEpochDir(epoch):
	"""
	指定エポック番号の保存用ディレクトリ名取得.
	"""
	return path.join(s.resultTestDir, "e" + str(epoch))

def mkEpochDir(epoch):
	"""
	指定エポック番号の保存用ディレクトリ作成.
	"""
	epochDir = getEpochDir(epoch)
	if not path.isdir(epochDir):
		os.mkdir(epochDir)
	return epochDir

def getTestHrGraphFileBase():
	"""
	的中率計測結果グラフファイル名のベース名
	"""
	return "g_" + s.trainDataFile + "_"

def getTestHrStatFileBase():
	"""
	的中率計測結果統計値ファイル名のベース名
	"""
	return "a_" + s.trainDataFile

def getEpochs(reverse=True):
	"""
	保存データが存在するエポックの番号リストを取得.
	"""
	# 保存してあるエポックディレクトリ取得
	p = Path(s.resultTestDir)
	pls = list(p.glob("e[0-9]*"))
	epochs = []
	for pl in pls:
		name = pl.name
		if path.isdir(path.join(s.resultTestDir, name)):
			epochs.append(int(name[1:]))
	epochs.sort(reverse=reverse)
	return epochs

def loadWeights(dir1, dir2=None, lname=None):
	"""
	重みデータを読み込む.

	Args:
		dir1: エポックの重み保存ディレクトリその１.
		dir2: エポックの重み保存ディレクトリその２、省略可能.
		lname: 特定のレイヤ重みのみ読み込みたい場合にレイヤ名を指定する.

	Returns:
		(レイヤ名, 重み1, 重み2)タプルリスト.
	"""
	weights = []
	p = Path(dir1)
	pls = list(p.glob("*.w.npy" if lname is None else lname + ".w.npy"))
	for pl in pls:
		name = pl.name
		a1 = np.load(path.join(dir1, name))
		a2 = None if dir2 is None else np.load(path.join(dir2, name))
		weights.append((name[:-6], a1, a2))
	weights.sort(key=lambda k: k[0])
	return weights

def loadTrainDataset():
	"""
	学習用データセットを読み込む
	"""
	if s.sharedTrainDataset is None:
		print("Loading data from  " + s.trainDataFile)
		s.sharedTrainDataset = dataset = s.mk.readDataset(s.trainDataFile, s.inMA, s.datasetNoise)
		print("    length = {}".format(dataset.shape[1]))
	else:
		dataset = s.sharedTrainDataset
	return dataset

def makeWeightFig(e1, e2=None, lname=None, epochs=None, lastFigAxesIms=None):
	"""
	重みグラフを作成する.

	Args:
		e1: エポック１のインデックス.
		e2: エポック２のインデックスを指定するとエポック１との差分が表示される.
		lname: 特定のレイヤのみ表示したい場合にレイヤ名を指定する.
		epochs: 既にエポック番号リスト取得済みの場合に指定する.
		lastFigAxesIms: 前回作った (fig, axes, ims) を再利用する場合に指定する.

	Returns:
		(fig, axes, ims) のタプル
	"""

	# 保存してあるエポックディレクトリ取得
	if epochs is None:
		epochs = getEpochs()

	# 重みデータリスト読み込み
	dir1 = getEpochDir(epochs[e1])
	dir2 = None if e2 is None else getEpochDir(epochs[e2])
	weights = loadWeights(dir1, dir2, lname)

	# グラフ作成
	n = len(weights)
	nh, nw = calcGraphLayout(n)
	nn = nh * nw
	if lastFigAxesIms is None:
		fig, axes = plt.subplots(nh, nw, figsize=(16, 9), dpi=110, facecolor='w', edgecolor='k')
		fig.subplots_adjust(top=0.97, bottom=0.0, left=0.0, right=1.0, wspace=0.0, hspace=0.15)
		ims = []
		if e2 is None:
			fig.canvas.set_window_title("Epoch {0}".format(epochs[e1]))
		else:
			fig.canvas.set_window_title("Epoch {0} - {1}".format(epochs[e1], epochs[e2]))
	else:
		fig = lastFigAxesIms[0]
		axes = lastFigAxesIms[1]
		ims = lastFigAxesIms[2]

	for i in range(nn):
		ax = axes if n == 1 else axes[i // nw, i % nw]
		if i < n:
			w = weights[i]

			ax.set_title(w[0])

			w1 = w[1]
			w2 = w[2]
			if len(w1.shape) != 2:
				w1 = w1[0]
				if w2 is not None:
					w2 = w2[0]

			data = w1 if w2 is None else w1 - w2

			if len(ims) <= i:
				im = ax.imshow(data, interpolation="nearest")
				im.autoscale()
				ims.append(im)
				ax.xaxis.set_visible(False)
				ax.yaxis.set_visible(False)
				ax.axis("image")
				if n == 1:
					divider = make_axes_locatable(ax)
					cax = divider.append_axes("right", size="5%", pad=0.05)
					cbar = plt.colorbar(im, cax=cax)
			else:
				im = ims[i]
				im.set_data(data)
				im.autoscale()
		else:
			ax.set_axis_off()

	return (fig, axes, ims)

def makeTeachDataset(trainDataset):
	"""
	学習用データセットから教師データセットを作成する
	"""
	if s.sharedTeachDataset is None:
		print("Making teach dataset")
		s.sharedTeachDataset = dataset = s.mk.makeTeachDataset(trainDataset)
	else:
		dataset = s.sharedTeachDataset
	return dataset

def trainInit(trainDataset):
	"""
	指定された長さの学習データで学習に必要な変数を初期化する
	"""
	s.batchRangeStart = 0
	s.batchRangeEnd = trainDataset.shape[1] - s.minEvalLen
	if s.batchRangeEnd < 0:
		print("Data length not enough")
		sys.exit()
	s.batchRangeSize = s.batchRangeEnd - s.batchRangeStart
	s.batchIndices = [0] * s.batchSize
	for i in range(s.batchSize):
		s.batchIndices[i] = s.batchRangeStart + i * s.batchRangeSize // s.batchSize
	s.batchOffsetInitial = s.batchRangeEnd - s.batchIndices[-1]
	s.batchOffset = s.batchOffsetInitial
	s.evalIndex = s.batchRangeStart
	s.evalIndexMove = s.frameSize // 5
	s.requestQuit = False
	s.quitNow = False
	s.forceEval = False

def serverTrainInit(wholeLen):
	"""
	サーバー用に指定された長さの学習データで学習に必要な変数を初期化する
	"""
	s.batchRangeEnd = wholeLen - s.minEvalLen
	s.batchIndices = [0] * s.batchSize
	for i in range(s.batchSize):
		s.batchIndices[i] = s.batchSize * i
	s.batchOffsetInitial = s.batchRangeEnd - s.batchIndices[-1]
	s.batchOffset = s.batchOffsetInitial

def snapShotPredictionModel():
	"""
	学習中のモデルからドル円未来予測用のモデルを作成する
	"""
	e = s.dnn.model.copy()  # to use different state
	e.reset_state()  # initialize state
	e.train = False  # dropout does nothing
	s.fxYenPredictionModel = e

#@jit
def npMaxMin(arrays):
	"""
	指定された複数の配列の最大最小を取得する
	"""
	rmax = float(arrays[0].max())
	rmin = float(arrays[0].min())
	for i in range(1, len(arrays)):
		tmax = float(arrays[i].max())
		tmin = float(arrays[i].min())
		if rmax < tmax: rmax = tmax
		if tmin < rmin: rmin = tmin
	return rmin, rmax

def writeTestHrGraphCsv(xvals, tvals, yvals):
	"""
	テスト結果CSVファイルに書き込む
	"""
	fname = path.join(s.resultHrDir, getTestHrGraphFileBase() + str(s.curEpoch) + ".csv")
	with codecs.open(fname, 'w', "shift_jis") as file:
		writer = csv.writer(file)
		for i in range(tvals.shape[0]):
			writer.writerow([xvals[0][i], xvals[1][i], xvals[2][i], xvals[3][i], tvals[i], yvals[i]])

def writeTestHrStatCsv(epoch, hitRate, nonZeroHitRate, sameDirRate, distance):
	"""
	的中率統計CSVファイルへ書き込む.

	Args:
		epoch: エポック数.
		hitrate: 的中率%.
	"""
	fname = path.join(s.resultHrDir, getTestHrStatFileBase() + ".csv")
	fileExists = path.isfile(fname)
	with codecs.open(fname, 'a', "shift_jis") as file:
		writer = csv.writer(file)
		if not fileExists:
			writer.writerow(["epoch", "hit rate[%]", "non zero hit rate[%]", "same dir rate[%]", "rms err"])
		writer.writerow([epoch, hitRate, nonZeroHitRate, sameDirRate, distance])

def readTestHrCsv(filename):
	"""
	指定された的中率テスト結果CSVを読み込む

	Args:
		filename: 読み込むCSVファイル名.
	"""
	with open(filename, "r") as f:
		# 円データをそのまま使用する
		dr = csv.reader(f)
		xdata = [[], [], [], []]
		tdata = []
		ydata = []
		for row in dr:
			xdata[0].append(float(row[0]))
			xdata[1].append(float(row[1]))
			xdata[2].append(float(row[2]))
			xdata[3].append(float(row[3]))
			tdata.append(float(row[4]))
			ydata.append(float(row[5]))
	return np.asarray(xdata, dtype=np.float32), np.asarray(tdata, dtype=np.float32), np.asarray(ydata, dtype=np.float32)

def readTestHrGraphY(filename):
	"""
	指定された的中率テスト結果CSV内の計算出力だけを読み込む

	Args:
		filename: 読み込むCSVファイル名.
	"""
	with open(filename, "r") as f:
		# 円データをそのまま使用する
		dr = csv.reader(f)
		ydata = []
		for row in dr:
			ydata.append(float(row[2]))
	return np.asarray(ydata, dtype=np.float32)

def testhr_g():
	baseName = getTestHrGraphFileBase()
	p = Path(s.resultTestDir)
	l = list(p.glob(baseName + "*.csv"))
	i = len(baseName)
	epochs = []
	for pl in l:
		name, ext = path.splitext(pl.name)
		epochs.append(int(name[i:]))

	epochs.sort()
	count = 1
	for epoch in epochs:
		csvFile = path.join(s.resultTestDir, baseName + str(epoch) + ".csv")

		if count == 1:
			xvals, tvals, yvals = readTestHrCsv(csvFile)
			x = np.arange(0, tvals.shape[0], 1)
			plt.plot(x, xvals[0], label="x open")
			plt.plot(x, xvals[1], label="x high")
			plt.plot(x, xvals[2], label="x low")
			plt.plot(x, xvals[3], label="x close")
			plt.plot(x, tvals, label="t")
			plt.plot(x, yvals, label="y " + str(epoch))
		else:
			plt.plot(x, readTestHrGraphY(csvFile), label="y " + str(epoch))

		count += 1

	# クラス分類版のデータなら入力値の平均に水平線を引く
	if s.model.getModelKind() == "clas":
		xvalsAverage = np.average(xvals)
		plt.axhline(y=xvalsAverage, color='black')

	plt.gcf().canvas.set_window_title(s.trainDataFile)
	plt.legend(loc='lower left') # 凡例表示
	plt.xlim(xmin=0, xmax=tvals.shape[0] - 1)
	plt.show()

#@jit
def trainFlowControl():
	"""
	ユーザー入力による学習処理時の評価位置移動、終了要求などの共通制御処理
	"""
	if s.grEnable:
		# 評価範囲移動速度変更
		if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD1) & 0x8000) != 0:
			s.evalIndexMove = 1
		if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD2) & 0x8000) != 0:
			s.evalIndexMove = s.frameSize // 5
		if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD3) & 0x8000) != 0:
			s.evalIndexMove = s.minEvalLen

		# 評価範囲移動処理
		if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD4) & 0x8000) != 0:
			s.evalIndex -= s.evalIndexMove
			s.forceEval = True
			if s.evalIndex < s.batchRangeStart:
				s.evalIndex = s.batchRangeStart
		if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD6) & 0x8000) != 0:
			s.evalIndex += s.evalIndexMove
			s.forceEval = True
			if s.batchRangeEnd < s.evalIndex:
				s.evalIndex = s.batchRangeEnd
		if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD7) & 0x8000) != 0:
			s.evalIndex = s.batchRangeStart
			s.forceEval = True
		if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD9) & 0x8000) != 0:
			s.evalIndex = s.batchRangeEnd
			s.forceEval = True

		# 評価強制
		if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD8) & 0x8000) != 0:
			s.forceEval = True

	# 終了判定処理
	if (win32api.GetAsyncKeyState(win32con.VK_NUMLOCK) & 0x8000) != 0:
		if not s.requestQuit:
			print("Quit requested. It will stop next epoch.")
			print("Press pause key to stop now.")
			s.requestQuit = True
	if s.requestQuit and (win32api.GetAsyncKeyState(win32con.VK_PAUSE) & 0x8000) != 0:
		s.quitNow = True

#@jit
def train():
	"""
	学習モード処理
	"""

	print('Train mode', s.netType)

	# すでに目標エポック到達しているなら終了
	if s.epoch <= s.curEpoch:
		print("It have already reached the specified epoch")
		return

	# 学習データ読み込み
	trainDataset = loadTrainDataset()
	# 教師データ作成
	teachDataset = makeTeachDataset(trainDataset)

	# 学習ループ関係変数初期化
	trainInit(trainDataset)
	itrStart = s.batchOffsetInitial * s.curEpoch
	itrEnd = s.batchOffsetInitial * s.epoch
	itrCount = itrEnd - itrStart
	itr = 0

	# 学習ループ
	print('going to train {} iterations'.format(itrEnd))
	startTime = prevTime = time.time()
	while(s.curEpoch < s.epoch):
		# 即時終了要求があったら抜ける
		if s.quitNow:
			break

		# バッチ位置調整
		if s.batchRandom:
			s.batchStartIndices = np.random.randint(s.batchRangeStart, s.batchRangeEnd + 1, s.batchSize)
		else:
			s.batchStartIndices = np.asarray(s.batchIndices, dtype=np.integer)
			s.batchStartIndices += s.batchOffset

		# バッチ学習
		accumLoss = s.mk.trainBatch(trainDataset, teachDataset, itr)

		# ニューラルネットワーク更新
		if not s.quitNow:
			s.dnn.update(accumLoss)
			del(accumLoss)

			# 時間計測＆残り時間表示
			if itr % s.itrCountInterval == 0:
				curTime = time.time()
				elpTime = curTime - startTime
				endTime = elpTime * itrCount / itr if itr else 0.0
				print("{0:.2f}itr/s : remain {1:.2f}s".format(s.itrCountInterval / (curTime - prevTime), endTime - elpTime))
				prevTime = curTime

			if s.batchOffset == 0:
				# 今回バッチ位置が０だったら一周とする
				s.batchOffset = s.batchOffsetInitial
				s.curEpoch += 1
				print("Current epoch is", s.curEpoch)

				# 終了要求状態なら終了する
				if s.requestQuit:
					break
			else:
				# バッチ位置の移動
				s.batchOffset -= 1
				# 移動し過ぎを抑制
				if s.batchOffset < 0:
					s.batchOffset = 0

		itr += 1

	# モデルとオプティマイザ保存
	s.saveModelAndOptimizer()

	# エポック別フォルダへコピー
	print("Saving current epoch.")
	epochDir = mkEpochDir(s.curEpoch)
	shutil.copy(s.configFileName, epochDir)
	shutil.copy(s.testFilePath + ".ini", epochDir)
	if s.backupEpoch:
		shutil.copy(s.modelFile, epochDir)
		shutil.copy(s.stateFile, epochDir)

def plotw():
	"""
	モデルの重みを可視化＆ファイル保存
	"""

	print('Weight plot mode')

	s.loadModel(s.modelFile)
	model = s.dnn.model
	links = []
	for l in model.links(skipself=True):
		if "W" in l.__dict__:
			links.append(l)

	# 重みを画像として表示する
	if s.grEnable:
		links = sorted(links, key=lambda x: x.name)
		n = len(links)
		nh, nw = calcGraphLayout(n)
		nn = nh * nw

		fig, axes = plt.subplots(nh, nw)
		fig.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0, wspace=0.0, hspace=0.0)
		fig.canvas.set_window_title("Epoch " + str(s.curEpoch))

		for i in range(nn):
			ax = axes[i // nw, i % nw]
			l = links[i] if i < n else None
			if l is not None:
				ax.set_title(l.name)
				d = l.W.data
				ax.imshow(d if len(d.shape) == 2 else d[0], interpolation="nearest")
				ax.xaxis.set_visible(False)
				ax.yaxis.set_visible(False)
				ax.axis("image")
			else:
				ax.set_axis_off()
		plt.show()

	# 重みをファイルに保存する
	epochDir = mkEpochDir(s.curEpoch)
	for l in links:
		np.save(path.join(epochDir, l.name + ".w"), l.W.data)
		np.save(path.join(epochDir, l.name + ".b"), l.b.data)

def isInteger(value):
	"""
	整数値かどうかの検証.
	"""
	return re.match(r'^(?![-+]0*$)[-+]?([1-9][0-9]*)?0?$', '%s'%value) and True or False

def parseWeightOptions(o):
	"""
	重み表示関係のオプションを解析する.

	Args:
		o: オプション文字列.

	Returns:
		(エポックインデックス配列, レイヤ名配列, オプション配列) タプル
	"""
	fields = o.split(",")

	epochs = []
	lnames = []
	options = []

	for f in fields:
		if isInteger(f):
			epochs.append(int(f))
		elif f.startswith(":"):
			lnames.append(f[1:])
		else:
			options.append(f)

	return (epochs, lnames, options)

def wdiff():
	"""
	保存してある重みの差分を表示する.
	"""

	print('Weight diff mode')

	# 重み表示オプション取得
	wo = parseWeightOptions(s.wdiff)
	e1 = wo[0][0]
	e2 = wo[0][1] if 2 <= len(wo[0]) else None
	lname = wo[1][0] if len(wo[1]) != 0 else None

	# グラフ作成して表示
	fig, axes, ims = makeWeightFig(e1, e2, lname)
	plt.show()

def wmov():
	"""
	保存してある重みを動画化する.
	"""

	print('Create weight movie mode')

	# エポック一覧取得
	epochs = getEpochs(False)

	# 作業ディレクトリ作成
	wdir = path.join(s.resultTestDir, 'movie_tmp')
	if path.isdir(wdir):
		# 一旦削除
		shutil.rmtree(wdir)
	os.mkdir(wdir)

	# 重み表示オプション取得
	wo = parseWeightOptions(s.wmov)
	lname = wo[1][0] if len(wo[1]) != 0 else None
	diff = "diff" in wo[2]

	# 各エポックの重み画像を作業ディレクトリに保存していく
	n = len(epochs)
	ctx = None
	if diff:
		n -= 1
		for i in range(n):
			print("epoch", epochs[i + 1], "-", epochs[i])
			ctx = makeWeightFig(i + 1, i, lname=lname, epochs=epochs, lastFigAxesIms=ctx)
			ctx[0].savefig(path.join(wdir, "%05d.png" % (i)))
	else:
		for i in range(n):
			print("epoch", epochs[i])
			ctx = makeWeightFig(i, lname=lname, epochs=epochs, lastFigAxesIms=ctx)
			ctx[0].savefig(path.join(wdir, "%05d.png" % (i)))

	# mp4作成
	print("Creating movie...")
	movfname = "all.mp4" if lname is None else lname + ".mp4"
	if diff:
		movfname = "diff_" + movfname
	os.system("ffmpeg.exe -y -r 10 -i " + wdir + "\\%05d.png -c:v libx264 -crf 10 -preset ultrafast -pix_fmt yuv420p -threads 0 -tune zerolatency -f mpegts " + path.join(s.resultTestDir, movfname))

	# 作業ディレクトリ削除
	if "keep" not in s.wmov.split(","):
		shutil.rmtree(wdir)
