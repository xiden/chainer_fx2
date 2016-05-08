﻿#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import sys
import csv
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import os.path as path
from pathlib import Path
import win32api
import win32con
import time
import share as s


def trainInit(wholeLen):
	"""指定された長さの学習データで必要変数を初期化する"""

	s.batchRangeStart = 0
	s.batchRangeEnd = wholeLen - s.minEvalLen
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

def snapShotPredictionModel():
	"""学習中のモデルからドル円未来予測用のモデルを作成する"""
	e = s.dnn.model.copy()  # to use different state
	e.reset_state()  # initialize state
	e.train = False  # dropout does nothing
	s.fxYenPredictionModel = e

@jit
def getFxYenData(dataset, i):
	"""ドル円データ取得"""
	tstart = i + s.frameSize
	tend = tstart + s.predLen
	if tend <= dataset.shape[0]:
		t = (dataset[tstart : tend] * s.predMeanK).sum()
	else:
		t = None
	return s.xp.asarray([dataset[i : tstart]], dtype=np.float32), t

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


def getTestHrFileBase():
	return s.testFileName + "_" + s.trainDataFile + "_testhr_"


def readTestHrGraphBase(filename):
	"""指定された的中率テスト結果CSVを読み込む
	Args:
		filename: 読み込むCSVファイル名.
	"""
	with open(filename, "r") as f:
		# 円データをそのまま使用する
		dr = csv.reader(f)
		idata = []
		tdata = []
		ydata = []
		for row in dr:
			idata.append(float(row[0]))
			tdata.append(float(row[1]))
			ydata.append(float(row[2]))
	return np.asarray(idata, dtype=np.float32), np.asarray(tdata, dtype=np.float32), np.asarray(ydata, dtype=np.float32)

def readTestHrGraphY(filename):
	"""指定された的中率テスト結果CSV内の計算出力だけを読み込む
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
	baseName = getTestHrFileBase()
	p = Path(".")
	l = list(p.glob(baseName + "*.csv"))
	i = len(baseName)
	epochs = []
	for pl in l:
		name, ext = path.splitext(pl.name)
		epochs.append(int(name[i:]))

	epochs.sort()
	count = 1
	for epoch in epochs:
		csvFile = baseName + str(epoch) + ".csv"

		if count == 1:
			xvals, tvals, yvals = readTestHrGraphBase(csvFile)
			x = np.arange(0, xvals.shape[0], 1)
			plt.plot(x, xvals, label="x")
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
	plt.xlim(xmin=0, xmax=xvals.shape[0] - 1)
	plt.show()

def trainFlowControl():
	"""ユーザー入力による学習処理時の評価位置移動、終了要求などの共通制御処理"""

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

	# 終了判定処理
	if (win32api.GetAsyncKeyState(win32con.VK_NUMLOCK) & 0x8000) != 0:
		if not s.requestQuit:
			print("Quit requested. It will stop next epoch.")
			print("Press pause key to stop now.")
			s.requestQuit = True
	if s.requestQuit and (win32api.GetAsyncKeyState(win32con.VK_PAUSE) & 0x8000) != 0:
		s.quitNow = True

def train():
	"""学習モード処理"""

	print('Train mode')

	# すでに目標エポック到達しているなら終了
	if s.epoch <= s.curEpoch:
		print("It have already reached the specified epoch")
		return

	# 学習データ読み込み
	print("Loading data from  " + s.trainDataFile)
	dataset = s.mk.read(s.trainDataFile, s.inMA)
	print("    length = {}".format(dataset.shape[0]))

	# 学習ループ関係変数初期化
	trainInit(dataset.shape[0])
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
		accumLoss = s.mk.trainBatch(dataset, itr)

		# ニューラルネットワーク更新
		if not s.quitNow:
			s.dnn.update(accumLoss)

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

	# INIファイルへ保存
	if s.optm == "Adam":
		s.configIni.set("learningRrate", s.dnn.optimizer.lr)

	# モデルとオプティマイザ保存
	s.saveModelAndOptimizer()