#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""Sample script of recurrent neural network language model.

This code is ported from following implementation written in Torch.
https://github.com/tomsercu/lstm

"""
from __future__ import print_function
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
from chainer import cuda
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import serializers

import net
import candle
import win32api
import win32con
import os.path
import configparser
import share as s
import server

def train():
	"""学習モード処理"""

	global grEnable

	print('Train mode')

	# 学習ループ初期化
	print("Loading data from  " + s.trainDataFile)
	trainData = candle.read(s.trainDataFile, s.inMA)
	print("    length = {}".format(trainData.shape[0]))

	wholeLen = trainData.shape[0]
	batchRangeStart = 0
	batchRangeEnd = wholeLen - s.minEvalLen
	if batchRangeEnd < 0:
		print("Data length not enough")
		sys.exit()
	batchIndices = [0] * s.batchSize
	for i in range(s.batchSize):
		batchIndices[i] = batchRangeStart + i * (batchRangeEnd - batchRangeStart) // s.batchSize
	batchStart = 0
	batchStep = (batchRangeEnd - batchRangeStart) // s.batchSize
	batchOffsetInitial = batchRangeEnd - batchIndices[-1]
	batchOffset = batchOffsetInitial
	lastperp = -1.0
	samePerpCount = 0
	evalIndex = batchRangeStart
	evalIndexMove = s.frameSize // 5
	requestQuit = False
	lossMag = s.lossMag * s.outLen / (s.outLen - s.bpropHeadLossCut)
	print('going to train {} iterations'.format(s.epoch))

	# 学習ループ
	for i in six.moves.range(s.curEpoch, s.epoch):
		if requestQuit:
			break

		# LSTMによる一連の学習
		accumLoss = 0
		if s.batchRandom:
			batchStartIndices = np.random.randint(batchRangeStart, batchRangeEnd + 1, s.batchSize)
		else:
			batchStartIndices = np.asarray(batchIndices, dtype=np.integer)
			batchStartIndices += batchOffset
		for j in range(s.bpropLen):
			# 学習データと教師データ取得
			xa = s.xp.zeros(shape=(s.batchSize, s.n_in), dtype=np.float32)
			ta = s.xp.zeros(shape=(s.batchSize, s.n_out), dtype=np.float32)
			offset = j * s.bpropStep
			for bi in range(s.batchSize):
				xa[bi][:], ta[bi][:] = s.getTrainData(trainData, batchStartIndices[bi] + offset)
			x = chainer.Variable(xa)
			t = chainer.Variable(ta)

			# 学習実行
			y, loss = s.dnn.forward(x, t, True)
			if s.bpropHeadLossCut <= j:
				accumLoss += loss

			forceEval = False
			onlyAveDYVals = False

			# 予測データ合成係数変更
			if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD8) & 0x8000) != 0:
				s.fxAveYenK += 0.05
				s.initAveYenKs(s.fxAveYenK)
				forceEval = True
				onlyAveDYVals = True
			if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD5) & 0x8000) != 0:
				s.fxAveYenK -= 0.05
				s.initAveYenKs(s.fxAveYenK)
				forceEval = True
				onlyAveDYVals = True

			# 評価範囲移動速度変更
			if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD1) & 0x8000) != 0:
				evalIndexMove = 1
			if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD2) & 0x8000) != 0:
				evalIndexMove = s.frameSize // 5
			if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD3) & 0x8000) != 0:
				evalIndexMove = s.minEvalLen

			# 評価範囲移動処理
			if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD4) & 0x8000) != 0:
				evalIndex -= evalIndexMove
				forceEval = True
				if evalIndex < batchRangeStart:
					evalIndex = batchRangeStart
			if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD6) & 0x8000) != 0:
				evalIndex += evalIndexMove
				forceEval = True
				if batchRangeEnd < evalIndex:
					evalIndex = batchRangeEnd
			if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD7) & 0x8000) != 0:
				evalIndex = batchRangeStart
				forceEval = True
			if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD9) & 0x8000) != 0:
				evalIndex = batchRangeEnd
				forceEval = True

			# 評価処理
			if (j == 0 and i % s.evalInterval == 0) or forceEval:
				print('evaluate')
				now = time.time()
				perp = s.evaluate(trainData, evalIndex, onlyAveDYVals)
				print('epoch {} validation perplexity: {}'.format(s.curEpoch + 1, perp))
				if 1 <= i and s.optm == "Adam":
					print('learning rate =', s.dnn.optimizer.lr)

			# 終了判定処理
			if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD0) & 0x8000) != 0:
				requestQuit = True
				break

		# ニューラルネットワーク更新
		if not requestQuit:
			accumLoss *= lossMag
			s.dnn.update(accumLoss)

			# バッチ位置の移動
			batchOffset -= s.predLen
			if batchOffset < 0:
				batchOffset = batchOffsetInitial
				print("一周した")

			s.curEpoch = i + 1

			sys.stdout.flush()

	# INIファイルへ保存
	if s.optm == "Adam":
		s.configIni.set("learningRrate", s.dnn.optimizer.lr)

	# モデルとオプティマイザ保存
	s.saveModelAndOptimizer()

def test():
	# Evaluate on test dataset
	print('Test mode')

	print("Loading data from  " + s.trainDataFile)
	train_data = candle.read(s.trainDataFile, s.inMA)

	whole_len = train_data.shape[0]
	batchRangeStart = 0
	batchRangeEnd = whole_len - s.minEvalLen
	evalIndex = batchRangeStart
	evalIndexMove = s.frameSize // 5
	lastEvalIndex = -1

	while True:
		forceEval = False
		onlyAveDYVals = False

		# 予測データ合成係数変更
		if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD8) & 0x8000) != 0:
			s.fxAveYenK += 0.05
			s.initAveYenKs(s.fxAveYenK)
			forceEval = True
			onlyAveDYVals = True
		if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD5) & 0x8000) != 0:
			s.fxAveYenK -= 0.05
			s.initAveYenKs(s.fxAveYenK)
			forceEval = True
			onlyAveDYVals = True

		# 評価範囲移動速度変更
		if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD1) & 0x8000) != 0:
			evalIndexMove = 1
		if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD2) & 0x8000) != 0:
			evalIndexMove = s.frameSize // 5
		if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD3) & 0x8000) != 0:
			evalIndexMove = s.minEvalLen

		# 評価範囲移動処理
		if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD4) & 0x8000) != 0:
			evalIndex -= evalIndexMove
			if evalIndex < batchRangeStart:
				evalIndex = batchRangeStart
		if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD6) & 0x8000) != 0:
			evalIndex += evalIndexMove
			if batchRangeEnd < evalIndex:
				evalIndex = batchRangeEnd
		if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD7) & 0x8000) != 0:
			evalIndex = batchRangeStart
		if (win32api.GetAsyncKeyState(win32con.VK_NUMPAD9) & 0x8000) != 0:
			evalIndex = batchRangeEnd

		# 評価処理
		if evalIndex != lastEvalIndex or forceEval:
			lastEvalIndex = evalIndex
			test_perp = s.evaluate(train_data, evalIndex, onlyAveDYVals)
			print('test perplexity:', test_perp)
		plt.pause(0.01)

if s.mode == "train":
	train()
elif s.mode == "test":
	test()
elif s.mode == "server":
	sv = server.Server()
	sv.launch()
	input()
else:
	print("Unknown mode " + mode)
