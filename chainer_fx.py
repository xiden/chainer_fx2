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

	# すでに目標エポック到達しているなら終了
	if s.epoch <= s.curEpoch:
		print("It have already reached the specified epoch")
		return

	# 学習ループ初期化
	print("Loading data from  " + s.trainDataFile)
	trainData = candle.read(s.trainDataFile, s.inMA)
	print("    length = {}".format(trainData.shape[0]))

	batchRangeStart, batchRangeEnd, batchIndices = s.getBatchRangeAndIndices(trainData.shape[0])
	batchStart = 0
	batchOffsetInitial = batchRangeEnd - batchIndices[-1]
	batchOffset = batchOffsetInitial
	lastperp = -1.0
	samePerpCount = 0
	evalIndex = batchRangeStart
	evalIndexMove = s.frameSize // 5
	requestQuit = False
	quitNow = False
	lossMag = s.lossMag * s.outLen / (s.outLen - s.bpropHeadLossCut)
	itrCount = batchOffsetInitial * s.epoch
	itr = 0
	print('going to train {} iterations'.format(itrCount))

	# 学習ループ
	startTime = time.time()
	while(s.curEpoch < s.epoch):
		# 終了要求状態で↓キー押されたら即座に終了
		if quitNow:
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
			if (j == 0 and itr % s.evalInterval == 0) or forceEval:
				print('evaluate')
				now = time.time()
				perp = s.evaluate(trainData, evalIndex, onlyAveDYVals)
				print('epoch {} validation perplexity: {}'.format(s.curEpoch + 1, perp))
				if 1 <= itr and s.optm == "Adam":
					print('learning rate =', s.dnn.optimizer.lr)

			# 終了判定処理
			if (win32api.GetAsyncKeyState(win32con.VK_NUMLOCK) & 0x8000) != 0:
				if not requestQuit:
					print("Quit requested. It will stop next epoch.")
					print("Press down arrow key to stop now.")
				requestQuit = True
				break
			if requestQuit and (win32api.GetAsyncKeyState(win32con.VK_DOWN) & 0x8000) != 0:
				quitNow = True

		# ニューラルネットワーク更新
		if not quitNow:
			accumLoss *= lossMag
			s.dnn.update(accumLoss)

			# 時間計測
			if itr % 10 == 0:
				curTime = time.time()
				print(10.0 / (curTime - startTime), "/s")
				startTime = curTime

			if batchOffset == 0:
				# 今回バッチ位置が０だったら一周とする
				batchOffset = batchOffsetInitial
				s.curEpoch += 1
				print("Current epoch is", s.curEpoch)

				# 終了要求状態なら終了する
				if requestQuit:
					break
			else:
				# バッチ位置の移動
				batchOffset -= s.bpropStep
				# 移動し過ぎを抑制
				if batchOffset < 0:
					batchOffset = 0

		itr += 1

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

	batchRangeStart, batchRangeEnd, batchIndices = s.getBatchRangeAndIndices(trainData.shape[0])
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
elif s.mode == "testhr":
	print('Hit rate test mode')
	print("Loading data from  " + s.trainDataFile)
	data = candle.read(s.trainDataFile, s.inMA)
	s.testhr(data, 0)
elif s.mode == "testhr_g":
	s.testhr_g()
else:
	print("Unknown mode " + mode)
