#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import csv
import copy
import math
import numpy as np
from numba import jit
import share as s
import random

#encodedValueToPips = 1000.0
#@jit('f8(f8)')
#def encode(v):
#	return v
#@jit
#def encodeArray(v):
#	return v
#@jit('f8(f8)')
#def decode(v):
#	return v
#@jit
#def decodeArray(v):
#	return v


encodedValueToPips = 2000.0
@jit('f8(f8)')
def encode(v):
	return (v - 110.0) / 20.0
@jit
def encodeArray(v):
	return (v - 110.0) / 20.0
@jit('f8(f8)')
def decode(v):
	return v * 20.0 + 110.0
@jit
def decodeArray(v):
	return v * 20.0 + 110.0


def read(filename, inMA):
	"""指定された分足為替CSVからロウソク足データを作成する
	Args:
		filename: 読み込むCSVファイル名.
		Returns: [int]
	"""

	with open(filename, "r") as f:
		if s.trainDataDummy == "sin":
			# sin関数でダミーデータ作成
			data = []
			delta = math.pi / 100.0
			for i in range(300000):
				data.append(math.sin(i * delta) + random.uniform(-0.05, 0.05))
			return np.asarray(data, dtype=np.float32)
		elif s.trainDataDummy == "sweep":
			# sin関数でダミーデータ作成
			data = []
			delta = math.pi / 100.0
			ddelta = delta / 1000.0
			for i in range(300000):
				data.append(math.sin(i * delta) + random.uniform(-0.05, 0.05))
				delta += ddelta
			return np.asarray(data, dtype=np.float32)
		else:
			# 円データをそのまま使用する
			dr = csv.reader(f)
			data = []
			for row in dr:
				data.append(encode(float(row[5])))

		## 円変化量を使用する
		#dr = csv.reader(f)
		#data = []
		#lastVal = -10000.0
		#for row in dr:
		#	val = float(row[5])
		#	if lastVal != -10000.0:
		#		v = val - lastVal
		#		data.append(v)
		#		#print(v)
		#	lastVal = val

	# 指定されていたら移動平均を行う
	if inMA != 1:
		k = np.ones(inMA) / inMA
		return np.asarray(np.convolve(np.asarray(data), k, 'valid'), dtype=np.float32)
	else:
		return np.asarray(data, dtype=np.float32)
