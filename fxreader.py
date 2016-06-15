import math
import random
import os.path as path
import numpy as np
import pandas as pd
from numba import jit
import share as s

@jit("void(f4[:,:])", nopython=True)
def normalizeAfterNoise(a):
	"""
	開始、高値、低値、終値に乱数を加えた後の正規化処理.
	"""
	n = a.shape[0]
	for i in range(n):
		a[i, 1] = a[i].max()
		a[i, 2] = a[i].min()

def readDataset(filename, inMA, noise):
	"""
	指定された分足為替CSVからロウソク足データを作成する

	Args:
		filename: 読み込むCSVファイル名.
		inMA: 移動平均サイズ.
		noise: 加えるノイズ量.

	Returns:
		開始値配列、高値配列、低値配列、終値配列の2次元データ.
	"""
	filename = path.join("Datasets", filename)
	print(filename)

	data = [[], [], [], []]

	if s.trainDataDummy == "line":
		# 直線データ作成
		for i in range(10000):
			data[0].append(i + random.uniform(-noise, noise))
			data[1].append(i + random.uniform(-noise, noise))
			data[2].append(i + random.uniform(-noise, noise))
			data[3].append(i + random.uniform(-noise, noise))
	elif s.trainDataDummy == "sin":
		# sin関数でダミーデータ作成
		delta1 = math.pi / 30.0
		for i in range(30000):
			t = 110.0 + math.sin(i * delta1) * 0.1
			data[0].append(t + random.uniform(-noise, noise))
			data[1].append(t + random.uniform(-noise, noise))
			data[2].append(t + random.uniform(-noise, noise))
			data[3].append(t + random.uniform(-noise, noise))
			data[1][i] = max([data[0][i], data[1][i], data[2][i], data[3][i]])
			data[2][i] = min([data[0][i], data[1][i], data[2][i], data[3][i]])
	elif s.trainDataDummy == "sweep":
		# sin関数でダミーデータ作成
		delta1 = math.pi / 1000.0
		ddelta1 = delta1 / 1000.0
		delta2 = math.pi / 570.0
		ddelta2 = delta2 / 570.0
		delta3 = math.pi / 15700.0
		ddelta3 = delta2 / 15700.0
		for i in range(30000):
			t = 110.0 + math.sin(i * delta1) * math.cos(i * delta2) * 0.1 + math.sin(i * delta3) * 2
			data[0].append(t + random.uniform(-noise, noise))
			data[1].append(t + random.uniform(-noise, noise))
			data[2].append(t + random.uniform(-noise, noise))
			data[3].append(t + random.uniform(-noise, noise))
			data[1][i] = max([data[0][i], data[1][i], data[2][i], data[3][i]])
			data[2][i] = min([data[0][i], data[1][i], data[2][i], data[3][i]])
			delta1 += ddelta1
			delta2 += ddelta2
			delta3 += ddelta3
	else:
		# CSVを一気に読み込む、めっちゃ速い
		df = pd.read_csv(filename, header=None)
		vals = df.values
		# numpy配列へコピー
		data = np.empty((vals.shape[0], 4), dtype=np.float32)
		data[:,:] = vals[:, 2: 6]
		# ノイズを加える
		if noise:
			data += np.random.uniform(-noise, noise, data.shape)
			normalizeAfterNoise(data)
		# 転置
		data = data.transpose()

	# numpy 配列にする
	data = np.asarray(data, dtype=np.float32)

	# 指定されていたら移動平均を行う
	if 3 <= inMA:
		ma2 = (inMA // 2) * 2
		inMA = ma2 + 1
		k = np.ones(inMA) / inMA
		src = data
		data = np.zeros((4, src.shape[1] - ma2), dtype=np.float)
		data[0,:] = np.convolve(src[0], k, 'valid')
		data[1,:] = np.convolve(src[1], k, 'valid')
		data[2,:] = np.convolve(src[2], k, 'valid')
		data[3,:] = np.convolve(src[3], k, 'valid')
	
	return data
