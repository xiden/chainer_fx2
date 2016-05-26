import csv
import codecs
import math
import random
import os.path as path
import numpy as np
import share as s

def readDataset(filename, inMA, noise):
	"""指定された分足為替CSVからロウソク足データを作成する
	Args:
		filename: 読み込むCSVファイル名.
		Returns: 開始値配列、高値配列、低値配列、終値配列の2次元データ
	"""
	filename = path.join("Datasets", filename)
	print(filename)

	data = [[], [], [], []]

	with open(filename, "r") as f:
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
			for i in range(3000):
				t = 110.0 + math.sin(i * delta1) * 0.1
				data[0].append(t + random.uniform(-noise, noise))
				data[1].append(t + random.uniform(-noise, noise))
				data[2].append(t + random.uniform(-noise, noise))
				data[3].append(t + random.uniform(-noise, noise))
				data[1][i] = max([data[0][i], data[1][i], data[2][i], data[3][i]])
				data[2][i] = min([data[0][i], data[1][i], data[2][i], data[3][i]])
		elif s.trainDataDummy == "sweep":
			# sin関数でダミーデータ作成
			delta1 = math.pi / 100.0
			ddelta1 = delta1 / 100.0
			delta2 = math.pi / 70.0
			ddelta2 = delta2 / 70.0
			for i in range(3000):
				t = 110.0 + math.sin(i * delta1) * math.cos(i * delta2) * 0.1
				data[0].append(t + random.uniform(-noise, noise))
				data[1].append(t + random.uniform(-noise, noise))
				data[2].append(t + random.uniform(-noise, noise))
				data[3].append(t + random.uniform(-noise, noise))
				data[1][i] = max([data[0][i], data[1][i], data[2][i], data[3][i]])
				data[2][i] = min([data[0][i], data[1][i], data[2][i], data[3][i]])
				delta1 += ddelta1
				delta2 += ddelta2
		else:
			# 円データをそのまま使用する
			dr = csv.reader(f)
			for row in dr:
				o = float(row[2])
				h = float(row[3])
				l = float(row[4])
				c = float(row[5])
				if noise:
					o += random.uniform(-noise, noise)
					h += random.uniform(-noise, noise)
					l += random.uniform(-noise, noise)
					c += random.uniform(-noise, noise)
					if h < o: h = o
					if h < c: h = c
					if l > o: l = o
					if l > c: l = c
				data[0].append(o)
				data[1].append(h)
				data[2].append(l)
				data[3].append(c)

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
