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
		Returns: 開始値、高値、低値、終値が縦に並んでるイメージの2次元データ
	"""
	filename = path.join("Datasets", filename)
	print(filename)

	with open(filename, "r") as f:
		if s.trainDataDummy == "line":
			# 直線データ作成
			data = np.arange(0, 10000, 1)
			data += np.random.uniform(-noise, noise, data.shape)
		elif s.trainDataDummy == "sin":
			# sin関数でダミーデータ作成
			data = []
			delta = math.pi / 30.0
			for i in range(3000):
				#t = math.sin(i * delta) + random.uniform(-0.05, 0.05)
				t = 110.0 + math.sin(i * delta) * 10.0
				data.append(t + random.uniform(-noise, noise))
				data.append(t + random.uniform(-noise, noise))
				data.append(t + random.uniform(-noise, noise))
				data.append(t + random.uniform(-noise, noise))
		elif s.trainDataDummy == "sweep":
			# sin関数でダミーデータ作成
			data = []
			delta = math.pi / 100.0
			ddelta = delta / 1000.0
			for i in range(10000):
				t = 110.0 + math.sin(i * delta) * 1.0
				data.append(t + random.uniform(-noise, noise))
				data.append(t + random.uniform(-noise, noise))
				data.append(t + random.uniform(-noise, noise))
				data.append(t + random.uniform(-noise, noise))
				delta += ddelta
		else:
			# 円データをそのまま使用する
			dr = csv.reader(f)
			data = []
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
				data.append(o)
				data.append(h)
				data.append(l)
				data.append(c)

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
		k = np.ones(inMA * 4) / (inMA * 4)
		a = np.asarray(np.convolve(np.asarray(data), k, 'valid'), dtype=np.float32)
	else:
		a = np.asarray(data, dtype=np.float32)
	return np.reshape(a, (a.shape[0] / 4, 4))
