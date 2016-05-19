import csv
import codecs
import math
import random
import os.path as path
import numpy as np
import share as s

def readDataset(filename, inMA):
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
		elif s.trainDataDummy == "sin":
			# sin関数でダミーデータ作成
			data = []
			delta = math.pi / 100.0
			for i in range(3000):
				#t = math.sin(i * delta) + random.uniform(-0.05, 0.05)
				t = math.sin(i * delta)
				data.append(t)
				data.append(t)
				data.append(t)
				data.append(t)
		elif s.trainDataDummy == "sweep":
			# sin関数でダミーデータ作成
			data = []
			delta = math.pi / 100.0
			ddelta = delta / 1000.0
			for i in range(3000):
				t = math.sin(i * delta) + random.uniform(-0.05, 0.05)
				data.append(t)
				data.append(t)
				data.append(t)
				data.append(t)
				delta += ddelta
		else:
			# 円データをそのまま使用する
			dr = csv.reader(f)
			data = []
			for row in dr:
				data.append(float(row[2]))
				data.append(float(row[3]))
				data.append(float(row[4]))
				data.append(float(row[5]))

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
