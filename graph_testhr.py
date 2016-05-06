# -*- coding: utf-8 -*- 
#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
import csv
import numpy as np
from numba import jit

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

def read(filename):
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

def readOnlyY(filename):
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
	
ivals, tvals, yvals = read(sys.argv[1])

x = np.arange(0, ivals.shape[0], 1)
plt.plot(x, ivals, label="x")
plt.plot(x, tvals, label="t")
plt.plot(x, yvals, label="y " + sys.argv[1])
for i in range(2, len(sys.argv)):
    plt.plot(x, readOnlyY(sys.argv[i]), label="y " + sys.argv[i])

plt.legend(loc='lower left') # 凡例表示
plt.xlim(xmin=0, xmax=ivals.shape[0] - 1)
plt.show()
