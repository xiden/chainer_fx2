# -*- coding: utf-8 -*- 
#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
import csv
import numpy as np
from numba import jit

def read(filename):
	"""指定された分足為替CSVのクローズ時間を
	Args:
		filename: 読み込むCSVファイル名.
	"""

	with open(filename, "r") as f:
			# 円データをそのまま使用する
			dr = csv.reader(f)
			data = []
			for row in dr:
				data.append(float(row[5]))

	return np.asarray(data, dtype=np.float32)
	
data = read(sys.argv[1])
x = np.arange(0, data.shape[0], 1)
y = data
plt.plot(x, y)
plt.xlim(xmin=0, xmax=data.shape[0] - 1)
plt.show()
