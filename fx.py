#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import share as s
import funcs as f
import server

if s.mode == "train":
	# 学習モード
	f.train()
elif s.mode == "server":
	# 予測サーバーモード
	sv = server.Server()
	sv.launch()
	input()
elif s.mode == "testhr":
	# 的中率計測モード
	s.mk.testhr()
elif s.mode == "trainhr":
	# 学習＆的中率計測モード
	f.train()
	s.mk.testhr()
elif s.mode == "testhr_g":
	# 的中率計測結果表示モード
	f.testhr_g()
elif s.mode == "plotw":
	# モデルの重み可視化
	f.plotw()
elif s.mode == "wdiff":
	# ２つの重みの差分の可視化
	f.wdiff()
elif s.mode == "wmov":
	# 重みの動画化
	f.wmov()
else:
	print("Unknown mode " + s.mode)
