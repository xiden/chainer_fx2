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
else:
	print("Unknown mode " + s.mode)
