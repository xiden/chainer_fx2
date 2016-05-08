#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import share as s
import funcs as f
import server

if s.mode == "train":
	f.train()
elif s.mode == "server":
	sv = server.Server()
	sv.launch()
	input()
elif s.mode == "testhr":
	s.mk.testhr()
elif s.mode == "testhr_g":
	f.testhr_g()
else:
	print("Unknown mode " + mode)
