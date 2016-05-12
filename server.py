#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import socket
from contextlib import closing
import struct
import numpy as np
import threading
import time
from numba import jit
import share as s
import traceback

@jit
def recvI8Arr(sock, buf, count):
	bufLen = buf.shape[0]
	if bufLen < count:
		raise IndexError
	toRead = count
	i = 0
	while toRead != 0:
		n = sock.recv_into(buf[i: bufLen], toRead)
		if n == 0:
			return None
		toRead -= n
		i += n
	return buf[0 : count]

@jit
def recvI32(sock, buf):
	bufLen = buf.shape[0]
	toRead = 4
	i = 0
	while toRead != 0:
		n = sock.recv_into(buf[i: bufLen], toRead)
		if n == 0:
			return None
		toRead -= n
		i += n
	return int(buf[0 : 4].view(dtype=np.int32))

@jit
def recvI32Arr(sock, buf, count):
	toread = 4 * count
	buflen = buf.shape[0]
	i = 0
	while toread != 0:
		n = sock.recv_into(buf[i: buflen], toread)
		if n == 0:
			return None
		toread -= n
		i += n
	return np.frombuffer(buf, count=count, dtype=np.int32)

@jit
def recvF32Arr(sock, buf, count):
	toread = 4 * count
	buflen = buf.shape[0]
	i = 0
	while toread != 0:
		n = sock.recv_into(buf[i: buflen], toread)
		if n == 0:
			return None
		toread -= n
		i += n
	return np.frombuffer(buf, count=count, dtype=np.float32)

@jit
def recvPacket(sock, buf):
	size = recvI32(sock, buf)
	if size is None:
		return None
	if size < 4:
		return None
	return recvI8Arr(sock, buf, size)

class Server(threading.Thread):
	"""クライアントへ予測情報を提供するサーバー"""

	def __init__(self):
		threading.Thread.__init__(self)

		self.host = '127.0.0.1'
		self.port = 4000
		self.backlog = 10
		self.buf = np.zeros(2 ** 20, dtype=np.int8)
		self.acceptanceSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

	def launch(self):
		# とりあえず現状のモデルをドル円予測で使える様にコピー
		s.snapShotPredictionModel()

		# 学習用スレッド開始
		self.start()

		with closing(self.acceptanceSock):
			# 接続受付ループ
			self.acceptanceSock.bind((self.host, self.port))
			self.acceptanceSock.listen(self.backlog)

			while True:
				try:
					# 接続待ち
					print("waitin connection...")
					conn, address = self.acceptanceSock.accept()
					print("connected from {}".format(address))
					with closing(conn):
						while True:
							if not self.procFunc(conn):
								break

				except Exception as e:
					#print('=== エラー内容 ===')
					#print('type:' + str(type(e)))
					#print('args:' + str(e.args))
					#print('message:' + e.message)
					#print('e自身:' + str(e))
					print(traceback.format_exc())

				## サーバークラッシュしても大丈夫なように
				## モデルとオプティマイザを保存する
				#s.saveModelAndOptimizer()

			self.acceptanceSock.shutdown(socket.SHUT_RDWR)
			self.acceptanceSock.close()

	@jit
	def run(self):
		#s.trainFxYen()
		pass

	@jit
	def procFunc(self, conn):
		# 命令パケット受け取り
		pkt = recvPacket(conn, self.buf)
		if pkt is None:
			return 0

		# コマンド種類取得
		size = pkt.shape[0]
		if size < 4:
			return 0
		cmdType = int(pkt[0 : 4].view(dtype=np.int32))
		ipkt = 4
		size -= 4

		# コマンド種類ごとに処理を分ける
		if cmdType == 0:
			# 終了
			conn.send(np.asarray(1, dtype=np.int32))
			conn.shutdown(socket.SHUT_RDWR)
			conn.close()
			return 0
		if cmdType == 1:
			# 初期化に必要なデータ数とサーバーが返す予測データ数の取得
			a = np.zeros(2, dtype=np.int32)
			a[0] = s.minEvalLen * 3
			a[1] = s.retLen
			conn.send(a)
			return 1
		elif cmdType == 2:
			# 初期化データの受け取り
			n = size / 8
			if n < s.minEvalLen:
				conn.send(np.asarray(0, dtype=np.int32))
				return 0
			s.fxYenData = candle.encodeArray(pkt[ipkt : ipkt + 4 * n].view(dtype=np.float32))
			ipkt += 4 * n
			s.fxMinData = pkt[ipkt : ipkt + 4 * n].view(dtype=np.int32)
			if s.fxYenDataTrain is None:
				s.fxYenDataTrain = s.fxYenData
			conn.send(np.asarray(1, dtype=np.int32))
			return 1
		elif cmdType == 3:
			# チャート更新時の逐次データ受け取り
			count = size // (2 * 4)
			if count == 0:
				conn.send(np.asarray(0, dtype=np.int32))
				return 0
			yenData = pkt[ipkt : ipkt + 4 * count].view(dtype=np.float32)
			ipkt += 4 * count
			minData = pkt[ipkt : ipkt + 4 * count].view(dtype=np.int32)

			m = s.fxMinData[-1]
			index = int(np.searchsorted(minData, int(m)))

			# 追加開始インデックスと追加データ数計算、実際に追加されるのは１つ後のデータからとなる
			index += 1
			n = count - index
			if n < 0:
				n = 0

			# 現在保持しているデータ最後尾に対応するデータを上書き
			s.fxYenData[-1] = candle.encode(yenData[index - 1])
			print("set: ", yenData[index - 1])

			# 追加するデータがあるなら追加する
			if n != 0:
				maxDataCount = 60 * 24
				yens = s.fxYenData
				mins = s.fxMinData
				if maxDataCount < yens.shape[0] + n:
					n = yens.shape[0] + n - maxDataCount
					yens = yens[n:]
					mins = mins[n:]
				appendYens = yenData[index : count]
				s.fxYenData = np.append(yens, candle.encodeArray(appendYens))
				s.fxMinData = np.append(mins, minData[index : count])
				print("append: ", appendYens)

				if s.fxYenDataTrain is None:
					s.fxYenDataTrain = s.fxYenData

			conn.send(np.asarray(1, dtype=np.int32))
			return 1
		elif cmdType == 4:
			# 現在受け取ってるドル円データで未来を予測する
			data = s.prediction()
			conn.send(data)
			return 1
		elif cmdType == 5:
			# 予測値の合成係数の取得
			if size < 8:
				conn.send(np.asarray(0, dtype=np.int32))
				return 0
			k = float(pkt[ipkt : ipkt + 8].view(dtype=np.float64))
			s.initAveYenKs(k)
			print(k)
			conn.send(np.asarray(1, dtype=np.int32))
			return 1

		return 0
