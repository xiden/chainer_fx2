import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from numba import jit
import mk_clas
import mk_claslstm


@jit("f4[:,:,:](i8, f4[:,:], i4[:])", nopython=True)
def buildMiniBatchCloseHighLow(inCount, trainDataset, batchIndices):
	"""
	学習データセットの終値、高値、低値を使い指定位置から全ミニバッチデータを作成する.

	Args:
		inCount: ニューラルネットの入力次元数.
		trainDataset: 学習データセット.
		batchIndices: 各バッチの学習データセット開始インデックス番号.

	Returns:
		全バッチ分の学習データセット.
	"""
	batchSize = batchIndices.shape[0]
	x = np.empty(shape=(3, batchSize, inCount), dtype=np.float32)
	for i, p in enumerate(batchIndices):
		pe = p + inCount
		x[0,i,:] = trainDataset[3,p:pe]
		x[1,i,:] = trainDataset[1,p:pe]
		x[2,i,:] = trainDataset[2,p:pe]
	return x

def buildRnnEvalDataCloseHighLow(inCount, trainDataset, startIndex, rnnLen, rnnStep):
	"""
	RNN評価用のデータセットを終値、高値、低値で作成する.

	Args:
		inCount: ニューラルネットの入力次元数.
		trainDataset: 学習データセット.
		startIndex: 学習データセット内RNN開始インデックス番号.
		rnnLen: RNN長.
		rnnStep: RNN一回の移動量.

	Returns:
		１バッチRNN全長分のデータ学習データ.
	"""
	n = inCount + (rnnLen - 1) * rnnStep
	end = startIndex + n
	return [
		trainDataset[3, startIndex : end],
		trainDataset[1, startIndex : end],
		trainDataset[2, startIndex : end]]


class OpenHighLowN6N5(chainer.Chain):
	"""
	絞らず混ぜていくスタイル
	"""
	def __init__(m):
		pass

	def create(m, inCount, unitCount, outCount, gpu, train=True):
		uc1 = unitCount
		uc2 = unitCount
		uc3 = unitCount
		uc4 = unitCount
		uc5 = unitCount
		super().__init__(
			no01=L.Linear(inCount, uc1),
			nh01=L.Linear(inCount, uc1),
			nl01=L.Linear(inCount, uc1),

			no02=L.Linear(uc1, uc2),
			nh02=L.Linear(uc1, uc2),
			nl02=L.Linear(uc1, uc2),

			no03=L.Linear(uc2, uc3),
			nh03=L.Linear(uc2, uc3),
			nl03=L.Linear(uc2, uc3),

			no04=L.Linear(uc3, uc4),
			nh04=L.Linear(uc3, uc4),
			nl04=L.Linear(uc3, uc4),

			no05=L.Linear(uc4, uc5),
			nh05=L.Linear(uc4, uc5),
			nl05=L.Linear(uc4, uc5),

			no06=L.Linear(uc5, uc5),
			nh06=L.Linear(uc5, uc5),
			nl06=L.Linear(uc5, uc5),

			nx07=L.Linear(uc5, uc4),
			nx08=L.Linear(uc4, uc3),
			nx09=L.Linear(uc3, uc2),
			nx10=L.Linear(uc2, uc1),
			nx11=L.Linear(uc1, outCount)
		)
		m.inCount = inCount
		m.train = train

	#@jit(nopython=True)
	def reset_state(m):
		pass

	#@jit(nopython=True)
	def __call__(m, x, volatile):
		# 開始値、高値、低値それぞれを通す
		h = F.relu(m.no01(chainer.Variable(x[0], volatile=volatile)))
		h = F.relu(m.no02(h))
		h = F.relu(m.no03(h))
		h = F.relu(m.no04(h))
		h = F.relu(m.no05(h))
		ho = F.relu(m.no06(h))

		h = F.relu(m.nh01(chainer.Variable(x[1], volatile=volatile)))
		h = F.relu(m.nh02(h))
		h = F.relu(m.nh03(h))
		h = F.relu(m.nh04(h))
		h = F.relu(m.nh05(h))
		hh = F.relu(m.nh06(h))

		h = F.relu(m.nl01(chainer.Variable(x[2], volatile=volatile)))
		h = F.relu(m.nl02(h))
		h = F.relu(m.nl03(h))
		h = F.relu(m.nl04(h))
		h = F.relu(m.nl05(h))
		hl = F.relu(m.nl06(h))

		# 混ぜる
		h = hh * 0.25 + hl * 0.25 + ho * 0.5

		h = F.relu(m.nx07(h))
		h = F.relu(m.nx08(h))
		h = F.relu(m.nx09(h))
		h = F.relu(m.nx10(h))
		h = m.nx11(h)

		return h

	def buildMiniBatchData(m, dataset, batchIndices):
		"""学習データセットの指定位置から全ミニバッチデータを作成する"""
		batchSize = batchIndices.shape[0]
		inCount = m.inCount
		x = np.empty(shape=(3, batchSize, inCount), dtype=np.float32)
		for i, p in enumerate(batchIndices):
			pe = p + inCount
			o = dataset[0, p : pe]
			a = (o.max() + o.min()) * 0.5
			x[0,i,:] = o - a
			x[1,i,:] = dataset[1, p : pe] - a
			x[2,i,:] = dataset[2, p : pe] - a
		return x

	def getModelKind(m):
		return "clas"


class AllSqwzBnN6N5(chainer.Chain):
	"""
	開始値、高値、低値、終値全てを使い絞って混ぜて広げていくスタイル
	"""
	def __init__(m):
		pass

	def create(m, inCount, unitCount, outCount, gpu, train=True):
		uc1 = 8 * unitCount // 10
		uc2 = 6 * unitCount // 10
		uc3 = 4 * unitCount // 10
		uc4 = 2 * unitCount // 10
		uc5 = 1 * unitCount // 10
		super().__init__(
			bno=L.BatchNormalization(inCount),
			bnh=L.BatchNormalization(inCount),
			bnl=L.BatchNormalization(inCount),
			bnc=L.BatchNormalization(inCount),

			no01=L.Linear(inCount, uc1),
			nh01=L.Linear(inCount, uc1),
			nl01=L.Linear(inCount, uc1),
			nc01=L.Linear(inCount, uc1),

			no02=L.Linear(uc1, uc2),
			nh02=L.Linear(uc1, uc2),
			nl02=L.Linear(uc1, uc2),
			nc02=L.Linear(uc1, uc2),

			no03=L.Linear(uc2, uc3),
			nh03=L.Linear(uc2, uc3),
			nl03=L.Linear(uc2, uc3),
			nc03=L.Linear(uc2, uc3),

			no04=L.Linear(uc3, uc4),
			nh04=L.Linear(uc3, uc4),
			nl04=L.Linear(uc3, uc4),
			nc04=L.Linear(uc3, uc4),

			no05=L.Linear(uc4, uc5),
			nh05=L.Linear(uc4, uc5),
			nl05=L.Linear(uc4, uc5),
			nc05=L.Linear(uc4, uc5),

			no06=L.Linear(uc5, uc5),
			nh06=L.Linear(uc5, uc5),
			nl06=L.Linear(uc5, uc5),
			nc06=L.Linear(uc5, uc5),

			nx07=L.Linear(uc5, uc4),
			nx08=L.Linear(uc4, uc3),
			nx09=L.Linear(uc3, uc2),
			nx10=L.Linear(uc2, uc1),
			nx11=L.Linear(uc1, outCount)
		)
		m.inCount = inCount
		m.train = train

	#@jit(nopython=True)
	def reset_state(m):
		pass

	#@jit(nopython=True)
	def __call__(m, x, volatile):
		test = not m.train

		# 開始値、高値、低値、終値それぞれを絞る
		h = F.relu(m.bno(chainer.Variable(x[0], volatile=volatile), test=test))
		h = F.relu(m.no01(h))
		h = F.relu(m.no02(h))
		h = F.relu(m.no03(h))
		h = F.relu(m.no04(h))
		h = F.relu(m.no05(h))
		ho = F.relu(m.no06(h))

		h = F.relu(m.bnh(chainer.Variable(x[1], volatile=volatile), test=test))
		h = F.relu(m.nh01(h))
		h = F.relu(m.nh02(h))
		h = F.relu(m.nh03(h))
		h = F.relu(m.nh04(h))
		h = F.relu(m.nh05(h))
		hh = F.relu(m.nh06(h))

		h = F.relu(m.bnl(chainer.Variable(x[2], volatile=volatile), test=test))
		h = F.relu(m.nl01(h))
		h = F.relu(m.nl02(h))
		h = F.relu(m.nl03(h))
		h = F.relu(m.nl04(h))
		h = F.relu(m.nl05(h))
		hl = F.relu(m.nl06(h))

		h = F.relu(m.bnc(chainer.Variable(x[3], volatile=volatile), test=test))
		h = F.relu(m.nc01(h))
		h = F.relu(m.nc02(h))
		h = F.relu(m.nc03(h))
		h = F.relu(m.nc04(h))
		h = F.relu(m.nc05(h))
		hc = F.relu(m.nc06(h))

		# 混ぜる
		h = ho + hh + hl + hc

		# 広げていく
		h = F.relu(m.nx07(h))
		h = F.relu(m.nx08(h))
		h = F.relu(m.nx09(h))
		h = F.relu(m.nx10(h))
		h = m.nx11(h)

		return h

	def buildMiniBatchData(m, dataset, batchIndices):
		"""学習データセットの指定位置から全ミニバッチデータを作成する"""
		batchSize = batchIndices.shape[0]
		inCount = m.inCount
		x = np.empty(shape=(4, batchSize, inCount), dtype=np.float32)
		for i, p in enumerate(batchIndices):
			pe = p + inCount
			x[0,i,:] = dataset[0, p : pe]
			x[1,i,:] = dataset[1, p : pe]
			x[2,i,:] = dataset[2, p : pe]
			x[3,i,:] = dataset[3, p : pe]
		return x

	def getModelKind(m):
		return "clas"


class AllSqwzN6N5(chainer.Chain):
	"""
	開始値、高値、低値、終値全てを使い絞って混ぜて広げていくスタイル
	"""
	def __init__(m):
		pass

	def create(m, inCount, unitCount, outCount, gpu, train=True):
		uc1 = 8 * unitCount // 10
		uc2 = 6 * unitCount // 10
		uc3 = 4 * unitCount // 10
		uc4 = 2 * unitCount // 10
		uc5 = 1 * unitCount // 10
		super().__init__(
			no01=L.Linear(inCount, uc1),
			nh01=L.Linear(inCount, uc1),
			nl01=L.Linear(inCount, uc1),
			nc01=L.Linear(inCount, uc1),

			no02=L.Linear(uc1, uc2),
			nh02=L.Linear(uc1, uc2),
			nl02=L.Linear(uc1, uc2),
			nc02=L.Linear(uc1, uc2),

			no03=L.Linear(uc2, uc3),
			nh03=L.Linear(uc2, uc3),
			nl03=L.Linear(uc2, uc3),
			nc03=L.Linear(uc2, uc3),

			no04=L.Linear(uc3, uc4),
			nh04=L.Linear(uc3, uc4),
			nl04=L.Linear(uc3, uc4),
			nc04=L.Linear(uc3, uc4),

			no05=L.Linear(uc4, uc5),
			nh05=L.Linear(uc4, uc5),
			nl05=L.Linear(uc4, uc5),
			nc05=L.Linear(uc4, uc5),

			no06=L.Linear(uc5, uc5),
			nh06=L.Linear(uc5, uc5),
			nl06=L.Linear(uc5, uc5),
			nc06=L.Linear(uc5, uc5),

			nx07=L.Linear(uc5, uc4),
			nx08=L.Linear(uc4, uc3),
			nx09=L.Linear(uc3, uc2),
			nx10=L.Linear(uc2, uc1),
			nx11=L.Linear(uc1, outCount)
		)
		m.inCount = inCount
		m.train = train

	#@jit(nopython=True)
	def reset_state(m):
		pass

	#@jit(nopython=True)
	def __call__(m, x, volatile):
		# 高値の最大値と低値の最小値の中間が０になるようシフトする
		a = (x[1].max(1, keepdims=True) + x[2].min(1, keepdims=True)) * 0.5
		x[0] -= a
		x[1] -= a
		x[2] -= a
		x[3] -= a

		# 開始値、高値、低値、終値それぞれを絞る
		h = F.relu(m.no01(chainer.Variable(x[0], volatile=volatile)))
		h = F.relu(m.no02(h))
		h = F.relu(m.no03(h))
		h = F.relu(m.no04(h))
		h = F.relu(m.no05(h))
		ho = F.relu(m.no06(h))

		h = F.relu(m.nh01(chainer.Variable(x[1], volatile=volatile)))
		h = F.relu(m.nh02(h))
		h = F.relu(m.nh03(h))
		h = F.relu(m.nh04(h))
		h = F.relu(m.nh05(h))
		hh = F.relu(m.nh06(h))

		h = F.relu(m.nl01(chainer.Variable(x[2], volatile=volatile)))
		h = F.relu(m.nl02(h))
		h = F.relu(m.nl03(h))
		h = F.relu(m.nl04(h))
		h = F.relu(m.nl05(h))
		hl = F.relu(m.nl06(h))

		h = F.relu(m.nc01(chainer.Variable(x[3], volatile=volatile)))
		h = F.relu(m.nc02(h))
		h = F.relu(m.nc03(h))
		h = F.relu(m.nc04(h))
		h = F.relu(m.nc05(h))
		hc = F.relu(m.nc06(h))

		# 混ぜる
		h = ho + hh + hl + hc

		# 広げていく
		h = F.relu(m.nx07(h))
		h = F.relu(m.nx08(h))
		h = F.relu(m.nx09(h))
		h = F.relu(m.nx10(h))
		h = m.nx11(h)

		return h

	def buildMiniBatchData(m, dataset, batchIndices):
		"""学習データセットの指定位置から全ミニバッチデータを作成する"""
		batchSize = batchIndices.shape[0]
		inCount = m.inCount
		x = np.empty(shape=(4, batchSize, inCount), dtype=np.float32)
		for i, p in enumerate(batchIndices):
			pe = p + inCount
			x[0,i,:] = dataset[0,p:pe]
			x[1,i,:] = dataset[1,p:pe]
			x[2,i,:] = dataset[2,p:pe]
			x[3,i,:] = dataset[3,p:pe]
		return x

	def getModelKind(m):
		return "clas"


class CloseN2L2X1(chainer.Chain):
	"""
	終値を加工してLSTM後に仕上げ.
	"""
	def __init__(m):
		pass

	def create(m, inCount, unitCount, outCount, gpu, train=True):
		uc1 = unitCount
		uc2 = 1000
		super().__init__(
			nc01=L.Linear(inCount, uc1),
			nc02=L.Linear(uc1, uc2),

			l01=L.LSTM(uc2, uc2),
			l02=L.LSTM(uc2, uc2),

			nx01=L.Linear(uc2, outCount)
		)
		m.inCount = inCount
		m.train = train

	#@jit(nopython=True)
	def reset_state(m):
		m.l01.reset_state()
		m.l02.reset_state()

	#@jit(nopython=True)
	def __call__(m, x, volatile):
		# 各バッチの高値最大値と低値最小値の中間が０になるようシフトする
		a = (x[1].max(1, keepdims=True) + x[2].min(1, keepdims=True)) * 0.5
		x[0] -= a

		# 終値を加工
		h = F.tanh(m.nc01(chainer.Variable(x[0], volatile=volatile)))
		h = F.tanh(m.nc02(h))

		# LSTMを通す
		tr = m.train
		do = mk_claslstm.dropoutRatio
		h = F.dropout(m.l01(h), ratio=do, train=tr)
		h = F.dropout(m.l02(h), ratio=do, train=tr)

		# 仕上げ
		h = F.tanh(m.nx01(h))

		return h

	def buildMiniBatchData(m, dataset, batchIndices):
		"""学習データセットの指定位置から全ミニバッチデータを作成する"""
		return buildMiniBatchCloseHighLow(m.inCount, dataset, batchIndices)

	def buildRnnEvalData(m, inCount, trainDataset, startIndex, rnnLen, rnnStep):
		"""RNN評価用のデータセットを終値、高値、低値で作成する."""
		return buildRnnEvalDataCloseHighLow(inCount, trainDataset, startIndex, rnnLen, rnnStep)

	def getModelKind(m):
		return "claslstm"


class CloseHighLowN6N10Ver2(chainer.Chain):
	"""
	高値、低値、終値を使い絞って混ぜて広げていくスタイル
	"""
	def __init__(m):
		pass

	def create(m, inCount, unitCount, outCount, gpu, train=True):
		uc1 = 8 * unitCount // 10
		uc2 = 6 * unitCount // 10
		uc3 = 4 * unitCount // 10
		uc4 = 3 * unitCount // 10
		uc5 = 2 * unitCount // 10
		super().__init__(
			nh01=L.Linear(inCount, uc1),
			nl01=L.Linear(inCount, uc1),
			nc01=L.Linear(inCount, uc1),

			nh02=L.Linear(uc1, uc2),
			nl02=L.Linear(uc1, uc2),
			nc02=L.Linear(uc1, uc2),

			nh03=L.Linear(uc2, uc3),
			nl03=L.Linear(uc2, uc3),
			nc03=L.Linear(uc2, uc3),

			nh04=L.Linear(uc3, uc4),
			nl04=L.Linear(uc3, uc4),
			nc04=L.Linear(uc3, uc4),

			nh05=L.Linear(uc4, uc5),
			nl05=L.Linear(uc4, uc5),
			nc05=L.Linear(uc4, uc5),

			nh06=L.Linear(uc5, uc5),
			nl06=L.Linear(uc5, uc5),
			nc06=L.Linear(uc5, uc5),

			nx07=L.Linear(uc5, uc5),
			nx08=L.Linear(uc5, uc5),
			nx09=L.Linear(uc5, uc5),
			nx10=L.Linear(uc5, uc5),
			nx11=L.Linear(uc5, uc5),
			nx12=L.Linear(uc5, uc4),
			nx13=L.Linear(uc4, uc3),
			nx14=L.Linear(uc3, uc2),
			nx15=L.Linear(uc2, uc1),
			nx16=L.Linear(uc1, outCount)
		)
		m.inCount = inCount
		m.train = train

	#@jit(nopython=True)
	def reset_state(m):
		pass

	#@jit(nopython=True)
	def __call__(m, x, volatile):
		# 各バッチの高値最大値と低値最小値の中間が０になるようシフトする
		a = (x[1].max(1, keepdims=True) + x[2].min(1, keepdims=True)) * 0.5
		x[0] -= a
		x[1] -= a
		x[2] -= a

		# 終値、高値、低値それぞれを絞る
		h = F.relu(m.nc01(chainer.Variable(x[0], volatile=volatile)))
		h = F.relu(m.nc02(h))
		h = F.relu(m.nc03(h))
		h = F.relu(m.nc04(h))
		h = F.relu(m.nc05(h))
		hc = F.relu(m.nc06(h))

		h = F.relu(m.nh01(chainer.Variable(x[1], volatile=volatile)))
		h = F.relu(m.nh02(h))
		h = F.relu(m.nh03(h))
		h = F.relu(m.nh04(h))
		h = F.relu(m.nh05(h))
		hh = F.relu(m.nh06(h))

		h = F.relu(m.nl01(chainer.Variable(x[2], volatile=volatile)))
		h = F.relu(m.nl02(h))
		h = F.relu(m.nl03(h))
		h = F.relu(m.nl04(h))
		h = F.relu(m.nl05(h))
		hl = F.relu(m.nl06(h))

		# 混ぜる
		h = hc + hh + hl

		# いくつかレイヤを通す
		h = F.relu(m.nx07(h))
		h = F.relu(m.nx08(h))
		h = F.relu(m.nx09(h))
		h = F.relu(m.nx10(h))
		h = F.relu(m.nx11(h))

		# 広げていく
		h = F.relu(m.nx12(h))
		h = F.relu(m.nx13(h))
		h = F.relu(m.nx14(h))
		h = F.relu(m.nx15(h))
		h = m.nx16(h)

		return h

	def buildMiniBatchData(m, dataset, batchIndices):
		"""学習データセットの指定位置から全ミニバッチデータを作成する"""
		return buildMiniBatchCloseHighLow(m.inCount, dataset, batchIndices)

	def getModelKind(m):
		return "clas"


class CloseHighLowN6N10Ver3(chainer.Chain):
	"""
	高値、低値、終値を使い絞って混ぜて広げていくスタイルReLuの代わりにTanhを使ってみた
	"""
	def __init__(m):
		pass

	def create(m, inCount, unitCount, outCount, gpu, train=True):
		uc1 = 8 * unitCount // 10
		uc2 = 6 * unitCount // 10
		uc3 = 4 * unitCount // 10
		uc4 = 3 * unitCount // 10
		uc5 = 2 * unitCount // 10
		super().__init__(
			nh01=L.Linear(inCount, uc1),
			nl01=L.Linear(inCount, uc1),
			nc01=L.Linear(inCount, uc1),

			nh02=L.Linear(uc1, uc2),
			nl02=L.Linear(uc1, uc2),
			nc02=L.Linear(uc1, uc2),

			nh03=L.Linear(uc2, uc3),
			nl03=L.Linear(uc2, uc3),
			nc03=L.Linear(uc2, uc3),

			nh04=L.Linear(uc3, uc4),
			nl04=L.Linear(uc3, uc4),
			nc04=L.Linear(uc3, uc4),

			nh05=L.Linear(uc4, uc5),
			nl05=L.Linear(uc4, uc5),
			nc05=L.Linear(uc4, uc5),

			nh06=L.Linear(uc5, uc5),
			nl06=L.Linear(uc5, uc5),
			nc06=L.Linear(uc5, uc5),

			nx07=L.Linear(uc5, uc5),
			nx08=L.Linear(uc5, uc5),
			nx09=L.Linear(uc5, uc5),
			nx10=L.Linear(uc5, uc5),
			nx11=L.Linear(uc5, uc5),
			nx12=L.Linear(uc5, uc4),
			nx13=L.Linear(uc4, uc3),
			nx14=L.Linear(uc3, uc2),
			nx15=L.Linear(uc2, uc1),
			nx16=L.Linear(uc1, outCount)
		)
		m.inCount = inCount
		m.train = train

	#@jit(nopython=True)
	def reset_state(m):
		pass

	#@jit(nopython=True)
	def __call__(m, x, volatile):
		# 各バッチの高値最大値と低値最小値の中間が０になるようシフトする
		a = (x[1].max(1, keepdims=True) + x[2].min(1, keepdims=True)) * 0.5
		x[0] -= a
		x[1] -= a
		x[2] -= a

		# 終値、高値、低値それぞれを絞る
		h = F.tanh(m.nc01(chainer.Variable(x[0], volatile=volatile)))
		h = F.tanh(m.nc02(h))
		h = F.tanh(m.nc03(h))
		h = F.tanh(m.nc04(h))
		h = F.tanh(m.nc05(h))
		hc = F.tanh(m.nc06(h))

		h = F.tanh(m.nh01(chainer.Variable(x[1], volatile=volatile)))
		h = F.tanh(m.nh02(h))
		h = F.tanh(m.nh03(h))
		h = F.tanh(m.nh04(h))
		h = F.tanh(m.nh05(h))
		hh = F.tanh(m.nh06(h))

		h = F.tanh(m.nl01(chainer.Variable(x[2], volatile=volatile)))
		h = F.tanh(m.nl02(h))
		h = F.tanh(m.nl03(h))
		h = F.tanh(m.nl04(h))
		h = F.tanh(m.nl05(h))
		hl = F.tanh(m.nl06(h))

		# 混ぜる
		h = hc + hh + hl

		# いくつかレイヤを通す
		h = F.tanh(m.nx07(h))
		h = F.tanh(m.nx08(h))
		h = F.tanh(m.nx09(h))
		h = F.tanh(m.nx10(h))
		h = F.tanh(m.nx11(h))

		# 広げていく
		h = F.tanh(m.nx12(h))
		h = F.tanh(m.nx13(h))
		h = F.tanh(m.nx14(h))
		h = F.tanh(m.nx15(h))
		h = m.nx16(h)

		return h

	def buildMiniBatchData(m, dataset, batchIndices):
		"""学習データセットの指定位置から全ミニバッチデータを作成する"""
		return buildMiniBatchCloseHighLow(m.inCount, dataset, batchIndices)

	def getModelKind(m):
		return "clas"


class CloseHighLowN6N10(chainer.Chain):
	"""
	高値、低値、終値を使い絞って混ぜて広げていくスタイル
	"""
	def __init__(m):
		pass

	def create(m, inCount, unitCount, outCount, gpu, train=True):
		uc1 = 8 * unitCount // 10
		uc2 = 6 * unitCount // 10
		uc3 = 4 * unitCount // 10
		uc4 = 2 * unitCount // 10
		uc5 = 1 * unitCount // 10
		super().__init__(
			nh01=L.Linear(inCount, uc1),
			nl01=L.Linear(inCount, uc1),
			nc01=L.Linear(inCount, uc1),

			nh02=L.Linear(uc1, uc2),
			nl02=L.Linear(uc1, uc2),
			nc02=L.Linear(uc1, uc2),

			nh03=L.Linear(uc2, uc3),
			nl03=L.Linear(uc2, uc3),
			nc03=L.Linear(uc2, uc3),

			nh04=L.Linear(uc3, uc4),
			nl04=L.Linear(uc3, uc4),
			nc04=L.Linear(uc3, uc4),

			nh05=L.Linear(uc4, uc5),
			nl05=L.Linear(uc4, uc5),
			nc05=L.Linear(uc4, uc5),

			nh06=L.Linear(uc5, uc5),
			nl06=L.Linear(uc5, uc5),
			nc06=L.Linear(uc5, uc5),

			nx07=L.Linear(uc5, uc5),
			nx08=L.Linear(uc5, uc5),
			nx09=L.Linear(uc5, uc5),
			nx10=L.Linear(uc5, uc5),
			nx11=L.Linear(uc5, uc5),
			nx12=L.Linear(uc5, uc4),
			nx13=L.Linear(uc4, uc3),
			nx14=L.Linear(uc3, uc2),
			nx15=L.Linear(uc2, uc1),
			nx16=L.Linear(uc1, outCount)
		)
		m.inCount = inCount
		m.train = train

	#@jit(nopython=True)
	def reset_state(m):
		pass

	#@jit(nopython=True)
	def __call__(m, x, volatile):
		# 各バッチの高値最大値と低値最小値の中間が０になるようシフトする
		a = (x[1].max(1, keepdims=True) + x[2].min(1, keepdims=True)) * 0.5
		x[0] -= a
		x[1] -= a
		x[2] -= a

		# 終値、高値、低値それぞれを絞る
		h = F.relu(m.nc01(chainer.Variable(x[0], volatile=volatile)))
		h = F.relu(m.nc02(h))
		h = F.relu(m.nc03(h))
		h = F.relu(m.nc04(h))
		h = F.relu(m.nc05(h))
		hc = F.relu(m.nc06(h))

		h = F.relu(m.nh01(chainer.Variable(x[1], volatile=volatile)))
		h = F.relu(m.nh02(h))
		h = F.relu(m.nh03(h))
		h = F.relu(m.nh04(h))
		h = F.relu(m.nh05(h))
		hh = F.relu(m.nh06(h))

		h = F.relu(m.nl01(chainer.Variable(x[2], volatile=volatile)))
		h = F.relu(m.nl02(h))
		h = F.relu(m.nl03(h))
		h = F.relu(m.nl04(h))
		h = F.relu(m.nl05(h))
		hl = F.relu(m.nl06(h))

		# 混ぜる
		h = hc + hh + hl

		# いくつかレイヤを通す
		h = F.relu(m.nx07(h))
		h = F.relu(m.nx08(h))
		h = F.relu(m.nx09(h))
		h = F.relu(m.nx10(h))
		h = F.relu(m.nx11(h))

		# 広げていく
		h = F.relu(m.nx12(h))
		h = F.relu(m.nx13(h))
		h = F.relu(m.nx14(h))
		h = F.relu(m.nx15(h))
		h = m.nx16(h)

		return h

	def buildMiniBatchData(m, dataset, batchIndices):
		"""学習データセットの指定位置から全ミニバッチデータを作成する"""
		return buildMiniBatchCloseHighLow(m.inCount, dataset, batchIndices)

	def getModelKind(m):
		return "clas"


class ZigZagAllSqwzN6N10(chainer.Chain):
	"""
	開始値、高値、低値、終値全てを使い絞って混ぜて広げていくスタイル
	"""
	def __init__(m):
		pass

	def create(m, inCount, unitCount, outCount, gpu, train=True):
		uc1 = 8 * unitCount // 10
		uc2 = 6 * unitCount // 10
		uc3 = 4 * unitCount // 10
		uc4 = 2 * unitCount // 10
		uc5 = 1 * unitCount // 10
		super().__init__(
			no01=L.Linear(inCount, uc1),
			nh01=L.Linear(inCount, uc1),
			nl01=L.Linear(inCount, uc1),
			nc01=L.Linear(inCount, uc1),

			no02=L.Linear(uc1, uc2),
			nh02=L.Linear(uc1, uc2),
			nl02=L.Linear(uc1, uc2),
			nc02=L.Linear(uc1, uc2),

			no03=L.Linear(uc2, uc3),
			nh03=L.Linear(uc2, uc3),
			nl03=L.Linear(uc2, uc3),
			nc03=L.Linear(uc2, uc3),

			no04=L.Linear(uc3, uc4),
			nh04=L.Linear(uc3, uc4),
			nl04=L.Linear(uc3, uc4),
			nc04=L.Linear(uc3, uc4),

			no05=L.Linear(uc4, uc5),
			nh05=L.Linear(uc4, uc5),
			nl05=L.Linear(uc4, uc5),
			nc05=L.Linear(uc4, uc5),

			no06=L.Linear(uc5, uc5),
			nh06=L.Linear(uc5, uc5),
			nl06=L.Linear(uc5, uc5),
			nc06=L.Linear(uc5, uc5),

			nx07=L.Linear(uc5, uc5),
			nx08=L.Linear(uc5, uc5),
			nx09=L.Linear(uc5, uc5),
			nx10=L.Linear(uc5, uc5),
			nx11=L.Linear(uc5, uc5),
			nx12=L.Linear(uc5, uc4),
			nx13=L.Linear(uc4, uc3),
			nx14=L.Linear(uc3, uc2),
			nx15=L.Linear(uc2, uc1),
			nx16=L.Linear(uc1, outCount)
		)
		m.inCount = inCount
		m.train = train

	#@jit(nopython=True)
	def reset_state(m):
		pass

	#@jit(nopython=True)
	def __call__(m, x, volatile):
		# 高値の最大値と低値の最小値の中間が０になるようシフトする
		a = (x[1].max(1, keepdims=True) + x[2].min(1, keepdims=True)) * 0.5
		x[0] -= a
		x[1] -= a
		x[2] -= a
		x[3] -= a

		# 開始値、高値、低値、終値それぞれを絞る
		h = F.relu(m.no01(chainer.Variable(x[0], volatile=volatile)))
		h = F.relu(m.no02(h))
		h = F.relu(m.no03(h))
		h = F.relu(m.no04(h))
		h = F.relu(m.no05(h))
		ho = F.relu(m.no06(h))

		h = F.relu(m.nh01(chainer.Variable(x[1], volatile=volatile)))
		h = F.relu(m.nh02(h))
		h = F.relu(m.nh03(h))
		h = F.relu(m.nh04(h))
		h = F.relu(m.nh05(h))
		hh = F.relu(m.nh06(h))

		h = F.relu(m.nl01(chainer.Variable(x[2], volatile=volatile)))
		h = F.relu(m.nl02(h))
		h = F.relu(m.nl03(h))
		h = F.relu(m.nl04(h))
		h = F.relu(m.nl05(h))
		hl = F.relu(m.nl06(h))

		h = F.relu(m.nc01(chainer.Variable(x[3], volatile=volatile)))
		h = F.relu(m.nc02(h))
		h = F.relu(m.nc03(h))
		h = F.relu(m.nc04(h))
		h = F.relu(m.nc05(h))
		hc = F.relu(m.nc06(h))

		# 混ぜる
		h = ho + hh + hl + hc

		# いくつかレイヤーを通す
		h = F.relu(m.nx07(h))
		h = F.relu(m.nx08(h))
		h = F.relu(m.nx09(h))
		h = F.relu(m.nx10(h))
		h = F.relu(m.nx11(h))

		# 広げていく
		h = F.relu(m.nx12(h))
		h = F.relu(m.nx13(h))
		h = F.relu(m.nx14(h))
		h = F.relu(m.nx15(h))
		h = m.nx16(h)

		return h

	def buildMiniBatchData(m, dataset, batchIndices):
		"""学習データセットの指定位置から全ミニバッチデータを作成する"""
		batchSize = batchIndices.shape[0]
		inCount = m.inCount
		x = np.empty(shape=(4, batchSize, inCount), dtype=np.float32)
		for i, p in enumerate(batchIndices):
			pe = p + inCount
			x[0,i,:] = dataset[0,p:pe]
			x[1,i,:] = dataset[1,p:pe]
			x[2,i,:] = dataset[2,p:pe]
			x[3,i,:] = dataset[3,p:pe]
		return x

	def getModelKind(m):
		return "zigzag"


class ZigZagAllZSqwzN6N10(chainer.Chain):
	"""
	開始値、高値、低値、終値+ジグザグ全てを使い絞って混ぜて広げていくスタイル
	"""
	def __init__(m):
		pass

	def create(m, inCount, unitCount, outCount, gpu, train=True):
		uc1 = 8 * unitCount // 10
		uc2 = 6 * unitCount // 10
		uc3 = 4 * unitCount // 10
		uc4 = 2 * unitCount // 10
		uc5 = 1 * unitCount // 10
		super().__init__(
			no01=L.Linear(inCount, uc1),
			nh01=L.Linear(inCount, uc1),
			nl01=L.Linear(inCount, uc1),
			nc01=L.Linear(inCount, uc1),
			nz01=L.Linear(inCount, uc1),

			no02=L.Linear(uc1, uc2),
			nh02=L.Linear(uc1, uc2),
			nl02=L.Linear(uc1, uc2),
			nc02=L.Linear(uc1, uc2),
			nz02=L.Linear(uc1, uc2),

			no03=L.Linear(uc2, uc3),
			nh03=L.Linear(uc2, uc3),
			nl03=L.Linear(uc2, uc3),
			nc03=L.Linear(uc2, uc3),
			nz03=L.Linear(uc2, uc3),

			no04=L.Linear(uc3, uc4),
			nh04=L.Linear(uc3, uc4),
			nl04=L.Linear(uc3, uc4),
			nc04=L.Linear(uc3, uc4),
			nz04=L.Linear(uc3, uc4),

			no05=L.Linear(uc4, uc5),
			nh05=L.Linear(uc4, uc5),
			nl05=L.Linear(uc4, uc5),
			nc05=L.Linear(uc4, uc5),
			nz05=L.Linear(uc4, uc5),

			no06=L.Linear(uc5, uc5),
			nh06=L.Linear(uc5, uc5),
			nl06=L.Linear(uc5, uc5),
			nc06=L.Linear(uc5, uc5),
			nz06=L.Linear(uc5, uc5),

			nx07=L.Linear(uc5, uc5),
			nx08=L.Linear(uc5, uc5),
			nx09=L.Linear(uc5, uc5),
			nx10=L.Linear(uc5, uc5),
			nx11=L.Linear(uc5, uc5),
			nx12=L.Linear(uc5, uc4),
			nx13=L.Linear(uc4, uc3),
			nx14=L.Linear(uc3, uc2),
			nx15=L.Linear(uc2, uc1),
			nx16=L.Linear(uc1, outCount)
		)
		m.inCount = inCount
		m.train = train

	#@jit(nopython=True)
	def reset_state(m):
		pass

	#@jit(nopython=True)
	def __call__(m, x, volatile):
		# 高値の最大値と低値の最小値の中間が０になるようシフトする
		a = (x[1].max(1, keepdims=True) + x[2].min(1, keepdims=True)) * 0.5
		x[0] -= a
		x[1] -= a
		x[2] -= a
		x[3] -= a

		# 開始値、高値、低値、終値、ジグザグそれぞれを絞る
		h = F.relu(m.no01(chainer.Variable(x[0], volatile=volatile)))
		h = F.relu(m.no02(h))
		h = F.relu(m.no03(h))
		h = F.relu(m.no04(h))
		h = F.relu(m.no05(h))
		ho = F.relu(m.no06(h))

		h = F.relu(m.nh01(chainer.Variable(x[1], volatile=volatile)))
		h = F.relu(m.nh02(h))
		h = F.relu(m.nh03(h))
		h = F.relu(m.nh04(h))
		h = F.relu(m.nh05(h))
		hh = F.relu(m.nh06(h))

		h = F.relu(m.nl01(chainer.Variable(x[2], volatile=volatile)))
		h = F.relu(m.nl02(h))
		h = F.relu(m.nl03(h))
		h = F.relu(m.nl04(h))
		h = F.relu(m.nl05(h))
		hl = F.relu(m.nl06(h))

		h = F.relu(m.nc01(chainer.Variable(x[3], volatile=volatile)))
		h = F.relu(m.nc02(h))
		h = F.relu(m.nc03(h))
		h = F.relu(m.nc04(h))
		h = F.relu(m.nc05(h))
		hc = F.relu(m.nc06(h))

		h = F.relu(m.nz01(chainer.Variable(x[4], volatile=volatile)))
		h = F.relu(m.nz02(h))
		h = F.relu(m.nz03(h))
		h = F.relu(m.nz04(h))
		h = F.relu(m.nz05(h))
		hz = F.relu(m.nz06(h))

		# 混ぜる
		h = ho + hh + hl + hc + hz

		# いくつかレイヤーを通す
		h = F.relu(m.nx07(h))
		h = F.relu(m.nx08(h))
		h = F.relu(m.nx09(h))
		h = F.relu(m.nx10(h))
		h = F.relu(m.nx11(h))

		# 広げていく
		h = F.relu(m.nx12(h))
		h = F.relu(m.nx13(h))
		h = F.relu(m.nx14(h))
		h = F.relu(m.nx15(h))
		h = m.nx16(h)

		return h

	def buildMiniBatchData(m, dataset, batchIndices):
		"""学習データセットの指定位置から全ミニバッチデータを作成する"""
		batchSize = batchIndices.shape[0]
		inCount = m.inCount
		x = np.empty(shape=(5, batchSize, inCount), dtype=np.float32)
		for i, p in enumerate(batchIndices):
			pe = p + inCount
			x[0,i,:] = dataset[0,p:pe]
			x[1,i,:] = dataset[1,p:pe]
			x[2,i,:] = dataset[2,p:pe]
			x[3,i,:] = dataset[3,p:pe]
			x[4,i,:] = dataset[4,p:pe]
		return x

	def getModelKind(m):
		return "zigzag"


class ZigZagAllSqwzN6N5(chainer.Chain):
	"""
	開始値、高値、低値、終値全てを使い絞って混ぜて広げていくスタイル
	"""
	def __init__(m):
		pass

	def create(m, inCount, unitCount, outCount, gpu, train=True):
		uc1 = 8 * unitCount // 10
		uc2 = 6 * unitCount // 10
		uc3 = 4 * unitCount // 10
		uc4 = 2 * unitCount // 10
		uc5 = 1 * unitCount // 10
		super().__init__(
			no01=L.Linear(inCount, uc1),
			nh01=L.Linear(inCount, uc1),
			nl01=L.Linear(inCount, uc1),
			nc01=L.Linear(inCount, uc1),

			no02=L.Linear(uc1, uc2),
			nh02=L.Linear(uc1, uc2),
			nl02=L.Linear(uc1, uc2),
			nc02=L.Linear(uc1, uc2),

			no03=L.Linear(uc2, uc3),
			nh03=L.Linear(uc2, uc3),
			nl03=L.Linear(uc2, uc3),
			nc03=L.Linear(uc2, uc3),

			no04=L.Linear(uc3, uc4),
			nh04=L.Linear(uc3, uc4),
			nl04=L.Linear(uc3, uc4),
			nc04=L.Linear(uc3, uc4),

			no05=L.Linear(uc4, uc5),
			nh05=L.Linear(uc4, uc5),
			nl05=L.Linear(uc4, uc5),
			nc05=L.Linear(uc4, uc5),

			no06=L.Linear(uc5, uc5),
			nh06=L.Linear(uc5, uc5),
			nl06=L.Linear(uc5, uc5),
			nc06=L.Linear(uc5, uc5),

			nx07=L.Linear(uc5, uc4),
			nx08=L.Linear(uc4, uc3),
			nx09=L.Linear(uc3, uc2),
			nx10=L.Linear(uc2, uc1),
			nx11=L.Linear(uc1, outCount)
		)
		m.inCount = inCount
		m.train = train

	#@jit(nopython=True)
	def reset_state(m):
		pass

	#@jit(nopython=True)
	def __call__(m, x, volatile):
		# 高値の最大値と低値の最小値の中間が０になるようシフトする
		a = (x[1].max(1, keepdims=True) + x[2].min(1, keepdims=True)) * 0.5
		x[0] -= a
		x[1] -= a
		x[2] -= a
		x[3] -= a

		# 開始値、高値、低値、終値それぞれを絞る
		h = F.relu(m.no01(chainer.Variable(x[0], volatile=volatile)))
		h = F.relu(m.no02(h))
		h = F.relu(m.no03(h))
		h = F.relu(m.no04(h))
		h = F.relu(m.no05(h))
		ho = F.relu(m.no06(h))

		h = F.relu(m.nh01(chainer.Variable(x[1], volatile=volatile)))
		h = F.relu(m.nh02(h))
		h = F.relu(m.nh03(h))
		h = F.relu(m.nh04(h))
		h = F.relu(m.nh05(h))
		hh = F.relu(m.nh06(h))

		h = F.relu(m.nl01(chainer.Variable(x[2], volatile=volatile)))
		h = F.relu(m.nl02(h))
		h = F.relu(m.nl03(h))
		h = F.relu(m.nl04(h))
		h = F.relu(m.nl05(h))
		hl = F.relu(m.nl06(h))

		h = F.relu(m.nc01(chainer.Variable(x[3], volatile=volatile)))
		h = F.relu(m.nc02(h))
		h = F.relu(m.nc03(h))
		h = F.relu(m.nc04(h))
		h = F.relu(m.nc05(h))
		hc = F.relu(m.nc06(h))

		# 混ぜる
		h = ho + hh + hl + hc

		# 広げていく
		h = F.relu(m.nx07(h))
		h = F.relu(m.nx08(h))
		h = F.relu(m.nx09(h))
		h = F.relu(m.nx10(h))
		h = m.nx11(h)

		return h

	def buildMiniBatchData(m, dataset, batchIndices):
		"""学習データセットの指定位置から全ミニバッチデータを作成する"""
		batchSize = batchIndices.shape[0]
		inCount = m.inCount
		x = np.empty(shape=(4, batchSize, inCount), dtype=np.float32)
		for i, p in enumerate(batchIndices):
			pe = p + inCount
			x[0,i,:] = dataset[0,p:pe]
			x[1,i,:] = dataset[1,p:pe]
			x[2,i,:] = dataset[2,p:pe]
			x[3,i,:] = dataset[3,p:pe]
		return x

	def getModelKind(m):
		return "zigzag"


class OpenHighLowSqwzN6N5(chainer.Chain):
	"""
	絞って混ぜて広げていくスタイル
	"""
	def __init__(m):
		pass

	def create(m, inCount, unitCount, outCount, gpu, train=True):
		uc1 = 8 * unitCount // 10
		uc2 = 6 * unitCount // 10
		uc3 = 4 * unitCount // 10
		uc4 = 2 * unitCount // 10
		uc5 = 1 * unitCount // 10
		super().__init__(
			no01=L.Linear(inCount, uc1),
			nh01=L.Linear(inCount, uc1),
			nl01=L.Linear(inCount, uc1),

			no02=L.Linear(uc1, uc2),
			nh02=L.Linear(uc1, uc2),
			nl02=L.Linear(uc1, uc2),

			no03=L.Linear(uc2, uc3),
			nh03=L.Linear(uc2, uc3),
			nl03=L.Linear(uc2, uc3),

			no04=L.Linear(uc3, uc4),
			nh04=L.Linear(uc3, uc4),
			nl04=L.Linear(uc3, uc4),

			no05=L.Linear(uc4, uc5),
			nh05=L.Linear(uc4, uc5),
			nl05=L.Linear(uc4, uc5),

			no06=L.Linear(uc5, uc5),
			nh06=L.Linear(uc5, uc5),
			nl06=L.Linear(uc5, uc5),

			nx07=L.Linear(uc5, uc4),
			nx08=L.Linear(uc4, uc3),
			nx09=L.Linear(uc3, uc2),
			nx10=L.Linear(uc2, uc1),
			nx11=L.Linear(uc1, outCount)
		)
		m.inCount = inCount
		m.train = train

	#@jit(nopython=True)
	def reset_state(m):
		pass

	#@jit(nopython=True)
	def __call__(m, x, volatile):
		# 開始値、高値、低値それぞれを絞る
		h = F.relu(m.no01(chainer.Variable(x[0], volatile=volatile)))
		h = F.relu(m.no02(h))
		h = F.relu(m.no03(h))
		h = F.relu(m.no04(h))
		h = F.relu(m.no05(h))
		ho = F.relu(m.no06(h))

		h = F.relu(m.nh01(chainer.Variable(x[1], volatile=volatile)))
		h = F.relu(m.nh02(h))
		h = F.relu(m.nh03(h))
		h = F.relu(m.nh04(h))
		h = F.relu(m.nh05(h))
		hh = F.relu(m.nh06(h))

		h = F.relu(m.nl01(chainer.Variable(x[2], volatile=volatile)))
		h = F.relu(m.nl02(h))
		h = F.relu(m.nl03(h))
		h = F.relu(m.nl04(h))
		h = F.relu(m.nl05(h))
		hl = F.relu(m.nl06(h))

		# 混ぜる
		h = hh * 0.25 + hl * 0.25 + ho * 0.5

		# 広げていく
		h = F.relu(m.nx07(h))
		h = F.relu(m.nx08(h))
		h = F.relu(m.nx09(h))
		h = F.relu(m.nx10(h))
		h = m.nx11(h)

		return h

	def buildMiniBatchData(m, dataset, batchIndices):
		"""学習データセットの指定位置から全ミニバッチデータを作成する"""
		batchSize = batchIndices.shape[0]
		inCount = m.inCount
		x = np.empty(shape=(3, batchSize, inCount), dtype=np.float32)
		for i, p in enumerate(batchIndices):
			pe = p + inCount
			o = dataset[0, p : pe]
			a = (o.max() + o.min()) * 0.5
			x[0,i,:] = o - a
			x[1,i,:] = dataset[1, p : pe] - a
			x[2,i,:] = dataset[2, p : pe] - a
		return x

	def getModelKind(m):
		return "clas"


class OpenHighLowDiv2N10N1(chainer.Chain):
	def __init__(m):
		pass

	def create(m, inCount, unitCount, outCount, gpu, train=True):
		uc1 = unitCount
		uc2 = unitCount / 2
		super().__init__(
			no01_1=L.Linear(inCount, uc1),
			no01_2=L.Linear(inCount, uc2),
			nh01_1=L.Linear(inCount, uc1),
			nh01_2=L.Linear(inCount, uc2),
			nl01_1=L.Linear(inCount, uc1),
			nl01_2=L.Linear(inCount, uc2),

			no02_1=L.Linear(uc1, uc1),
			no02_2=L.Linear(uc2, uc2),
			nh02_1=L.Linear(uc1, uc1),
			nh02_2=L.Linear(uc2, uc2),
			nl02_1=L.Linear(uc1, uc1),
			nl02_2=L.Linear(uc2, uc2),

			no03_1=L.Linear(uc1, uc1),
			no03_2=L.Linear(uc2, uc2),
			nh03_1=L.Linear(uc1, uc1),
			nh03_2=L.Linear(uc2, uc2),
			nl03_1=L.Linear(uc1, uc1),
			nl03_2=L.Linear(uc2, uc2),

			no04_1=L.Linear(uc1, uc1),
			no04_2=L.Linear(uc2, uc2),
			nh04_1=L.Linear(uc1, uc1),
			nh04_2=L.Linear(uc2, uc2),
			nl04_1=L.Linear(uc1, uc1),
			nl04_2=L.Linear(uc2, uc2),

			no05_1=L.Linear(uc1, uc1),
			no05_2=L.Linear(uc2, uc2),
			nh05_1=L.Linear(uc1, uc1),
			nh05_2=L.Linear(uc2, uc2),
			nl05_1=L.Linear(uc1, uc1),
			nl05_2=L.Linear(uc2, uc2),

			no06_1=L.Linear(uc1, uc1),
			no06_2=L.Linear(uc2, uc2),
			nh06_1=L.Linear(uc1, uc1),
			nh06_2=L.Linear(uc2, uc2),
			nl06_1=L.Linear(uc1, uc1),
			nl06_2=L.Linear(uc2, uc2),

			no07_1=L.Linear(uc1, uc1),
			no07_2=L.Linear(uc2, uc2),
			nh07_1=L.Linear(uc1, uc1),
			nh07_2=L.Linear(uc2, uc2),
			nl07_1=L.Linear(uc1, uc1),
			nl07_2=L.Linear(uc2, uc2),

			no08_1=L.Linear(uc1, uc1),
			no08_2=L.Linear(uc2, uc2),
			nh08_1=L.Linear(uc1, uc1),
			nh08_2=L.Linear(uc2, uc2),
			nl08_1=L.Linear(uc1, uc1),
			nl08_2=L.Linear(uc2, uc2),

			no09_1=L.Linear(uc1, uc1),
			no09_2=L.Linear(uc2, uc2),
			nh09_1=L.Linear(uc1, uc1),
			nh09_2=L.Linear(uc2, uc2),
			nl09_1=L.Linear(uc1, uc1),
			nl09_2=L.Linear(uc2, uc2),

			no10_1=L.Linear(uc1, unitCount),
			no10_2=L.Linear(uc2, unitCount),
			nh10_1=L.Linear(uc1, unitCount),
			nh10_2=L.Linear(uc2, unitCount),
			nl10_1=L.Linear(uc1, unitCount),
			nl10_2=L.Linear(uc2, unitCount),

			nx11=L.Linear(unitCount, outCount)
		)
		m.inCount = inCount
		m.train = train

	#@jit(nopython=True)
	def reset_state(m):
		pass

	#@jit(nopython=True)
	def __call__(m, x, volatile):
		# 値取得
		x1 = chainer.Variable(x[0], volatile=volatile)
		x2 = chainer.Variable(x[1], volatile=volatile)
		x3 = chainer.Variable(x[2], volatile=volatile)

		# 次元圧縮して畳み込みの様な効果を期待
		ho1 = F.relu(m.no01_1(x1))
		ho2 = F.relu(m.no01_2(x1))
		hh1 = F.relu(m.nh01_1(x2))
		hh2 = F.relu(m.nh01_2(x2))
		hl1 = F.relu(m.nl01_1(x3))
		hl2 = F.relu(m.nl01_2(x3))

		# それぞれをレイヤに通していく
		ho1 = F.relu(m.no02_1(ho1))
		ho2 = F.relu(m.no02_2(ho2))
		hh1 = F.relu(m.nh02_1(hh1))
		hh2 = F.relu(m.nh02_2(hh2))
		hl1 = F.relu(m.nl02_1(hl1))
		hl2 = F.relu(m.nl02_2(hl2))

		ho1 = F.relu(m.no03_1(ho1))
		ho2 = F.relu(m.no03_2(ho2))
		hh1 = F.relu(m.nh03_1(hh1))
		hh2 = F.relu(m.nh03_2(hh2))
		hl1 = F.relu(m.nl03_1(hl1))
		hl2 = F.relu(m.nl03_2(hl2))

		ho1 = F.relu(m.no04_1(ho1))
		ho2 = F.relu(m.no04_2(ho2))
		hh1 = F.relu(m.nh04_1(hh1))
		hh2 = F.relu(m.nh04_2(hh2))
		hl1 = F.relu(m.nl04_1(hl1))
		hl2 = F.relu(m.nl04_2(hl2))

		ho1 = F.relu(m.no05_1(ho1))
		ho2 = F.relu(m.no05_2(ho2))
		hh1 = F.relu(m.nh05_1(hh1))
		hh2 = F.relu(m.nh05_2(hh2))
		hl1 = F.relu(m.nl05_1(hl1))
		hl2 = F.relu(m.nl05_2(hl2))

		ho1 = F.relu(m.no06_1(ho1))
		ho2 = F.relu(m.no06_2(ho2))
		hh1 = F.relu(m.nh06_1(hh1))
		hh2 = F.relu(m.nh06_2(hh2))
		hl1 = F.relu(m.nl06_1(hl1))
		hl2 = F.relu(m.nl06_2(hl2))

		ho1 = F.relu(m.no07_1(ho1))
		ho2 = F.relu(m.no07_2(ho2))
		hh1 = F.relu(m.nh07_1(hh1))
		hh2 = F.relu(m.nh07_2(hh2))
		hl1 = F.relu(m.nl07_1(hl1))
		hl2 = F.relu(m.nl07_2(hl2))

		ho1 = F.relu(m.no08_1(ho1))
		ho2 = F.relu(m.no08_2(ho2))
		hh1 = F.relu(m.nh08_1(hh1))
		hh2 = F.relu(m.nh08_2(hh2))
		hl1 = F.relu(m.nl08_1(hl1))
		hl2 = F.relu(m.nl08_2(hl2))

		ho1 = F.relu(m.no09_1(ho1))
		ho2 = F.relu(m.no09_2(ho2))
		hh1 = F.relu(m.nh09_1(hh1))
		hh2 = F.relu(m.nh09_2(hh2))
		hl1 = F.relu(m.nl09_1(hl1))
		hl2 = F.relu(m.nl09_2(hl2))

		# 同じ次元数に整える
		ho1 = F.relu(m.no10_1(ho1))
		ho2 = F.relu(m.no10_2(ho2))
		hh1 = F.relu(m.nh10_1(hh1))
		hh2 = F.relu(m.nh10_2(hh2))
		hl1 = F.relu(m.nl10_1(hl1))
		hl2 = F.relu(m.nl10_2(hl2))

		# 全レイヤを合成
		h = hh1 + hh2 + hl1 + hl2 + ho1 + ho2

		# 最後に１レイヤ通す
		h = m.nx11(h)

		return h

	def buildMiniBatchData(m, dataset, batchIndices):
		"""学習データセットの指定位置から全ミニバッチデータを作成する"""
		batchSize = batchIndices.shape[0]
		inCount = m.inCount
		x = np.empty(shape=(3, batchSize, inCount), dtype=np.float32)
		for i, p in enumerate(batchIndices):
			pe = p + inCount
			o = dataset[0, p : pe]
			a = (o.max() + o.min()) * 0.5
			x[0,i,:] = o - a
			x[1,i,:] = o - dataset[1, p : pe]
			x[2,i,:] = o - dataset[2, p : pe]
		return x

	def getModelKind(m):
		return "clas"


class OpenDiv5N10N1(chainer.Chain):
	def __init__(m):
		pass

	def create(m, inCount, unitCount, outCount, gpu, train=True):
		uc1 = unitCount
		uc2 = 8 * unitCount // 10
		uc3 = 6 * unitCount // 10
		uc4 = 4 * unitCount // 10
		uc5 = 2 * unitCount // 10
		uc6 = 3 * unitCount // 10
		super().__init__(
			n1_1=L.Linear(inCount, uc1),
			n1_2=L.Linear(inCount, uc2),
			n1_3=L.Linear(inCount, uc3),
			n1_4=L.Linear(inCount, uc4),
			n1_5=L.Linear(inCount, uc5),

			n2_1=L.Linear(uc1, uc1),
			n2_2=L.Linear(uc2, uc2),
			n2_3=L.Linear(uc3, uc3),
			n2_4=L.Linear(uc4, uc4),
			n2_5=L.Linear(uc5, uc5),

			n3_1=L.Linear(uc1, uc1),
			n3_2=L.Linear(uc2, uc2),
			n3_3=L.Linear(uc3, uc3),
			n3_4=L.Linear(uc4, uc4),
			n3_5=L.Linear(uc5, uc5),

			n4_1=L.Linear(uc1, uc1),
			n4_2=L.Linear(uc2, uc2),
			n4_3=L.Linear(uc3, uc3),
			n4_4=L.Linear(uc4, uc4),
			n4_5=L.Linear(uc5, uc5),

			n5_1=L.Linear(uc1, uc1),
			n5_2=L.Linear(uc2, uc2),
			n5_3=L.Linear(uc3, uc3),
			n5_4=L.Linear(uc4, uc4),
			n5_5=L.Linear(uc5, uc5),

			n6_1=L.Linear(uc1, uc1),
			n6_2=L.Linear(uc2, uc2),
			n6_3=L.Linear(uc3, uc3),
			n6_4=L.Linear(uc4, uc4),
			n6_5=L.Linear(uc5, uc5),

			n7_1=L.Linear(uc1, uc1),
			n7_2=L.Linear(uc2, uc2),
			n7_3=L.Linear(uc3, uc3),
			n7_4=L.Linear(uc4, uc4),
			n7_5=L.Linear(uc5, uc5),

			n8_1=L.Linear(uc1, uc1),
			n8_2=L.Linear(uc2, uc2),
			n8_3=L.Linear(uc3, uc3),
			n8_4=L.Linear(uc4, uc4),
			n8_5=L.Linear(uc5, uc5),

			n9_1=L.Linear(uc1, uc1),
			n9_2=L.Linear(uc2, uc2),
			n9_3=L.Linear(uc3, uc3),
			n9_4=L.Linear(uc4, uc4),
			n9_5=L.Linear(uc5, uc5),

			n10_1=L.Linear(uc1, uc6),
			n10_2=L.Linear(uc2, uc6),
			n10_3=L.Linear(uc3, uc6),
			n10_4=L.Linear(uc4, uc6),
			n10_5=L.Linear(uc5, uc6),

			n11=L.Linear(uc6, outCount),
		)
		m.inCount = inCount
		m.train = train

	#@jit(nopython=True)
	def reset_state(m):
		pass

	#@jit(nopython=True)
	def __call__(m, x, volatile):
		# 値取得
		x = chainer.Variable(x, volatile=volatile)

		# ４種類の次元数に圧縮
		h1 = F.relu(m.n1_1(x))
		h2 = F.relu(m.n1_2(x))
		h3 = F.relu(m.n1_3(x))
		h4 = F.relu(m.n1_4(x))
		h5 = F.relu(m.n1_5(x))

		# 圧縮された次元数で5レイヤ分処理
		h1 = F.relu(m.n2_1(h1))
		h2 = F.relu(m.n2_2(h2))
		h3 = F.relu(m.n2_3(h3))
		h4 = F.relu(m.n2_4(h4))
		h5 = F.relu(m.n2_5(h5))

		h1 = F.relu(m.n3_1(h1))
		h2 = F.relu(m.n3_2(h2))
		h3 = F.relu(m.n3_3(h3))
		h4 = F.relu(m.n3_4(h4))
		h5 = F.relu(m.n3_5(h5))

		h1 = F.relu(m.n4_1(h1))
		h2 = F.relu(m.n4_2(h2))
		h3 = F.relu(m.n4_3(h3))
		h4 = F.relu(m.n4_4(h4))
		h5 = F.relu(m.n4_5(h5))

		h1 = F.relu(m.n5_1(h1))
		h2 = F.relu(m.n5_2(h2))
		h3 = F.relu(m.n5_3(h3))
		h4 = F.relu(m.n5_4(h4))
		h5 = F.relu(m.n5_5(h5))

		h1 = F.relu(m.n6_1(h1))
		h2 = F.relu(m.n6_2(h2))
		h3 = F.relu(m.n6_3(h3))
		h4 = F.relu(m.n6_4(h4))
		h5 = F.relu(m.n6_5(h5))

		h1 = F.relu(m.n7_1(h1))
		h2 = F.relu(m.n7_2(h2))
		h3 = F.relu(m.n7_3(h3))
		h4 = F.relu(m.n7_4(h4))
		h5 = F.relu(m.n7_5(h5))

		h1 = F.relu(m.n8_1(h1))
		h2 = F.relu(m.n8_2(h2))
		h3 = F.relu(m.n8_3(h3))
		h4 = F.relu(m.n8_4(h4))
		h5 = F.relu(m.n8_5(h5))

		h1 = F.relu(m.n9_1(h1))
		h2 = F.relu(m.n9_2(h2))
		h3 = F.relu(m.n9_3(h3))
		h4 = F.relu(m.n9_4(h4))
		h5 = F.relu(m.n9_5(h5))

		# 同じ次元数に整える
		h1 = F.relu(m.n10_1(h1))
		h2 = F.relu(m.n10_2(h2))
		h3 = F.relu(m.n10_3(h3))
		h4 = F.relu(m.n10_4(h4))
		h5 = F.relu(m.n10_5(h5))

		# 全レイヤを加算
		h = h1 + h2 + h3 + h4 + h5

		# 最後に１レイヤ通す
		h = m.n11(h)

		return h

	def buildMiniBatchData(m, dataset, batchIndices):
		"""学習データセットの指定位置から全ミニバッチデータを作成する"""
		batchSize = batchIndices.shape[0]
		inCount = m.inCount
		x = np.empty(shape=(batchSize, inCount), dtype=np.float32)
		for i, p in enumerate(batchIndices):
			f = dataset[0, p : p + inCount]
			x[i,:] = f - (f.max() + f.min()) * 0.5
		return x

	def getModelKind(m):
		return "clas"


class OpenDiv5N6N1(chainer.Chain):
	def __init__(m):
		pass

	def create(m, inCount, unitCount, outCount, gpu, train=True):
		uc1 = unitCount
		uc2 = 8 * unitCount // 10
		uc3 = 6 * unitCount // 10
		uc4 = 4 * unitCount // 10
		uc5 = 2 * unitCount // 10
		uc6 = 3 * unitCount // 10
		super().__init__(
			n1_1=L.Linear(inCount, uc1),
			n1_2=L.Linear(inCount, uc2),
			n1_3=L.Linear(inCount, uc3),
			n1_4=L.Linear(inCount, uc4),
			n1_5=L.Linear(inCount, uc5),

			n2_1=L.Linear(uc1, uc1),
			n2_2=L.Linear(uc2, uc2),
			n2_3=L.Linear(uc3, uc3),
			n2_4=L.Linear(uc4, uc4),
			n2_5=L.Linear(uc5, uc5),

			n3_1=L.Linear(uc1, uc1),
			n3_2=L.Linear(uc2, uc2),
			n3_3=L.Linear(uc3, uc3),
			n3_4=L.Linear(uc4, uc4),
			n3_5=L.Linear(uc5, uc5),

			n4_1=L.Linear(uc1, uc1),
			n4_2=L.Linear(uc2, uc2),
			n4_3=L.Linear(uc3, uc3),
			n4_4=L.Linear(uc4, uc4),
			n4_5=L.Linear(uc5, uc5),

			n5_1=L.Linear(uc1, uc1),
			n5_2=L.Linear(uc2, uc2),
			n5_3=L.Linear(uc3, uc3),
			n5_4=L.Linear(uc4, uc4),
			n5_5=L.Linear(uc5, uc5),

			n6_1=L.Linear(uc1, uc6),
			n6_2=L.Linear(uc2, uc6),
			n6_3=L.Linear(uc3, uc6),
			n6_4=L.Linear(uc4, uc6),
			n6_5=L.Linear(uc5, uc6),

			n7=L.Linear(uc6, outCount),
		)
		m.inCount = inCount
		m.train = train

	#@jit(nopython=True)
	def reset_state(m):
		pass

	#@jit(nopython=True)
	def __call__(m, x, volatile):
		# 値取得
		x = chainer.Variable(x, volatile=volatile)

		# ４種類の次元数に圧縮
		h1 = F.relu(m.n1_1(x))
		h2 = F.relu(m.n1_2(x))
		h3 = F.relu(m.n1_3(x))
		h4 = F.relu(m.n1_4(x))
		h5 = F.relu(m.n1_5(x))

		# 圧縮された次元数で5レイヤ分処理
		h1 = F.relu(m.n2_1(h1))
		h2 = F.relu(m.n2_2(h2))
		h3 = F.relu(m.n2_3(h3))
		h4 = F.relu(m.n2_4(h4))
		h5 = F.relu(m.n2_5(h5))

		h1 = F.relu(m.n3_1(h1))
		h2 = F.relu(m.n3_2(h2))
		h3 = F.relu(m.n3_3(h3))
		h4 = F.relu(m.n3_4(h4))
		h5 = F.relu(m.n3_5(h5))

		h1 = F.relu(m.n4_1(h1))
		h2 = F.relu(m.n4_2(h2))
		h3 = F.relu(m.n4_3(h3))
		h4 = F.relu(m.n4_4(h4))
		h5 = F.relu(m.n4_5(h5))

		h1 = F.relu(m.n5_1(h1))
		h2 = F.relu(m.n5_2(h2))
		h3 = F.relu(m.n5_3(h3))
		h4 = F.relu(m.n5_4(h4))
		h5 = F.relu(m.n5_5(h5))

		# 同じ次元数に整える
		h1 = F.relu(m.n6_1(h1))
		h2 = F.relu(m.n6_2(h2))
		h3 = F.relu(m.n6_3(h3))
		h4 = F.relu(m.n6_4(h4))
		h5 = F.relu(m.n6_5(h5))

		# 全レイヤを加算
		h = h1 + h2 + h3 + h4 + h5

		# 最後に１レイヤ通す
		h = m.n7(h)

		return h

	def buildMiniBatchData(m, dataset, batchIndices):
		"""学習データセットの指定位置から全ミニバッチデータを作成する"""
		batchSize = batchIndices.shape[0]
		inCount = m.inCount
		x = np.empty(shape=(batchSize, inCount), dtype=np.float32)
		for i, p in enumerate(batchIndices):
			f = dataset[0, p : p + inCount]
			x[i,:] = f - (f.max() + f.min()) * 0.5
		return x

	def getModelKind(m):
		return "clas"


class OpenDiv4N6N1(chainer.Chain):
	def __init__(m):
		pass

	def create(m, inCount, unitCount, outCount, gpu, train=True):
		uc1 = unitCount
		uc2 = 8 * unitCount // 10
		uc3 = 6 * unitCount // 10
		uc4 = 4 * unitCount // 10
		uc5 = 2 * unitCount // 10
		super().__init__(
			n1_1=L.Linear(inCount, uc1),
			n1_2=L.Linear(inCount, uc2),
			n1_3=L.Linear(inCount, uc3),
			n1_4=L.Linear(inCount, uc4),

			n2_1=L.Linear(uc1, uc1),
			n2_2=L.Linear(uc2, uc2),
			n2_3=L.Linear(uc3, uc3),
			n2_4=L.Linear(uc4, uc4),

			n3_1=L.Linear(uc1, uc1),
			n3_2=L.Linear(uc2, uc2),
			n3_3=L.Linear(uc3, uc3),
			n3_4=L.Linear(uc4, uc4),

			n4_1=L.Linear(uc1, uc1),
			n4_2=L.Linear(uc2, uc2),
			n4_3=L.Linear(uc3, uc3),
			n4_4=L.Linear(uc4, uc4),

			n5_1=L.Linear(uc1, uc1),
			n5_2=L.Linear(uc2, uc2),
			n5_3=L.Linear(uc3, uc3),
			n5_4=L.Linear(uc4, uc4),

			n6_1=L.Linear(uc1, uc5),
			n6_2=L.Linear(uc2, uc5),
			n6_3=L.Linear(uc3, uc5),
			n6_4=L.Linear(uc4, uc5),

			#b1_1=L.Bilinear(uc5, uc5, uc5),
			#b1_2=L.Bilinear(uc5, uc5, uc5),
			#b2=L.Bilinear(uc5, uc5, uc5),

			n7=L.Linear(uc5, outCount),
		)
		m.inCount = inCount
		m.train = train

	#@jit(nopython=True)
	def reset_state(m):
		pass

	#@jit(nopython=True)
	def __call__(m, x, volatile):
		# 値取得
		x = chainer.Variable(x, volatile=volatile)

		# ４種類の次元数に圧縮
		h1 = F.relu(m.n1_1(x))
		h2 = F.relu(m.n1_2(x))
		h3 = F.relu(m.n1_3(x))
		h4 = F.relu(m.n1_4(x))

		# 圧縮された次元数で４レイヤ分処理
		h1 = F.relu(m.n2_1(h1))
		h2 = F.relu(m.n2_2(h2))
		h3 = F.relu(m.n2_3(h3))
		h4 = F.relu(m.n2_4(h4))

		h1 = F.relu(m.n3_1(h1))
		h2 = F.relu(m.n3_2(h2))
		h3 = F.relu(m.n3_3(h3))
		h4 = F.relu(m.n3_4(h4))

		h1 = F.relu(m.n4_1(h1))
		h2 = F.relu(m.n4_2(h2))
		h3 = F.relu(m.n4_3(h3))
		h4 = F.relu(m.n4_4(h4))

		h1 = F.relu(m.n5_1(h1))
		h2 = F.relu(m.n5_2(h2))
		h3 = F.relu(m.n5_3(h3))
		h4 = F.relu(m.n5_4(h4))

		# 同じ次元数に整える
		h1 = F.relu(m.n6_1(h1))
		h2 = F.relu(m.n6_2(h2))
		h3 = F.relu(m.n6_3(h3))
		h4 = F.relu(m.n6_4(h4))

		# バイリニアで合成
		#h12 = m.b1_1(h1, h2)
		#h34 = m.b1_2(h3, h4) これ使うと何故かGPUメモリ不正アクセスで落ちる
		#h = m.b2(h12, h34)
		h = h1 + h2 + h3 + h4

		# 最後に１レイヤ通す
		h = m.n7(h)

		return h

	def buildMiniBatchData(m, dataset, batchIndices):
		"""学習データセットの指定位置から全ミニバッチデータを作成する"""
		batchSize = batchIndices.shape[0]
		inCount = m.inCount
		x = np.empty(shape=(batchSize, inCount), dtype=np.float32)
		for i, p in enumerate(batchIndices):
			f = dataset[0, p : p + inCount]
			x[i,:] = f - (f.max() + f.min()) * 0.5
		return x

	def getModelKind(m):
		return "clas"


class N15ReluSqueeze(chainer.Chain):
	def __init__(m):
		pass

	def create(m, inCount, unitCount, outCount, gpu, train=True):
		uc1 = unitCount
		uc2 = unitCount // 2
		uc3 = unitCount // 4
		super().__init__(
			l1=L.Linear(inCount, uc1),
			l2=L.Linear(uc1, uc1),
			l3=L.Linear(uc1, uc1),
			l4=L.Linear(uc1, uc2),
			l5=L.Linear(uc2, uc2),
			l6=L.Linear(uc2, uc2),
			l7=L.Linear(uc2, uc3),
			l8=L.Linear(uc3, uc3),
			l9=L.Linear(uc3, uc3),
			l10=L.Linear(uc3, uc2),
			l11=L.Linear(uc2, uc2),
			l12=L.Linear(uc2, uc1),
			l13=L.Linear(uc1, uc1),
			l14=L.Linear(uc1, uc1),
			l15=L.Linear(uc1, outCount),
		)
		m.inCount = inCount
		m.train = train

	def reset_state(m):
		pass

	#@jit
	def __call__(m, x, volatile):
		x = chainer.Variable(x, volatile=volatile)
		h = F.relu(m.l1(x))
		h = F.relu(m.l2(h))
		h = F.relu(m.l3(h))
		h = F.relu(m.l4(h))
		h = F.relu(m.l5(h))
		h = F.relu(m.l6(h))
		h = F.relu(m.l7(h))
		h = F.relu(m.l8(h))
		h = F.relu(m.l9(h))
		h = F.relu(m.l10(h))
		h = F.relu(m.l11(h))
		h = F.relu(m.l12(h))
		h = F.relu(m.l13(h))
		h = F.relu(m.l14(h))
		h = m.l15(h)
		return h

	def buildMiniBatchData(m, dataset, batchIndices):
		"""学習データセットの指定位置から全ミニバッチデータを作成する"""
		batchSize = batchIndices.shape[0]
		inCount = m.inCount
		x = np.empty(shape=(batchSize, inCount), dtype=np.float32)
		for i, p in enumerate(batchIndices):
			f = dataset[0, p : p + inCount]
			x[i,:] = f - (f.max() + f.min()) * 0.5
		return x

	def getModelKind(m):
		return "clas"

class N15Relu(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_midunits),
			l4=L.Linear(n_midunits, n_midunits),
			l5=L.Linear(n_midunits, n_midunits),
			l6=L.Linear(n_midunits, n_midunits),
			l7=L.Linear(n_midunits, n_midunits),
			l8=L.Linear(n_midunits, n_midunits),
			l9=L.Linear(n_midunits, n_midunits),
			l10=L.Linear(n_midunits, n_midunits),
			l11=L.Linear(n_midunits, n_midunits),
			l12=L.Linear(n_midunits, n_midunits),
			l13=L.Linear(n_midunits, n_midunits),
			l14=L.Linear(n_midunits, n_midunits),
			l15=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		pass

	#@jit
	def __call__(m, x):
		h = F.relu(m.l1(x))
		h = F.relu(m.l2(h))
		h = F.relu(m.l3(h))
		h = F.relu(m.l4(h))
		h = F.relu(m.l5(h))
		h = F.relu(m.l6(h))
		h = F.relu(m.l7(h))
		h = F.relu(m.l8(h))
		h = F.relu(m.l9(h))
		h = F.relu(m.l10(h))
		h = F.relu(m.l11(h))
		h = F.relu(m.l12(h))
		h = F.relu(m.l13(h))
		h = F.relu(m.l14(h))
		h = m.l15(h)
		return h

	def getModelKind(self):
		return "clas"

class N10Relu(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_midunits),
			l4=L.Linear(n_midunits, n_midunits),
			l5=L.Linear(n_midunits, n_midunits),
			l6=L.Linear(n_midunits, n_midunits),
			l7=L.Linear(n_midunits, n_midunits),
			l8=L.Linear(n_midunits, n_midunits),
			l9=L.Linear(n_midunits, n_midunits),
			l10=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		pass

	#@jit
	def __call__(m, x):
		h = F.relu(m.l1(x))
		h = F.relu(m.l2(h))
		h = F.relu(m.l3(h))
		h = F.relu(m.l4(h))
		h = F.relu(m.l5(h))
		h = F.relu(m.l6(h))
		h = F.relu(m.l7(h))
		h = F.relu(m.l8(h))
		h = F.relu(m.l9(h))
		h = m.l10(h)
		return h

	def getModelKind(self):
		return "clas"

class N5Relu(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_midunits),
			l4=L.Linear(n_midunits, n_midunits),
			l5=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		pass

	#@jit
	def __call__(m, x):
		h = F.relu(m.l1(x))
		h = F.relu(m.l2(h))
		h = F.relu(m.l3(h))
		h = F.relu(m.l4(h))
		h = m.l5(h)
		return h

	def getModelKind(self):
		return "clas"

class N7ReluDo(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_midunits),
			l4=L.Linear(n_midunits, n_midunits),
			l5=L.Linear(n_midunits, n_midunits),
			l6=L.Linear(n_midunits, n_midunits),
			l7=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		pass

	#@jit
	def __call__(m, x):
		do = mk_clas.dropoutRatio
		tr = m.train
		h = F.dropout(F.relu(m.l1(x)), ratio=do, train=tr)
		h = F.relu(m.l2(h))
		h = F.dropout(F.relu(m.l3(h)), ratio=do, train=tr)
		h = F.relu(m.l4(h))
		h = F.dropout(F.relu(m.l5(h)), ratio=do, train=tr)
		h = F.relu(m.l6(h))
		h = m.l7(h)
		return h

	def getModelKind(self):
		return "clas"

class N7Relu(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_midunits),
			l4=L.Linear(n_midunits, n_midunits),
			l5=L.Linear(n_midunits, n_midunits),
			l6=L.Linear(n_midunits, n_midunits),
			l7=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		pass

	#@jit
	def __call__(m, x):
		h = F.relu(m.l1(x))
		h = F.relu(m.l2(h))
		h = F.relu(m.l3(h))
		h = F.relu(m.l4(h))
		h = F.relu(m.l5(h))
		h = F.relu(m.l6(h))
		h = m.l7(h)
		return h

	def getModelKind(self):
		return "clas"

class N7relu7ls(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_midunits),
			l4=L.Linear(n_midunits, n_midunits),
			l5=L.Linear(n_midunits, n_midunits),
			l6=L.Linear(n_midunits, n_midunits),
			l7=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		pass

	#@jit
	def __call__(m, x):
		return F.log_softmax(F.relu(m.l7(F.relu(m.l6(F.relu(m.l5(F.relu(m.l4(F.relu(m.l3(F.relu(m.l2(F.relu(m.l1(x)))))))))))))))

	def getModelKind(self):
		return "clas"

class N7ls(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_midunits),
			l4=L.Linear(n_midunits, n_midunits),
			l5=L.Linear(n_midunits, n_midunits),
			l6=L.Linear(n_midunits, n_midunits),
			l7=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		pass

	#@jit
	def __call__(m, x):
		return F.log_softmax(m.l7(F.relu(m.l6(F.relu(m.l5(F.relu(m.l4(F.relu(m.l3(F.relu(m.l2(F.relu(m.l1(x))))))))))))))

	def getModelKind(self):
		return "clas"

class N7relu0ls(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_midunits),
			l4=L.Linear(n_midunits, n_midunits),
			l5=L.Linear(n_midunits, n_midunits),
			l6=L.Linear(n_midunits, n_midunits),
			l7=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		pass

	#@jit
	def __call__(m, x):
		return F.log_softmax(m.l7(m.l6(m.l5(m.l4(m.l3(m.l2(m.l1(x))))))))

	def getModelKind(self):
		return "clas"

class N6ls(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_midunits),
			l4=L.Linear(n_midunits, n_midunits),
			l5=L.Linear(n_midunits, n_midunits),
			l6=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		pass

	#@jit
	def __call__(m, x):
		return F.log_softmax(m.l6(F.relu(m.l5(F.relu(m.l4(F.relu(m.l3(F.relu(m.l2(F.relu(m.l1(x))))))))))))

	def getModelKind(self):
		return "clas"

class N6(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_midunits),
			l4=L.Linear(n_midunits, n_midunits),
			l5=L.Linear(n_midunits, n_midunits),
			l6=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		pass

	#@jit
	def __call__(m, x):
		return m.l6(F.relu(m.l5(F.relu(m.l4(F.relu(m.l3(F.relu(m.l2(F.relu(m.l1(x)))))))))))

	def getModelKind(self):
		return "clas"

class NNNNNN(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_midunits),
			l4=L.Linear(n_midunits, n_midunits),
			l5=L.Linear(n_midunits, n_midunits),
			l6=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		pass

	#@jit
	def __call__(self, x):
		return F.log_softmax(self.l6(F.relu(self.l5(F.relu(self.l4(F.relu(self.l3(F.relu(self.l2(F.relu(self.l1(x))))))))))))

	def getModelKind(self):
		return "clas"

class NNNNN(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_midunits),
			l4=L.Linear(n_midunits, n_midunits),
			l5=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		pass

	#@jit
	def __call__(self, x):
		return F.log_softmax(F.relu(self.l5(F.relu(self.l4(F.relu(self.l3(F.relu(self.l2(F.relu(self.l1(x)))))))))))

	def getModelKind(self):
		return "clas"

class NNNN(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_midunits),
			l4=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		pass

	#@jit
	def __call__(self, x):
		return F.log_softmax(F.relu(self.l4(F.relu(self.l3(F.relu(self.l2(F.relu(self.l1(x)))))))))

	def getModelKind(self):
		return "clas"

class NNN1sm(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		pass

	#@jit
	def __call__(m, x):
		return F.log_softmax(m.l3(F.relu(m.l2(F.relu(m.l1(x))))))

	def getModelKind(self):
		return "clas"

class NNN2sm(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		pass

	#@jit
	def __call__(m, x):
		return F.log_softmax(F.relu(m.l3(F.relu(m.l2(F.relu(m.l1(x)))))))

	def getModelKind(self):
		return "clas"

class NDNDN(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_midunits),
			l4=L.Linear(n_midunits, n_midunits),
			l5=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		pass

	#@jit
	def __call__(self, x):
		h1 = self.l1(x)
		h2 = self.l2(F.dropout(F.relu(h1), ratio=mk_clas.dropoutRatio, train=self.train))
		h3 = self.l3(F.relu(h2))
		h4 = self.l4(F.dropout(F.relu(h3), ratio=mk_clas.dropoutRatio, train=self.train))
		y = self.l5(F.relu(h4))
		return y

	def getModelKind(self):
		return "clas"

class NDDDDD(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_midunits),
			l4=L.Linear(n_midunits, n_midunits),
			l5=L.Linear(n_midunits, n_midunits),
			l6=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		pass

	#@jit
	def __call__(self, x):
		h1 = self.l1(x)
		h2 = self.l2(F.dropout(F.relu(h1), ratio=mk_clas.dropoutRatio, train=self.train))
		h3 = self.l3(F.dropout(F.relu(h2), ratio=mk_clas.dropoutRatio, train=self.train))
		h4 = self.l4(F.dropout(F.relu(h3), ratio=mk_clas.dropoutRatio, train=self.train))
		h5 = self.l5(F.dropout(F.relu(h4), ratio=mk_clas.dropoutRatio, train=self.train))
		y = self.l6(F.dropout(F.relu(h5), ratio=mk_clas.dropoutRatio, train=self.train))
		return y

	def getModelKind(self):
		return "clas"

class NNNN(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_midunits),
			l4=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		pass

	#@jit
	def __call__(self, x):
		h1 = self.l1(x)
		h2 = self.l2(F.relu(h1))
		h3 = self.l3(F.relu(h2))
		y = self.l4(F.relu(h3))
		return y

	def getModelKind(self):
		return "clas"

class NNLNLNLNLNN(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.LSTM(n_midunits, n_midunits),
			l4=L.Linear(n_midunits, n_midunits),
			l5=L.LSTM(n_midunits, n_midunits),
			l6=L.Linear(n_midunits, n_midunits),
			l7=L.LSTM(n_midunits, n_midunits),
			l8=L.Linear(n_midunits, n_midunits),
			l9=L.LSTM(n_midunits, n_midunits),
			l10=L.Linear(n_midunits, n_midunits),
			l11=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		self.l3.reset_state()
		self.l5.reset_state()
		self.l7.reset_state()
		self.l9.reset_state()

	#@jit
	def __call__(self, x):
		h1 = self.l1(x)
		h2 = self.l2(F.dropout(h1, train=self.train))
		h3 = self.l3(F.dropout(h2, train=self.train))
		h4 = self.l4(F.dropout(h3, train=self.train))
		h5 = self.l5(F.dropout(h4, train=self.train))
		h6 = self.l6(F.dropout(h5, train=self.train))
		h7 = self.l7(F.dropout(h6, train=self.train))
		h8 = self.l8(F.dropout(h7, train=self.train))
		h9 = self.l9(F.dropout(h8, train=self.train))
		h10 = self.l10(F.dropout(h9, train=self.train))
		y = self.l11(F.dropout(h10, train=self.train))
		return y

	def getModelKind(self):
		return "lstm"

class NLNLNLNLNN(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.LSTM(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_midunits),
			l4=L.LSTM(n_midunits, n_midunits),
			l5=L.Linear(n_midunits, n_midunits),
			l6=L.LSTM(n_midunits, n_midunits),
			l7=L.Linear(n_midunits, n_midunits),
			l8=L.LSTM(n_midunits, n_midunits),
			l9=L.Linear(n_midunits, n_midunits),
			l10=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		self.l2.reset_state()
		self.l4.reset_state()
		self.l6.reset_state()
		self.l8.reset_state()

	#@jit
	def __call__(self, x):
		h1 = self.l1(x)
		h2 = self.l2(F.dropout(h1, train=self.train))
		h3 = self.l3(F.dropout(h2, train=self.train))
		h4 = self.l4(F.dropout(h3, train=self.train))
		h5 = self.l5(F.dropout(h4, train=self.train))
		h6 = self.l6(F.dropout(h5, train=self.train))
		h7 = self.l7(F.dropout(h6, train=self.train))
		h8 = self.l8(F.dropout(h7, train=self.train))
		h9 = self.l9(F.dropout(h8, train=self.train))
		y = self.l10(F.dropout(h9, train=self.train))
		return y

	def getModelKind(self):
		return "lstm"

class NLNLNLNLN(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.LSTM(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_midunits),
			l4=L.LSTM(n_midunits, n_midunits),
			l5=L.Linear(n_midunits, n_midunits),
			l6=L.LSTM(n_midunits, n_midunits),
			l7=L.Linear(n_midunits, n_midunits),
			l8=L.LSTM(n_midunits, n_midunits),
			l9=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		self.l2.reset_state()
		self.l4.reset_state()
		self.l6.reset_state()
		self.l8.reset_state()

	#@jit
	def __call__(self, x):
		h1 = self.l1(x)
		h2 = self.l2(F.dropout(h1, train=self.train))
		h3 = self.l3(F.dropout(h2, train=self.train))
		h4 = self.l4(F.dropout(h3, train=self.train))
		h5 = self.l5(F.dropout(h4, train=self.train))
		h6 = self.l6(F.dropout(h5, train=self.train))
		h7 = self.l7(F.dropout(h6, train=self.train))
		h8 = self.l8(F.dropout(h7, train=self.train))
		y = self.l9(F.dropout(h8, train=self.train))
		return y

	def getModelKind(self):
		return "lstm"

class NNNLLLNNN(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_midunits),
			l4=L.LSTM(n_midunits, n_midunits),
			l5=L.LSTM(n_midunits, n_midunits),
			l6=L.LSTM(n_midunits, n_midunits),
			l7=L.Linear(n_midunits, n_midunits),
			l8=L.Linear(n_midunits, n_midunits),
			l9=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		self.l4.reset_state()
		self.l5.reset_state()
		self.l6.reset_state()

	#@jit
	def __call__(self, x):
		h1 = self.l1(x)
		h2 = self.l2(F.dropout(h1, train=self.train))
		h3 = self.l3(F.dropout(h2, train=self.train))
		h4 = self.l4(F.dropout(h3, train=self.train))
		h5 = self.l5(F.dropout(h4, train=self.train))
		h6 = self.l6(F.dropout(h5, train=self.train))
		h7 = self.l7(F.dropout(h6, train=self.train))
		h8 = self.l8(F.dropout(h7, train=self.train))
		y = self.l9(F.dropout(h8, train=self.train))
		return y

	def getModelKind(self):
		return "lstm"

class NNLLLLNN(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.LSTM(n_midunits, n_midunits),
			l4=L.LSTM(n_midunits, n_midunits),
			l5=L.LSTM(n_midunits, n_midunits),
			l6=L.LSTM(n_midunits, n_midunits),
			l7=L.Linear(n_midunits, n_midunits),
			l8=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		self.l3.reset_state()
		self.l4.reset_state()
		self.l5.reset_state()
		self.l6.reset_state()

	#@jit
	def __call__(self, x):
		h1 = self.l1(x)
		h2 = self.l2(F.dropout(h1, train=self.train))
		h3 = self.l3(F.dropout(h2, train=self.train))
		h4 = self.l4(F.dropout(h3, train=self.train))
		h5 = self.l5(F.dropout(h4, train=self.train))
		h6 = self.l6(F.dropout(h5, train=self.train))
		h7 = self.l7(F.dropout(h6, train=self.train))
		y = self.l8(F.dropout(h7, train=self.train))
		return y

	def getModelKind(self):
		return "lstm"

class NNLLLNN(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.LSTM(n_midunits, n_midunits),
			l4=L.LSTM(n_midunits, n_midunits),
			l5=L.LSTM(n_midunits, n_midunits),
			l6=L.Linear(n_midunits, n_midunits),
			l7=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		self.l3.reset_state()
		self.l4.reset_state()
		self.l5.reset_state()

	#@jit
	def __call__(self, x):
		h1 = self.l1(x)
		h2 = self.l2(F.dropout(h1, train=self.train))
		h3 = self.l3(F.dropout(h2, train=self.train))
		h4 = self.l4(F.dropout(h3, train=self.train))
		h5 = self.l5(F.dropout(h4, train=self.train))
		h6 = self.l6(F.dropout(h5, train=self.train))
		y = self.l7(F.dropout(h6, train=self.train))
		return y

	def getModelKind(self):
		return "lstm"

class NLLLLN(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.LSTM(n_midunits, n_midunits),
			l3=L.LSTM(n_midunits, n_midunits),
			l4=L.LSTM(n_midunits, n_midunits),
			l5=L.LSTM(n_midunits, n_midunits),
			l6=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		self.l2.reset_state()
		self.l3.reset_state()
		self.l4.reset_state()
		self.l5.reset_state()

	#@jit
	def __call__(self, x):
		h1 = self.l1(x)
		h2 = self.l2(F.dropout(h1, train=self.train))
		h3 = self.l3(F.dropout(h2, train=self.train))
		h4 = self.l4(F.dropout(h3, train=self.train))
		h5 = self.l5(F.dropout(h4, train=self.train))
		y = self.l6(F.dropout(h5, train=self.train))
		return y

	def getModelKind(self):
		return "lstm"

class NLLLN(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.LSTM(n_midunits, n_midunits),
			l3=L.LSTM(n_midunits, n_midunits),
			l4=L.LSTM(n_midunits, n_midunits),
			l5=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		self.l2.reset_state()
		self.l3.reset_state()
		self.l4.reset_state()

	#@jit
	def __call__(self, x):
		h1 = self.l1(x)
		h2 = self.l2(F.dropout(h1, train=self.train))
		h3 = self.l3(F.dropout(h2, train=self.train))
		h4 = self.l4(F.dropout(h3, train=self.train))
		y = self.l5(F.dropout(h4, train=self.train))
		return y

	def getModelKind(self):
		return "lstm"

class NNLLNN(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.LSTM(n_midunits, n_midunits),
			l4=L.LSTM(n_midunits, n_midunits),
			l5=L.Linear(n_midunits, n_midunits),
			l6=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		self.l3.reset_state()
		self.l4.reset_state()

	#@jit
	def __call__(self, x):
		h1 = self.l1(x)
		h2 = self.l2(F.dropout(h1, train=self.train))
		h3 = self.l3(F.dropout(h2, train=self.train))
		h4 = self.l4(F.dropout(h3, train=self.train))
		h5 = self.l5(F.dropout(h4, train=self.train))
		y = self.l6(F.dropout(h5, train=self.train))
		return y

	def getModelKind(self):
		return "lstm"

class NNLNN(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.LSTM(n_midunits, n_midunits),
			l4=L.Linear(n_midunits, n_midunits),
			l5=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		self.l3.reset_state()

	#@jit
	def __call__(self, x):
		h1 = self.l1(x)
		h2 = self.l2(F.dropout(h1, train=self.train))
		h3 = self.l3(F.dropout(h2, train=self.train))
		h4 = self.l4(F.dropout(h3, train=self.train))
		y = self.l5(F.dropout(h4, train=self.train))
		return y

	def getModelKind(self):
		return "lstm"

class NLLN(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.LSTM(n_midunits, n_midunits),
			l3=L.LSTM(n_midunits, n_midunits),
			l4=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		self.l2.reset_state()
		self.l3.reset_state()

	#@jit
	def __call__(self, x):
		h1 = self.l1(x)
		h2 = self.l2(F.dropout(h1, train=self.train))
		h3 = self.l3(F.dropout(h2, train=self.train))
		y = self.l4(F.dropout(h3, train=self.train))
		return y

	def getModelKind(self):
		return "lstm"

class NLN(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.LSTM(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		self.l2.reset_state()

	#@jit
	def __call__(self, x):
		h1 = self.l1(x)
		h2 = self.l2(F.dropout(h1, train=self.train))
		y = self.l3(F.dropout(h2, train=self.train))
		return y

	def getModelKind(self):
		return "lstm"


class OpenDiv4N6L1B2L1(chainer.Chain):
	def __init__(m):
		pass

	def create(m, inCount, unitCount, outCount, gpu, train=True):
		uc1 = unitCount
		uc2 = 7 * unitCount // 10
		uc3 = 5 * unitCount // 10
		uc4 = 3 * unitCount // 10
		uc5 = 2 * unitCount // 10
		super().__init__(
			n1_1=L.Linear(inCount, uc1),
			n1_2=L.Linear(inCount, uc2),
			n1_3=L.Linear(inCount, uc3),
			n1_4=L.Linear(inCount, uc4),

			n2_1=L.Linear(uc1, uc1),
			n2_2=L.Linear(uc2, uc2),
			n2_3=L.Linear(uc3, uc3),
			n2_4=L.Linear(uc4, uc4),

			n3_1=L.Linear(uc1, uc1),
			n3_2=L.Linear(uc2, uc2),
			n3_3=L.Linear(uc3, uc3),
			n3_4=L.Linear(uc4, uc4),

			n4_1=L.Linear(uc1, uc1),
			n4_2=L.Linear(uc2, uc2),
			n4_3=L.Linear(uc3, uc3),
			n4_4=L.Linear(uc4, uc4),

			n5_1=L.Linear(uc1, uc1),
			n5_2=L.Linear(uc2, uc2),
			n5_3=L.Linear(uc3, uc3),
			n5_4=L.Linear(uc4, uc4),

			l1_1=L.LSTM(uc1, uc1),
			l1_2=L.LSTM(uc2, uc2),
			l1_3=L.LSTM(uc3, uc3),
			l1_4=L.LSTM(uc4, uc4),

			b1_1=L.Bilinear(uc1, uc2, uc5),
			b1_2=L.Bilinear(uc3, uc4, uc5),
			b2=L.Bilinear(uc5, uc5, uc5),

			l2=L.LSTM(uc5, uc5),

			n6=L.Linear(uc5, outCount),
		)
		m.inCount = inCount
		m.train = train

	#@jit(nopython=True)
	def reset_state(m):
		m.l1_1.reset_state()
		m.l1_2.reset_state()
		m.l1_3.reset_state()
		m.l1_4.reset_state()
		m.l2.reset_state()

	#@jit(nopython=True)
	def __call__(m, x, volatile):
		tr = m.train

		# 値取得
		x1 = chainer.Variable(x, volatile=volatile)
		x2 = chainer.Variable(x, volatile=volatile)
		x3 = chainer.Variable(x, volatile=volatile)
		x4 = chainer.Variable(x, volatile=volatile)

		# ４種類の次元数に圧縮
		h1 = F.relu(m.n1_1(x1))
		h2 = F.relu(m.n1_2(x2))
		h3 = F.relu(m.n1_3(x3))
		h4 = F.relu(m.n1_4(x4))

		# 圧縮された次元数で４レイヤ分処理
		h1 = F.relu(m.n2_1(h1))
		h2 = F.relu(m.n2_2(h2))
		h3 = F.relu(m.n2_3(h3))
		h4 = F.relu(m.n2_4(h4))

		h1 = F.relu(m.n3_1(h1))
		h2 = F.relu(m.n3_2(h2))
		h3 = F.relu(m.n3_3(h3))
		h4 = F.relu(m.n3_4(h4))

		h1 = F.relu(m.n4_1(h1))
		h2 = F.relu(m.n4_2(h2))
		h3 = F.relu(m.n4_3(h3))
		h4 = F.relu(m.n4_4(h4))

		h1 = F.relu(m.n5_1(h1))
		h2 = F.relu(m.n5_2(h2))
		h3 = F.relu(m.n5_3(h3))
		h4 = F.relu(m.n5_4(h4))

		# LSTMに通す
		h1 = m.l1_1(F.dropout(h1, train=tr))
		h2 = m.l1_2(F.dropout(h2, train=tr))
		h3 = m.l1_3(F.dropout(h3, train=tr))
		h4 = m.l1_4(F.dropout(h4, train=tr))

		# 異なる次元数のレイヤを混ぜて１つにする
		h12 = m.b1_1(h1, h2)
		h34 = m.b1_2(h3, h4)
		h = m.b2(h12, h34)

		# またLSTMに通す
		h = m.l2(F.dropout(h, train=tr))

		# 最後に１レイヤ通す
		h = m.n6(h)

		return h

	def buildMiniBatchData(m, dataset, batchIndices, rnnLen, rnnStep):
		"""学習データセットの指定位置から全ミニバッチデータを作成する"""
		batchSize = batchIndices.shape[0]
		inCount = m.inCount
		data = np.empty(shape=(rnnLen, batchSize, inCount), dtype=np.float32)
		for i in range(rnnLen):
			ofs = i * rnnStep
			for bi in range(batchSize):
				index = batchIndices[bi] + ofs
				data[i,bi,:] = dataset[0, index : index + inCount]
		return data

	def buildSeqData(m, dataset):
		"""データセット全体をシーケンシャルに評価するためのデータ作成"""
		return dataset[0]

	def allocFrame(m):
		"""１回の処理で使用するバッファ確保"""
		return np.empty(shape=(1, m.inCount), dtype=np.float32)

	def copySeqDataToFrame(m, seq, index, frame):
		"""buildSeqData() で確保したデータの指定位置から allocFrame() で確保した領域へコピーする"""
		frame[:] = seq[index : index + m.inCount]

	def getModelKind(m):
		return "lstm"


class OpenDiv4N5L1N1B2L1N1(chainer.Chain):
	def __init__(m):
		pass

	def create(m, inCount, unitCount, outCount, gpu, train=True):
		uc1 = unitCount
		uc2 = 7 * unitCount // 10
		uc3 = 5 * unitCount // 10
		uc4 = 3 * unitCount // 10
		uc5 = 1 * unitCount // 10
		super().__init__(
			n1_1=L.Linear(inCount, uc1),
			n1_2=L.Linear(inCount, uc2),
			n1_3=L.Linear(inCount, uc3),
			n1_4=L.Linear(inCount, uc4),

			n2_1=L.Linear(uc1, uc1),
			n2_2=L.Linear(uc2, uc2),
			n2_3=L.Linear(uc3, uc3),
			n2_4=L.Linear(uc4, uc4),

			n3_1=L.Linear(uc1, uc1),
			n3_2=L.Linear(uc2, uc2),
			n3_3=L.Linear(uc3, uc3),
			n3_4=L.Linear(uc4, uc4),

			n4_1=L.Linear(uc1, uc1),
			n4_2=L.Linear(uc2, uc2),
			n4_3=L.Linear(uc3, uc3),
			n4_4=L.Linear(uc4, uc4),

			n5_1=L.Linear(uc1, uc1),
			n5_2=L.Linear(uc2, uc2),
			n5_3=L.Linear(uc3, uc3),
			n5_4=L.Linear(uc4, uc4),

			l1_1=L.LSTM(uc1, uc1),
			l1_2=L.LSTM(uc2, uc2),
			l1_3=L.LSTM(uc3, uc3),
			l1_4=L.LSTM(uc4, uc4),

			n6_1=L.Linear(uc1, uc5),
			n6_2=L.Linear(uc2, uc5),
			n6_3=L.Linear(uc3, uc5),
			n6_4=L.Linear(uc4, uc5),

			b1_1=L.Bilinear(uc5, uc5, uc5),
			b1_2=L.Bilinear(uc5, uc5, uc5),
			b2=L.Bilinear(uc5, uc5, uc5),

			l2=L.LSTM(uc5, uc5),

			n7=L.Linear(uc5, outCount),
		)
		m.inCount = inCount
		m.train = train

	#@jit(nopython=True)
	def reset_state(m):
		m.l1_1.reset_state()
		m.l1_2.reset_state()
		m.l1_3.reset_state()
		m.l1_4.reset_state()
		m.l2.reset_state()

	#@jit(nopython=True)
	def __call__(m, x, volatile):
		tr = m.train

		# 値取得
		x1 = chainer.Variable(x, volatile=volatile)
		x2 = chainer.Variable(x, volatile=volatile)
		x3 = chainer.Variable(x, volatile=volatile)
		x4 = chainer.Variable(x, volatile=volatile)

		# ４種類の次元数に圧縮
		h1 = F.relu(m.n1_1(x1))
		h2 = F.relu(m.n1_2(x2))
		h3 = F.relu(m.n1_3(x3))
		h4 = F.relu(m.n1_4(x4))

		# 圧縮された次元数で４レイヤ分処理
		h1 = F.relu(m.n2_1(h1))
		h2 = F.relu(m.n2_2(h2))
		h3 = F.relu(m.n2_3(h3))
		h4 = F.relu(m.n2_4(h4))

		h1 = F.relu(m.n3_1(h1))
		h2 = F.relu(m.n3_2(h2))
		h3 = F.relu(m.n3_3(h3))
		h4 = F.relu(m.n3_4(h4))

		h1 = F.relu(m.n4_1(h1))
		h2 = F.relu(m.n4_2(h2))
		h3 = F.relu(m.n4_3(h3))
		h4 = F.relu(m.n4_4(h4))

		h1 = F.relu(m.n5_1(h1))
		h2 = F.relu(m.n5_2(h2))
		h3 = F.relu(m.n5_3(h3))
		h4 = F.relu(m.n5_4(h4))

		# LSTMに通す
		h1 = m.l1_1(F.dropout(h1, train=tr))
		h2 = m.l1_2(F.dropout(h2, train=tr))
		h3 = m.l1_3(F.dropout(h3, train=tr))
		h4 = m.l1_4(F.dropout(h4, train=tr))

		# 同じ次元数に整える
		h1 = F.relu(m.n6_1(h1))
		h2 = F.relu(m.n6_2(h2))
		h3 = F.relu(m.n6_3(h3))
		h4 = F.relu(m.n6_4(h4))

		# バイリニアで合成
		h12 = m.b1_1(h1, h2)
		h34 = m.b1_2(h3, h4)
		h = m.b2(h12, h34)

		# またLSTMに通す
		h = m.l2(F.dropout(h, train=tr))

		# 最後に１レイヤ通す
		h = m.n7(h)

		return h

	def buildMiniBatchData(m, dataset, batchIndices, rnnLen, rnnStep):
		"""学習データセットの指定位置から全ミニバッチデータを作成する"""
		batchSize = batchIndices.shape[0]
		inCount = m.inCount
		data = np.empty(shape=(rnnLen, batchSize, inCount), dtype=np.float32)
		for i in range(rnnLen):
			ofs = i * rnnStep
			for bi in range(batchSize):
				index = batchIndices[bi] + ofs
				data[i,bi,:] = dataset[0, index : index + inCount]
		return data

	def buildSeqData(m, dataset):
		"""データセット全体をシーケンシャルに評価するためのデータ作成"""
		return dataset[0]

	def allocFrame(m):
		"""１回の処理で使用するバッファ確保"""
		return np.empty(shape=(1, m.inCount), dtype=np.float32)

	def copySeqDataToFrame(m, seq, index, frame):
		"""buildSeqData() で確保したデータの指定位置から allocFrame() で確保した領域へコピーする"""
		frame[:] = seq[index : index + m.inCount]

	def getModelKind(m):
		return "lstm"


class AllN6ReluSqueezeL2(chainer.Chain):
	def __init__(m):
		pass

	def create(m, inCount, unitCount, outCount, gpu, train=True):
		uc1 = unitCount
		uc2 = 7 * unitCount // 10
		uc3 = 5 * unitCount // 10
		uc4 = 4 * unitCount // 10
		uc5 = 3 * unitCount // 10
		uc6 = 2 * unitCount // 10
		super().__init__(
			n1_1=L.Linear(inCount, uc1),
			n1_2=L.Linear(inCount, uc1),
			n1_3=L.Linear(inCount, uc1),
			n1_4=L.Linear(inCount, uc1),

			n2_1=L.Linear(uc1, uc2),
			n2_2=L.Linear(uc1, uc2),
			n2_3=L.Linear(uc1, uc2),
			n2_4=L.Linear(uc1, uc2),

			n3_1=L.Linear(uc2, uc3),
			n3_2=L.Linear(uc2, uc3),
			n3_3=L.Linear(uc2, uc3),
			n3_4=L.Linear(uc2, uc3),

			n4_1=L.Linear(uc3, uc4),
			n4_2=L.Linear(uc3, uc4),
			n4_3=L.Linear(uc3, uc4),
			n4_4=L.Linear(uc3, uc4),

			n5_1=L.Linear(uc4, uc5),
			n5_2=L.Linear(uc4, uc5),
			n5_3=L.Linear(uc4, uc5),
			n5_4=L.Linear(uc4, uc5),

			l1_1=L.LSTM(uc5, uc5),
			l1_2=L.LSTM(uc5, uc5),
			l1_3=L.LSTM(uc5, uc5),
			l1_4=L.LSTM(uc5, uc5),

			b1_1=L.Bilinear(uc5, uc5, uc5),
			b1_2=L.Bilinear(uc5, uc5, uc5),
			b2_1=L.Bilinear(uc5, uc5, uc6),

			l2=L.LSTM(uc6, uc6),

			n6_1=L.Linear(uc6, outCount),
		)
		m.train = train

	#@jit(nopython=True)
	def reset_state(m):
		m.l1_1.reset_state()
		m.l1_2.reset_state()
		m.l1_3.reset_state()
		m.l1_4.reset_state()
		m.l2.reset_state()

	#@jit(nopython=True)
	def __call__(m, x, volatile):
		tr = m.train

		# 開始値を処理
		h1 = F.relu(m.n1_1(chainer.Variable(x[0], volatile=volatile)))
		h1 = F.relu(m.n2_1(h1))
		h1 = F.relu(m.n3_1(h1))
		h1 = F.relu(m.n4_1(h1))
		h1 = F.relu(m.n5_1(h1))

		# 高値を処理
		h2 = F.relu(m.n1_2(chainer.Variable(x[1], volatile=volatile)))
		h2 = F.relu(m.n2_2(h2))
		h2 = F.relu(m.n3_2(h2))
		h2 = F.relu(m.n4_2(h2))
		h2 = F.relu(m.n5_2(h2))

		# 低値を処理
		h3 = F.relu(m.n1_3(chainer.Variable(x[2], volatile=volatile)))
		h3 = F.relu(m.n2_3(h3))
		h3 = F.relu(m.n3_3(h3))
		h3 = F.relu(m.n4_3(h3))
		h3 = F.relu(m.n5_3(h3))

		# 終り値を処理
		h4 = F.relu(m.n1_4(chainer.Variable(x[3], volatile=volatile)))
		h4 = F.relu(m.n2_4(h4))
		h4 = F.relu(m.n3_4(h4))
		h4 = F.relu(m.n4_4(h4))
		h4 = F.relu(m.n5_4(h4))

		h1 = m.l1_1(F.dropout(h1, train=tr))
		h2 = m.l1_2(F.dropout(h2, train=tr))
		h3 = m.l1_3(F.dropout(h3, train=tr))
		h4 = m.l1_4(F.dropout(h4, train=tr))

		h1 = F.relu(m.b1_1(h1, h4))
		h2 = F.relu(m.b1_2(h2, h3))
		h1 = F.relu(m.b2_1(h1, h2))

		h1 = m.l2(F.dropout(h1, train=tr))

		h1 = m.n6_1(h1)

		return h1

	def getModelKind(m):
		return "lstm"

class N6ReluSqueezeL2Squeeze(chainer.Chain):
	def __init__(m):
		pass

	def create(m, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units
		n_midunits2 = n_units // 2
		n_midunits3 = n_units // 3
		n_midunits4 = n_units // 4
		super().__init__(
			n1=L.Linear(n_in, n_midunits),
			n2=L.Linear(n_midunits, n_midunits),
			n3=L.Linear(n_midunits, n_midunits2),
			n4=L.Linear(n_midunits2, n_midunits3),
			n5=L.Linear(n_midunits3, n_midunits3),
			l1=L.LSTM(n_midunits3, n_midunits4),
			l2=L.LSTM(n_midunits4, n_midunits4),
			n6=L.Linear(n_midunits4, n_out),
		)
		m.train = train

	#@jit(nopython=True)
	def reset_state(m):
		m.l1.reset_state()
		m.l2.reset_state()

	#@jit(nopython=True)
	def __call__(m, x):
		h = F.relu(m.n1(x))
		h = F.relu(m.n2(h))
		h = F.relu(m.n3(h))
		h = F.relu(m.n4(h))
		h = F.relu(m.n5(h))
		h = m.l1(F.dropout(h, train=m.train))
		h = m.l2(F.dropout(h, train=m.train))
		h = m.n6(h)
		return h

	def getModelKind(m):
		return "lstm"

class L4(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.LSTM(n_in, n_midunits),
			l2=L.LSTM(n_midunits, n_midunits),
			l3=L.LSTM(n_midunits, n_midunits),
			l4=L.LSTM(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		self.l1.reset_state()
		self.l2.reset_state()
		self.l3.reset_state()
		self.l4.reset_state()

	#@jit
	def __call__(self, x):
		h = self.l1(x)
		h = self.l2(F.dropout(h, train=self.train))
		h = self.l3(F.dropout(h, train=self.train))
		h = self.l4(F.dropout(h, train=self.train))
		return h

	def getModelKind(self):
		return "lstm"


class NoAi(chainer.Chain):
	def __init__(m):
		pass
	def create(m, inCount, unitCount, outCount, gpu, train=True):
		super().__init__()
	def reset_state(m):
		pass
	def __call__(m, x, volatile):
		return 0.0
	def getModelKind(m):
		return "noai"
