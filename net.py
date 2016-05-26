import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from numba import jit
import mk_clas as c

class N15ReluSqueeze(chainer.Chain):
	def __init__(self):
		pass

	def create(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units
		n_midunits2 = n_units // 2
		n_midunits3 = n_units // 4
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.Linear(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_midunits),
			l4=L.Linear(n_midunits, n_midunits2),
			l5=L.Linear(n_midunits2, n_midunits2),
			l6=L.Linear(n_midunits2, n_midunits2),
			l7=L.Linear(n_midunits2, n_midunits3),
			l8=L.Linear(n_midunits3, n_midunits3),
			l9=L.Linear(n_midunits3, n_midunits3),
			l10=L.Linear(n_midunits3, n_midunits2),
			l11=L.Linear(n_midunits2, n_midunits2),
			l12=L.Linear(n_midunits2, n_midunits),
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
		do = c.dropoutRatio
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
		h2 = self.l2(F.dropout(F.relu(h1), ratio=c.dropoutRatio, train=self.train))
		h3 = self.l3(F.relu(h2))
		h4 = self.l4(F.dropout(F.relu(h3), ratio=c.dropoutRatio, train=self.train))
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
		h2 = self.l2(F.dropout(F.relu(h1), ratio=c.dropoutRatio, train=self.train))
		h3 = self.l3(F.dropout(F.relu(h2), ratio=c.dropoutRatio, train=self.train))
		h4 = self.l4(F.dropout(F.relu(h3), ratio=c.dropoutRatio, train=self.train))
		h5 = self.l5(F.dropout(F.relu(h4), ratio=c.dropoutRatio, train=self.train))
		y = self.l6(F.dropout(F.relu(h5), ratio=c.dropoutRatio, train=self.train))
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
