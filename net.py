import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

class NNLNLNLNLNN(chainer.Chain):
	def __init__(self, n_in, n_units, n_out, gpu, train=True):
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
	def __init__(self, n_in, n_units, n_out, gpu, train=True):
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
	def __init__(self, n_in, n_units, n_out, gpu, train=True):
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
	def __init__(self, n_in, n_units, n_out, gpu, train=True):
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
	def __init__(self, n_in, n_units, n_out, gpu, train=True):
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
	def __init__(self, n_in, n_units, n_out, gpu, train=True):
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
	def __init__(self, n_in, n_units, n_out, gpu, train=True):
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
	def __init__(self, n_in, n_units, n_out, gpu, train=True):
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
	def __init__(self, n_in, n_units, n_out, gpu, train=True):
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
	def __init__(self, n_in, n_units, n_out, gpu, train=True):
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
	def __init__(self, n_in, n_units, n_out, gpu, train=True):
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

	def __call__(self, x):
		h1 = self.l1(x)
		h2 = self.l2(F.dropout(h1, train=self.train))
		h3 = self.l3(F.dropout(h2, train=self.train))
		y = self.l4(F.dropout(h3, train=self.train))
		return y

	def getModelKind(self):
		return "lstm"

class NLN(chainer.Chain):
	def __init__(self, n_in, n_units, n_out, gpu, train=True):
		n_midunits = n_units // 1
		super().__init__(
			l1=L.Linear(n_in, n_midunits),
			l2=L.LSTM(n_midunits, n_midunits),
			l3=L.Linear(n_midunits, n_out),
		)
		self.train = train

	def reset_state(self):
		self.l2.reset_state()

	def __call__(self, x):
		h1 = self.l1(x)
		h2 = self.l2(F.dropout(h1, train=self.train))
		y = self.l3(F.dropout(h2, train=self.train))
		return y

	def getModelKind(self):
		return "lstm"

class LLLL(chainer.Chain):
	def __init__(self, n_in, n_units, n_out, gpu, train=True):
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

	def __call__(self, x):
		h1 = self.l1(x)
		h2 = self.l2(F.dropout(h1, train=self.train))
		h3 = self.l3(F.dropout(h2, train=self.train))
		y = self.l4(F.dropout(h3, train=self.train))
		return y

	def getModelKind(self):
		return "lstm"
