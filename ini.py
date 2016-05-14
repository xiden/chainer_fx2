# -*- coding: utf-8 -*- 
#!/usr/bin/env python

import win32api as w32a
import os.path as path

class file(object):
	"""Windows用INIファイルアクセスクラス"""
	def __init__(self, iniFile, section):
		self.iniFile = path.normpath(path.join(path.dirname(__file__), path.relpath(iniFile)))
		self.section = section
		return super().__init__()

	def getStr(self, key, default = ""):
		"""INIファイル指定キーから文字列の取得"""
		return w32a.GetProfileVal(self.section, key, default, self.iniFile)

	def getInt(self, key, default = 0):
		"""INIファイル指定キーから整数の取得"""
		return int(w32a.GetProfileVal(self.section, key, str(default), self.iniFile))

	def getFloat(self, key, default = 0.0):
		"""INIファイル指定キーから実数の取得"""
		return float(w32a.GetProfileVal(self.section, key, str(default), self.iniFile))

	def set(self, key, value):
		"""INIファイル指定キーへ値を設定"""
		return w32a.WriteProfileVal(self.section, key, str(value), self.iniFile)

	def getSection(self, section):
		return w32a.GetProfileSection(section, self.iniFile)

	def setSection(self, section, list):
		sl = []
		for i in list:
			sl.append(i + "\0")
		sl.append("\0")
		return w32a.WriteProfileSection(section, "".join(sl), self.iniFile)
