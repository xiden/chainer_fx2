#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import argparse
import sys
import csv
from numba import jit
import random
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda
from chainer import optimizers
from chainer import serializers
import net
import os
import os.path as path
from pathlib import Path
import threading
import win32api
import win32con
import ini
import mk_lstm
import mk_clas

def loadModelAndOptimizer():
	"""モデルとオプティマイザ読み込み"""
	if modelFile and path.isfile(modelFile):
		print('Load model from', modelFile)
		serializers.load_npz(modelFile, dnn.model)
	if stateFile and path.isfile(stateFile):
		print('Load optimizer state from', stateFile)
		serializers.load_npz(stateFile, dnn.optimizer)

def saveModelAndOptimizer():
	"""モデルとオプティマイザ保存"""
	testFileIni.set("curEpoch", curEpoch) # 現在の実施済みエポック数保存
	with modelLock:
		print('save the model')
		serializers.save_npz(modelFile, dnn.model)
		print('save the optimizer')
		serializers.save_npz(stateFile, dnn.optimizer)

# コマンドライン引数解析
parser = argparse.ArgumentParser()
parser.add_argument('iniFileName', help='設定ファイル')
parser.add_argument('--mode', '-m', default='', help='実行モードオーバーライド')
parser.add_argument('--trainDataFile', '-tdf', default='', help='学習用ドル円CSVファイル')
parser.add_argument('--grEnable', '-gr', default='', help='グラフ表示するなら1、それ以外は0')
parser.add_argument('--epoch', '-e', default='', help='目標エポック数、INIファイルの方も書き換える')

args = parser.parse_args()
configFileName = path.join("Configs", args.iniFileName)

# 指定されたINIファイルからパラメータ取得
configIni = ini.file(configFileName, "DEFAULT")

mode = configIni.getStr("mode", "train") # 実行モード
trainDataFile = configIni.getStr("trainDataFile", "") # 学習データファイル
trainDataDummy = configIni.getStr("trainDataDummy", "") # 生成したダミーデータを学習データするするかどうか、 sin/sweep
gpu = configIni.getInt("gpu", "-1") # 使用GPU番号、0以上ならGPU使用
netType = configIni.getStr("netType", "") # ニューラルネットワークモデルタイプ
netInitParamRandom = configIni.getFloat("netInitParamRandom", "0.0") # ニューラルネットワーク重み初期化乱数サイズ
epoch = configIni.getInt("epoch", "1000") # 実行エポック数
numUnits = configIni.getInt("numUnits", "60") # ユニット数
inMA = configIni.getInt("inMA", "5") # 入力値の移動平均サイズ
frameSize = configIni.getInt("frameSize", "300") # 入力フレームサイズ
batchSize = configIni.getInt("batchSize", "20") # バッチ数
batchRandom = configIni.getInt("batchRandom", "1") # バッチ位置をランダムにするかどうか
gradClip = configIni.getFloat("gradClip", "5") # 勾配クリップ
grEnable = configIni.getInt("grEnable", "0") # グラフ表示有効かどうか
evalInterval = configIni.getInt("evalInterval", "20") # 評価（グラフも）間隔エポック数
itrCountInterval = configIni.getInt("itrCountInterval", "10") # イタレーション速度計測間隔
predLen = configIni.getInt("predLen", "1") # 未来予測のサンプル数
predAve = configIni.getInt("predAve", "1") # 未来予測分を平均化するかどうか
optm = configIni.getStr("optm", "Adam") # 勾配計算最適化オブジェクトタイプ
adamAlpha = configIni.getFloat("adamAlpha", "0.001") # Adamアルゴリズムのα値
adaDeltaRho = configIni.getFloat("adaDeltaRho", "0.95") # AdaDeltaアルゴリズムのrho値
adaDeltaEps = configIni.getFloat("adaDeltaEps", "0.000001") # AdaDeltaアルゴリズムのeps値

if len(args.mode) != 0: mode = args.mode # 実行モードオーバーライド
if len(args.trainDataFile) != 0: trainDataFile = args.trainDataFile # 学習データファイルオーバーライド
if len(args.epoch) != 0:
	epoch = int(args.epoch) # エポック数オーバーライド
	configIni.set("epoch", epoch)
if len(args.grEnable) != 0: grEnable = int(args.grEnable) # グラフ表示オーバーライド

# その他グローバル変数初期化
inMA = (inMA // 2) * 2 + 1 # 入力値移動平均サイズを奇数にする
predMeanK = np.ones(predLen) # 未来教師データの平均化係数
#predictionMeanK = np.arange(1.0 / predLen, 1.0, 1.0 / (predLen + 1))
#predictionMeanK *= predictionMeanK
predMeanK = predMeanK / predMeanK.sum()
minPredLen = 0 # ドル円未来予測に必要な最小データ数
minEvalLen = 0 # 学習結果の評価に必要な最小データ数
fxInitialYenDataLen = 0 # 初期化時にMT4から送る必要がある円データ数
fxYenData = np.zeros(1, dtype=np.float32) # MT4から送られる円データ、添え字は fxMinData と同じ
fxMinData = np.zeros(1, dtype=np.int32) # MT4から送られる分データ、添え字は fxYenData と同じ
fxYenDataTrain = None # 学習用の円データ、学習したいデータが更新されたら None 以外になる
fxYenPredictionModel = None # ドル円未来予測用のネットワークモデル
fxRetLen = 0 # クライアントに返す結果データ長
modelLock = threading.Lock() # model を排他処理するためのロック
evalIndex = 0 # 学習中の学習データ評価位置
evalIndexMove = 0 # evalIndex の移動量
requestQuit = False # 学習処理を次回エポック時に終了する要求
quitNow = False # 即座に終了する要求
forceEval = False # 本来一定間隔の評価を強制的に行わせるフラグ
batchRangeStart = 0 # 学習時バッチ処理開始インデックス番号
batchRangeEnd = 0 # 学習時バッチ処理終了インデックス番号
batchIndices = 0 # 学習時バッチ処理インデックス番号配列
batchOffsetInitial = 0 # 学習時バッチ処理の初期オフセット
batchOffset = 0 # 学習時バッチ処理の現在オフセット
n_in = 0 # ニューラルネットの入力次元数
n_out = 0 # ニューラルネットの出力次元数
resultRootDir = "Results" # プロジェクト結果保存用ルートディレクトリ名
resultDir = path.join(resultRootDir, path.splitext(path.basename(configFileName))[0]) # プロジェクト結果保存用ディレクトリ名

# プロジェクト結果保存用ディレクトリ無ければ作成
if not path.isdir(resultRootDir):
	os.mkdir(resultRootDir)
if not path.isdir(resultDir):
	os.mkdir(resultDir)

# ネットワークモデルの種類により大域的に変わる処理の初期化を行う
netClassDef = getattr(net, netType)
model = netClassDef()
if model.getModelKind() == "lstm":
	mk = mk_lstm
elif model.getModelKind() == "clas":
	mk = mk_clas
else:
	print("Unknown model kind", model.getModelKind())
	sys.exit()
dnn = mk.Dnn()
mk.init(configFileName)

# GPU使うならそれ用の数値処理ライブラリ取得
xp = cuda.cupy if gpu >= 0 else np

# 結果残すためのファイル名初期化
# ネットタイプと設定ファイル名によりモデルデータファイル名修飾文字列作成
# これはモデルファイル名に付与される
batchName = "btch" + str(batchSize) + ("rnd" if batchRandom else "")
predName = ("pa" if predAve else "p") + str(predLen)
testFileName = str(netType) + "_" + optm + "_" + batchName + "_u" + str(numUnits) + "f" + str(frameSize) + predName
testFileName = mk.getTestFileName(testFileName)
testFilePath = path.join(resultDir, testFileName)
if trainDataDummy:
	testFileName += "_" + trainDataDummy
modelFile = testFilePath + ".model"
stateFile = testFilePath + ".state"
testFileIni = ini.file(testFilePath + ".ini", "DEFAULT")
curEpoch = testFileIni.getInt("curEpoch", 0) # 現在の実施済みエポック数取得

if mode != "testhr_g":
	# モデル別のグラフ処理初期化
	mk.initGraph(testFileName + ": " + trainDataFile)

	# ネットワークモデルの初期化
	model.create(n_in, numUnits, n_out, gpu, True)
	dnn.model = model
	if netInitParamRandom:
		for param in dnn.model.params():
			data = param.data
			data[:] = np.random.uniform(-netInitParamRandom, netInitParamRandom, data.shape)
	if gpu >= 0:
		cuda.get_device(gpu).use()
		dnn.model.to_gpu()

	# オプティマイザも初期化
	if optm == "Adam":
		dnn.optimizer = optimizers.Adam(adamAlpha)
	elif optm == "AdaDelta":
		dnn.optimizer = optimizers.AdaDelta(adaDeltaRho, adaDeltaEps)
	else:
		print("Unknown optimizer: ", optm)
		sys.exit()
	dnn.optimizer.setup(dnn.model)
	if gradClip != 0.0:
		dnn.optimizer.add_hook(chainer.optimizer.GradientClipping(gradClip))

	# モデルとオプティマイザをロード
	loadModelAndOptimizer()
