﻿#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import sys
import os
import os.path as path
import pathlib
import argparse
import threading
import win32api
import win32con
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda
from chainer import optimizers
from chainer import serializers

import net
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

def findDataset(symbol):
	"""指定されたデータセットに合致するファイル名を取得する.
	Args:
		symbol: 0_5000_10000 の様な形式の文字列、0がデータセット番号(負数なら最古データ)、5000が最小データ数、10000が最大データ数.
	"""

	number, minDataCount, maxDataCount = symbol.split("_")
	number = int(number)
	minDataCount = int(minDataCount)
	maxDataCount = int(maxDataCount)

	p = pathlib.Path("Datasets")
	l = list(p.glob("[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9]_[0-9]*.csv"))
	datasets = []
	for pl in l:
		fn = path.splitext(pl.name)[0]
		fields = fn.split("_")
		count = int(fields[2])
		if count < minDataCount or maxDataCount < count:
			continue
		datasets.append([fields[0] + "_" + fields[1], fields[2]])

	datasets.sort(key=lambda x:x[0], reverse=True)
	d = datasets[number]
	return d[0] + "_" + d[1] + ".csv"



# コマンドライン引数解析
parser = argparse.ArgumentParser()
parser.add_argument('iniFileName', help='設定ファイル')
parser.add_argument('--mode', '-m', default='', help='実行モードオーバーライド')
parser.add_argument('--grEnable', '-g', default='', help='グラフ表示するなら1、それ以外は0')
parser.add_argument('--epoch', '-e', default='', help='目標エポック数、INIファイルの方も書き換える')
parser.add_argument('--train', '-t', default='', help='追加学習エポック数、INIファイルの目標エポック数が書き換わる')
parser.add_argument('--dataset', '-d', default='', help='データセット選択INIファイルも書き換わる、 0_5000_10000 の様に指定する、0が番号(負数なら最古データ)、5000が最小データ数、10000が最大データ数')
parser.add_argument('--nettype', '-n', default='', help='ニューラルネットワークタイプ名、INIファイルも書き換わる')
parser.add_argument('--backupEpoch', '-b', default='', help='学習完了時エポックデータをバックアップするかどうか、INIファイルも書き換わる')

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
frameSize = configIni.getInt("frameSize", "300") # 入力分足数
batchSize = configIni.getInt("batchSize", "20") # バッチ数
batchRandom = configIni.getInt("batchRandom", "1") # バッチ位置をランダムにするかどうか
gradClip = configIni.getFloat("gradClip", "5") # 勾配クリップ
grEnable = configIni.getInt("grEnable", "0") # グラフ表示有効かどうか
evalInterval = configIni.getInt("evalInterval", "20") # 評価（グラフも）間隔エポック数
itrCountInterval = configIni.getInt("itrCountInterval", "10") # イタレーション速度計測間隔
predLen = configIni.getInt("predLen", "1") # 未来予測の分足数
predAve = configIni.getInt("predAve", "1") # 未来予測分を平均化するかどうか
optm = configIni.getStr("optm", "Adam") # 勾配計算最適化オブジェクトタイプ
adamAlpha = configIni.getFloat("adamAlpha", "0.001") # Adamアルゴリズムのα値
adaDeltaRho = configIni.getFloat("adaDeltaRho", "0.95") # AdaDeltaアルゴリズムのrho値
adaDeltaEps = configIni.getFloat("adaDeltaEps", "0.000001") # AdaDeltaアルゴリズムのeps値
serverTrainCount = configIni.getInt("serverTrainCount", "0") # サーバーとして動作中に最新データ側から過去に向かって学習させる回数、全ミニバッチを接触させた状態で学習させる
backupEpoch = configIni.getInt("backupEpoch", "1") # 学習完了時エポックデータをバックアップするかどうか

# コマンドライン引数によるINI設定のオーバーライド
if len(args.mode) != 0:
	mode = args.mode # 実行モードオーバーライド
	configIni.set("mode", "mode")
if len(args.dataset) != 0:
	trainDataFile = findDataset(args.dataset)
	configIni.set("trainDataFile", trainDataFile)
if len(args.epoch) != 0:
	epoch = int(args.epoch) # エポック数オーバーライド
	configIni.set("epoch", epoch)
if len(args.nettype) != 0:
	netType = args.nettype # ネットワークタイプオーバーライド
	configIni.set("netType", netType)
if len(args.grEnable) != 0:
	grEnable = int(args.grEnable) # グラフ表示オーバーライド
	configIni.set("grEnable", grEnable)
if len(args.backupEpoch) != 0:
	backupEpoch = int(args.backupEpoch) # バックアップ処理オーバーライド
	configIni.set("backupEpoch", backupEpoch)

# その他グローバル変数初期化
inMA = (inMA // 2) * 2 + 1 # 入力値移動平均サイズを奇数にする
predMeanK = np.ones(predLen) # 未来教師データの平均化係数
#predictionMeanK = np.arange(1.0 / predLen, 1.0, 1.0 / (predLen + 1))
#predictionMeanK *= predictionMeanK
predMeanK = predMeanK / predMeanK.sum()
predMeanK = predMeanK.reshape((predLen, 1))
minPredLen = 0 # ドル円未来予測に必要な最小分足データ数、実際に必要なデータ数は4倍となる
minEvalLen = 0 # 学習結果の評価に必要な最小分足データ数、実際に必要なデータ数は4倍となる
fxRequiredYenDataLen = 0 # MT4から送る必要がある分足データ数、実際に必要なデータ数は4倍となる
fxYenData = np.zeros(1, dtype=np.float32) # MT4から送られる分足データ、開始値、高値、低値、終値の繰り返し
fxMinData = np.zeros(1, dtype=np.int32) # MT4から送られる分足時間データ、添え字は fxYenData の1/4
fxYenDataTrain = None # 学習用の分足データ、学習したいデータが更新されたら None 以外になる
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
resultConfigDir = path.join(resultRootDir, path.splitext(path.basename(configFileName))[0]) # 設定ファイル別の結果保存ディレクトリ名
resultTestDir = None # 試験設定別結果保存ディレクトリ名
resultHrDir = None # 的中率結果保存ディレクトリ名
sharedDataset = None # アプリ全体で共有する学習用データセット

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

# 結果残すための試験ファイル名初期化
# 試験設定をファイル名に付与する
testFileName = str(netType) # ネットモデル名
testFileName += "_" + optm  # オプティマイザ名
testFileName += "_" + "btch" + str(batchSize) + ("rnd" if batchRandom else "") # バッチ数
testFileName += "_u" + str(numUnits) # ユニット数
testFileName += "f" + str(frameSize) # フレームサイズ
testFileName += ("pa" if predAve else "p") + str(predLen) # 未来予測オフセット値
testFileName = mk.getTestFileName(testFileName) # ネットモデル用のポストフィックス付けて完成
if trainDataDummy:
	testFileName += "_" + trainDataDummy # ダミー学習データ使ったならその種類も付与
resultTestDir = path.join(resultConfigDir, testFileName) # 試験設定別結果保存ディレクトリ名確定
resultHrDir = path.join(resultTestDir, "hr") # 的中率結果保存ディレクトリ名確定

# 結果保存用ディレクトリ無ければ作成
if not path.isdir(resultRootDir):
	os.mkdir(resultRootDir)
if not path.isdir(resultConfigDir):
	os.mkdir(resultConfigDir)
if not path.isdir(resultTestDir):
	os.mkdir(resultTestDir)
if not path.isdir(resultHrDir):
	os.mkdir(resultHrDir)

testFilePath = path.join(resultTestDir, "test")
modelFile = testFilePath + ".model"
stateFile = testFilePath + ".state"
testFileIni = ini.file(testFilePath + ".ini", "DEFAULT")
curEpoch = testFileIni.getInt("curEpoch", 0) # 現在の実施済みエポック数取得
if len(args.train) != 0:
	# 追加学習エポック数が指定されていたら加算しておく
	train = int(args.train)
	epoch = curEpoch + train
	configIni.set("epoch", epoch)


# GPU使うならそれ用の数値処理ライブラリ取得
xp = cuda.cupy if gpu >= 0 else np

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
