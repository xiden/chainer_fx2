// FxDnnTest.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include "../FxDnn/FxDnn.h"
#include <vector>
#include <valarray>
#include <random>
#include <iostream>
#include <math.h>

#pragma comment(lib, "../FxDnn/Release/FxDnn.lib")


int main()
{
	std::mt19937 mt(0);

	if (FxDnnConnect() < 0)
		return -1;

	int minEvalLen;
	int predictionLen;
	if (FxDnnGetInitiateInfo(minEvalLen, predictionLen) < 0)
		return -1;

	auto minute = 1;
	std::valarray<float> yens(minEvalLen);
	std::valarray<int> mins(minEvalLen);
	for (int i = 0; i < minEvalLen; i++) {
		yens[i] = 110.0f + (double)mt() / mt.max();
		mins[i] = minute;
		minute++;
	}
	if (FxDnnInitialize(&yens[0], &mins[0]) < 0)
		return -1;

	auto rad = 0.0;
	for (;;) {
		yens.resize(60);
		mins.resize(60);
		for (int i = 0; i < 60; i++) {
			yens[i] = 110.0 + sin(rad) + (double)mt() / mt.max();
			mins[i] = minute;
			rad += 0.1;
			minute++;
		}
		if (FxDnnSendFxData(60, &yens[0], &mins[0]) < 0)
			return -1;

		yens.resize(predictionLen);
		yens = 0.0f;
		if (FxDnnRecvPredictionData(&yens[0]) < 0)
			return -1;
		for (auto val : yens) {
			std::cout << val << std::endl;
		}
	}

	FxDnnUninitialize();

    return 0;
}

