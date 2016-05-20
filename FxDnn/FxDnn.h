#pragma once
#ifndef __FXDNN_H__
#define __FXDNN_H__
#include <stdint.h>

#ifdef __FXDNN_DLL__
#define FXDNN_API extern "C" __declspec(dllexport)
#else
#define FXDNN_API extern "C" __declspec(dllimport)
#endif

FXDNN_API int __stdcall FxDnnConnect();
FXDNN_API int __stdcall FxDnnGetInitiateInfo(int& minEvalLen, int& predictionLen);
FXDNN_API int __stdcall FxDnnInitialize(const float* pOpenData, const float* pHighData, const float* pLowData, const float* pCloseData, const int* pMinData);
FXDNN_API void __stdcall FxDnnUninitialize();
FXDNN_API int __stdcall FxDnnSendFxData(int count, const float* pOpenData, const float* pHighData, const float* pLowData, const float* pCloseData, const int* pMinData);
FXDNN_API int __stdcall FxDnnRecvPredictionData(float* pPredictionData);
FXDNN_API int __stdcall FxDnnSetYenAveK(double k);
FXDNN_API int __stdcall FxDnnLog(int64_t tickTime, int64_t candleTime, int intCount, const int32_t* pIntData, int floatCount, const float* pFloatData);

#endif
