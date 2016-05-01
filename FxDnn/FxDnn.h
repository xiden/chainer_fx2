#pragma once
#ifndef __FXDNN_H__
#define __FXDNN_H__

#ifdef __FXDNN_DLL__
#define FXDNN_API extern "C" __declspec(dllexport)
#else
#define FXDNN_API extern "C" __declspec(dllimport)
#endif

FXDNN_API int __stdcall FxDnnConnect();
FXDNN_API int __stdcall FxDnnGetInitiateInfo(int& minEvalLen, int& predictionLen);
FXDNN_API int __stdcall FxDnnInitialize(const float* pYenData, const int* pMinData);
FXDNN_API void __stdcall FxDnnUninitialize();
FXDNN_API int __stdcall FxDnnSendFxData(int count, const float* pYenData, const int* pMinData);
FXDNN_API int __stdcall FxDnnRecvPredictionData(float* pYenData);
FXDNN_API int __stdcall FxDnnSetYenAveK(double k);

#endif
