// FxDnn.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//

#include "stdafx.h"
#include "Socket.h"
#include <vector>
#include <string.h>
#include "FxDnn.h"

using namespace jk;

enum class Cmd : int32_t {
	Uninitialize,
	GetInitiateInfo,
	Initialize,
	SendFxData,
	RecvPredictionData,
	SetYenAveK,
	Log,
};

#pragma pack(push, 1)
//! 送信コマンドパケットイメージ構造体
struct Pkt {
	static constexpr int HeaderSize = 8;

	int32_t Size; //!< 以降に続くパケット内データサイズ、PKT_SIZE_MIN～PKT_SIZE_MAX 範囲外の値が指定されるとパケットとはみなされず応答パケットも戻りません
	Cmd CmdId; //!< 先頭の4バイトはコマンド種類ID
	uint8_t Data[1]; //!< パケットデータプレースホルダ、nSize 分のデータが続く

	Pkt() {
	}
	Pkt(Cmd cmdId) {
		this->Size = 4;
		this->CmdId = cmdId;
	}
	Pkt(Cmd cmdId, int32_t dataSize) {
		this->Size = 4 + dataSize;
		this->CmdId = cmdId;
	}
};
#pragma pack(pop)

Socket g_Sk;
int g_nMinEvalLen;
int g_nPredictionLen;
std::vector<uint8_t> g_Buf;

static Pkt* GrowPktBuf(int size) {
	if ((int)g_Buf.size() < size)
		g_Buf.resize(size);
	auto pPkt = (Pkt*)&g_Buf[0];
	pPkt->Size = size - 4;
	return pPkt;
}

static void GrowBuf(int size) {
	if ((int)g_Buf.size() < size)
		g_Buf.resize(size);
}

//! 指定サイズ分読み込む
static bool RecvToSize(SocketRef sk, void* pBuf, size_t size) {
	while (size) {
		int n = sk.Recv(pBuf, size);
		if (n <= 0)
			return false;
		(uint8_t*&)pBuf += n;
		size -= n;
	}
	return true;
}

//! 指定サイズ分書き込む
static bool SendToSize(SocketRef sk, const void* pBuf, size_t size) {
	while (size) {
		int n = sk.Send(pBuf, size);
		if (n <= 0)
			return false;
		(uint8_t*&)pBuf += n;
		size -= n;
	}
	return true;
}


FXDNN_API int __stdcall FxDnnConnect() {
	Socket::Startup();

	g_Sk.Close();
	if (!g_Sk.Create())
		return -1;

	sockaddr_in adr = Socket::IPv4StrToAddress("127.0.0.1", 4000);
	if (!g_Sk.Connect(&adr))
		return -1;

	return 0;
}

FXDNN_API int __stdcall FxDnnGetInitiateInfo(int& minEvalLen, int& predictionLen) {
	Pkt pkt(Cmd::GetInitiateInfo);
	SendToSize(g_Sk, &pkt, Pkt::HeaderSize);

	int32_t vals[2];
	if (!RecvToSize(g_Sk, vals, sizeof(vals)))
		return -1;

	g_nMinEvalLen = minEvalLen = vals[0];
	g_nPredictionLen = predictionLen = vals[1];

	return 0;
}

FXDNN_API int __stdcall FxDnnInitialize(const float* pOpenData, const float* pHighData, const float* pLowData, const float* pCloseData, const int* pMinData) {
	auto size = Pkt::HeaderSize + g_nMinEvalLen * 4 * 5;
	auto pPkt = GrowPktBuf(size);
	auto copySize = g_nMinEvalLen * 4;
	auto p = &pPkt->Data[0];
	memcpy(p, pOpenData, copySize); p += copySize;
	memcpy(p, pHighData, copySize); p += copySize;
	memcpy(p, pLowData, copySize); p += copySize;
	memcpy(p, pCloseData, copySize); p += copySize;
	memcpy(p, pMinData, copySize); p += copySize;
	pPkt->CmdId = Cmd::Initialize;
	SendToSize(g_Sk, pPkt, size);

	int32_t result;
	if (!RecvToSize(g_Sk, &result, sizeof(result)))
		return -1;

	if (result == 0)
		return -2;

	return 0;
}

FXDNN_API void __stdcall FxDnnUninitialize() {
	Pkt pkt(Cmd::Uninitialize);
	SendToSize(g_Sk, &pkt, Pkt::HeaderSize);

	g_Sk.Shutdown(Socket::Sd::Both);
	g_Sk.Close();
	Socket::Cleanup();
	return;
}

FXDNN_API int __stdcall FxDnnSendFxData(int count, const float* pOpenData, const float* pHighData, const float* pLowData, const float* pCloseData, const int* pMinData) {
	auto size = Pkt::HeaderSize + count * 4 * 5;
	auto pPkt = GrowPktBuf(size);
	auto copySize = count * 4;
	auto p = &pPkt->Data[0];
	memcpy(p, pOpenData, copySize); p += copySize;
	memcpy(p, pHighData, copySize); p += copySize;
	memcpy(p, pLowData, copySize); p += copySize;
	memcpy(p, pCloseData, copySize); p += copySize;
	memcpy(p, pMinData, copySize); p += copySize;
	pPkt->CmdId = Cmd::SendFxData;
	SendToSize(g_Sk, pPkt, size);

	int32_t result;
	if (!RecvToSize(g_Sk, &result, sizeof(result)))
		return -1;

	if (result == 0)
		return -2;

	return 0;
}

FXDNN_API int __stdcall FxDnnRecvPredictionData(float* pPredictionData) {
	Pkt pkt(Cmd::RecvPredictionData);
	SendToSize(g_Sk, &pkt, Pkt::HeaderSize);

	auto size = g_nPredictionLen * 4;
	if (!RecvToSize(g_Sk, pPredictionData, size))
		return -1;

	return 0;
}

FXDNN_API int __stdcall FxDnnSetYenAveK(double k) {
	auto size = Pkt::HeaderSize + sizeof(double);
	auto pPkt = GrowPktBuf(size);
	memcpy(&pPkt->Data[0], &k, sizeof(double));
	pPkt->CmdId = Cmd::SetYenAveK;
	SendToSize(g_Sk, pPkt, size);

	int32_t result;
	if (!RecvToSize(g_Sk, &result, sizeof(result)))
		return -1;

	return 0;
}

FXDNN_API int __stdcall FxDnnLog(int64_t tickTime, int64_t candleTime, int intCount, const int32_t* pIntData, int floatCount, const float* pFloatData) {
	auto size = Pkt::HeaderSize + sizeof(int64_t) + sizeof(int64_t) + sizeof(int32_t) + sizeof(int32_t) * intCount * sizeof(int32_t) + floatCount * sizeof(float);
	auto pPkt = GrowPktBuf(size);
	auto p = &pPkt->Data[0];
	*(int64_t*)p = tickTime; p += sizeof(int64_t);
	*(int64_t*)p = candleTime; p += sizeof(int64_t);
	*(int32_t*)p = intCount; p += sizeof(int32_t);
	*(int32_t*)p = floatCount; p += sizeof(int32_t);
	for (int i = 0; i < intCount; i++) {
		*(int32_t*)p = pIntData[i];
		p += sizeof(int32_t);
	}
	for (int i = 0; i < floatCount; i++) {
		*(float*)p = pFloatData[i];
		p += sizeof(float);
	}
	pPkt->CmdId = Cmd::Log;
	SendToSize(g_Sk, pPkt, size);

	int32_t result;
	if (!RecvToSize(g_Sk, &result, sizeof(result)))
		return -1;

	return 0;
}
