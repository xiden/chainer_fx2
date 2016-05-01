// FxDnn.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//

#include "stdafx.h"
#include "Socket.h"
#include <vector>
#include <string.h>
#include "FxDnn.h"

using namespace Junk;

enum class Cmd : int32 {
	Uninitialize,
	GetInitiateInfo,
	Initialize,
	SendFxData,
	RecvPredictionData,
	SetYenAveK,
};

#pragma pack(push, 1)
//! 送信コマンドパケットイメージ構造体
struct Pkt {
	static constexpr int HeaderSize = 8;

	int32 Size; //!< 以降に続くパケット内データサイズ、PKT_SIZE_MIN〜PKT_SIZE_MAX 範囲外の値が指定されるとパケットとはみなされず応答パケットも戻りません
	Cmd CmdId; //!< 先頭の4バイトはコマンド種類ID
	uint8 Data[1]; //!< パケットデータプレースホルダ、nSize 分のデータが続く

	Pkt() {
	}
	Pkt(Cmd cmdId) {
		this->Size = 4;
		this->CmdId = cmdId;
	}
	Pkt(Cmd cmdId, int32 dataSize) {
		this->Size = 4 + dataSize;
		this->CmdId = cmdId;
	}
};
#pragma pack(pop)

Socket g_Sk;
int g_nMinEvalLen;
int g_nPredictionLen;
std::vector<uint8> g_Buf;

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
		(uint8*&)pBuf += n;
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
		(uint8*&)pBuf += n;
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

	int32 vals[2];
	if (!RecvToSize(g_Sk, vals, sizeof(vals)))
		return -1;

	g_nMinEvalLen = minEvalLen = vals[0];
	g_nPredictionLen = predictionLen = vals[1];

	return 0;
}

FXDNN_API int __stdcall FxDnnInitialize(const float* pYenData, const int* pMinData) {
	auto size = Pkt::HeaderSize + g_nMinEvalLen * 4 * 2;
	auto pPkt = GrowPktBuf(size);
	memcpy(&pPkt->Data[0], pYenData, g_nMinEvalLen * 4);
	memcpy(&pPkt->Data[g_nMinEvalLen * 4], pMinData, g_nMinEvalLen * 4);
	pPkt->CmdId = Cmd::Initialize;
	SendToSize(g_Sk, pPkt, size);

	int32 result;
	if (!RecvToSize(g_Sk, &result, sizeof(result)))
		return -1;

	if (result == 0)
		return -2;

	return 0;
}

FXDNN_API void __stdcall FxDnnUninitialize() {
	Pkt pkt(Cmd::Uninitialize);
	SendToSize(g_Sk, &pkt, Pkt::HeaderSize);

	g_Sk.Shutdown(Socket::SdBoth);
	g_Sk.Close();
	Socket::Cleanup();
	return;
}

FXDNN_API int __stdcall FxDnnSendFxData(int count, const float* pYenData, const int* pMinData) {
	auto size = Pkt::HeaderSize + count * 4 * 2;
	auto pPkt = GrowPktBuf(size);
	memcpy(&pPkt->Data[0], pYenData, count * 4);
	memcpy(&pPkt->Data[count * 4], pMinData, count * 4);
	pPkt->CmdId = Cmd::SendFxData;
	SendToSize(g_Sk, pPkt, size);

	int32 result;
	if (!RecvToSize(g_Sk, &result, sizeof(result)))
		return -1;

	if (result == 0)
		return -2;

	return 0;
}

FXDNN_API int __stdcall FxDnnRecvPredictionData(float* pYenData) {
	Pkt pkt(Cmd::RecvPredictionData);
	SendToSize(g_Sk, &pkt, Pkt::HeaderSize);

	auto size = g_nPredictionLen * 4;
	if (!RecvToSize(g_Sk, pYenData, size))
		return -1;

	return 0;
}

FXDNN_API int __stdcall FxDnnSetYenAveK(double k) {
	auto size = Pkt::HeaderSize + sizeof(double);
	auto pPkt = GrowPktBuf(size);
	memcpy(&pPkt->Data[0], &k, sizeof(double));
	pPkt->CmdId = Cmd::SetYenAveK;
	SendToSize(g_Sk, pPkt, size);

	int32 result;
	if (!RecvToSize(g_Sk, &result, sizeof(result)))
		return -1;

	return 0;
}
