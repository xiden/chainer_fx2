#include "Socket.h"

#ifdef _MSC_VER
#pragma comment(lib, "WS2_32.lib")
#endif

_JUNK_BEGIN

//! 時間を ms 単位から timeval に変換
timeval SocketRef::MsToTimeval(int64 ms) {
	timeval tv;
	int64 s = ms / 1000;
#if defined __GNUC__
	tv.tv_sec = s;
	tv.tv_usec = ((int64) ms - s * 1000) * 1000;
#elif defined  _MSC_VER
	tv.tv_sec = (long)s;
	tv.tv_usec = (long)(((int64)ms - s * 1000) * 1000);
#endif
	return tv;
}

//! 時間を timeval から ms 単位に変換
int64 SocketRef::TimevalToMs(timeval tv) {
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

//! SocketRef クラスを使用するプログラムの開始時に一度だけ呼び出す
bool SocketRef::Startup() {
#if defined __GNUC__
	return true;
#elif defined  _MSC_VER
	WSADATA wsaData;
	return WSAStartup(MAKEWORD(2, 0), &wsaData) == 0;
#endif
}

//! SocketRef クラスの使用が完全に終わったら呼び出す
bool SocketRef::Cleanup() {
#if defined __GNUC__
	return true;
#elif defined  _MSC_VER
	return WSACleanup() == 0;
#endif
}

//! 127.0.0.1 の様なIPv4アドレス文字列をバイナリに変換する
uint32 SocketRef::IPv4StrToBin(const char* pszIPv4) {
#if defined _MSC_VER && 1800 <= _MSC_VER
	in_addr adr;
	adr.S_un.S_addr = INADDR_NONE;
	InetPtonA(AF_INET, pszIPv4, &adr);
	return (uint32)adr.S_un.S_addr;
#else
	return inet_addr(pszIPv4);
#endif
}

//! 127.0.0.1 の様なIPv4アドレス文字列とポート番号からアドレス構造体を取得する
sockaddr_in SocketRef::IPv4StrToAddress(const char* pszIPv4, int port) {
	sockaddr_in adr;
	memset(&adr, 0, sizeof(adr));
	adr.sin_family = AF_INET;
	adr.sin_port = htons((uint16) port);
#if defined __GNUC__
	adr.sin_addr.s_addr = IPv4StrToBin(pszIPv4);
#elif defined  _MSC_VER
	adr.sin_addr.S_un.S_addr = IPv4StrToBin(pszIPv4);
#endif
	return adr;
}

//! 指定ホスト名、サービス名、ヒントからアドレス情報を取得する
bool SocketRef::Endpoint::Create(const char* pszHost, const char* pszService, const addrinfo* pHint) {
	addrinfo* pRet = NULL;
	if (getaddrinfo(pszHost, pszService, pHint, &pRet) != 0) {
		if (pRet != NULL)
			freeaddrinfo(pRet);
		return false;
	}
	this->Attach(pRet);
	return true;
}

//! 確保したメモリを破棄する
void SocketRef::Endpoint::Delete() {
	if (this->pRoot != NULL) {
		freeaddrinfo(this->pRoot);
		this->pRoot = NULL;
	}
}

//! ホスト名とサービス名を取得する
bool SocketRef::Endpoint::GetNames(std::vector<std::string>* pHosts, std::vector<std::string>* pServices) {
	bool result = true;
	char hbuf[NI_MAXHOST];
	char sbuf[NI_MAXSERV];
	size_t count = this->AddrInfos.size();

	pHosts->resize(count);
	pServices->resize(count);

	for (size_t i = 0; i < count; i++) {
		addrinfo* adrinf = this->AddrInfos[i];
		if (getnameinfo(
			(sockaddr*)adrinf->ai_addr,
			adrinf->ai_addrlen,
			hbuf, sizeof(hbuf),
			sbuf, sizeof(sbuf),
			NI_NUMERICHOST | NI_NUMERICSERV) == 0) {
			pHosts->at(i) = hbuf;
			pServices->at(i) = sbuf;
		} else {
			result = false;
		}
	}

	return result;
}

_JUNK_END
