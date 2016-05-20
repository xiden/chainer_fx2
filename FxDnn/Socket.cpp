#include "Socket.h"

#ifdef _MSC_VER
#pragma comment(lib, "WS2_32.lib")
#endif

_JUNK_BEGIN

//! 時間を ms 単位から timeval に変換
timeval SocketRef::MsToTimeval(int64_t ms) {
	timeval tv;
	int64_t s = ms / 1000;
#if defined __GNUC__
	tv.tv_sec = s;
	tv.tv_usec = ((int64_t)ms - s * 1000) * 1000;
#elif defined  _MSC_VER
	tv.tv_sec = (long)s;
	tv.tv_usec = (long)(((int64_t)ms - s * 1000) * 1000);
#endif
	return tv;
}

//! 時間を timeval から ms 単位に変換
int64_t SocketRef::TimevalToMs(timeval tv) {
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
uint32_t SocketRef::IPv4StrToBin(const char* pszIPv4) {
#if defined _MSC_VER && 1800 <= _MSC_VER
	in_addr adr;
	adr.S_un.S_addr = INADDR_NONE;
	InetPtonA(AF_INET, pszIPv4, &adr);
	return (uint32_t)adr.S_un.S_addr;
#else
	return inet_addr(pszIPv4);
#endif
}

//! 127.0.0.1 の様なIPv4アドレス文字列とポート番号からアドレス構造体を取得する
sockaddr_in SocketRef::IPv4StrToAddress(const char* pszIPv4, int port) {
	sockaddr_in adr;
	memset(&adr, 0, sizeof(adr));
	adr.sin_family = AF_INET;
	adr.sin_port = htons((uint16_t)port);
#if defined __GNUC__
	adr.sin_addr.s_addr = IPv4StrToBin(pszIPv4);
#elif defined  _MSC_VER
	adr.sin_addr.S_un.S_addr = IPv4StrToBin(pszIPv4);
#endif
	return adr;
}

//! 指定ホスト名、サービス名、ヒントからアドレス情報を取得し保持する
bool SocketRef::Endpoint::Create(
	const char* pszHost, //!< [in,optional] ホスト名、IPv4とIPv6の文字列もいける、サーバーの場合には NULL を指定すると INADDR_ANY (0.0.0.0), IN6ADDR_ANY_INIT (::) 扱いになる、ちなみにUDPのブロードキャストは 192.168.1.255 みたいな感じ
	const char* pszService, //!< [in,optional] サービス名(httpなど)またはポート番号
	const addrinfo* pHint //!< [in,optional] ソケットタイプを決めるヒント情報、ai_addrlen, ai_canonname, ai_addr, ai_next は0にしておくこと
) {
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

//! ホスト名とサービス名を std::vector に取得する
bool SocketRef::Endpoint::GetNames(
	std::vector<std::string>& hosts, //! [in,out] ホスト名配列が返る
	std::vector<std::string>& services //! [in,out] サービス名配列が返る
) {
	bool result = true;
	char hbuf[NI_MAXHOST];
	char sbuf[NI_MAXSERV];
	size_t count = this->AddrInfos.size();

	hosts.resize(count);
	services.resize(count);

#if defined _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4267)
#endif

	for (size_t i = 0; i < count; i++) {
		addrinfo* adrinf = this->AddrInfos[i];
		if (getnameinfo(
			(sockaddr*)adrinf->ai_addr,
			adrinf->ai_addrlen,
			hbuf, sizeof(hbuf),
			sbuf, sizeof(sbuf),
			NI_NUMERICHOST | NI_NUMERICSERV) == 0) {
			hosts[i] = hbuf;
			services[i] = sbuf;
		} else {
			result = false;
		}
	}

#if defined _MSC_VER
#pragma warning(pop)
#endif

	return result;
}

//! ソケットアドレスから名前情報を取得する
bool SocketRef::GetName(
	const sockaddr* pAddr, //!< [in] ソケットアドレス
	socklen_t addrLen, //!< [in] pAddr のサイズ(bytes)
	std::string* pHost, //!< [out] ノード名が返る
	std::string* pService //!< [out] サービス名が返る
) {
	char hbuf[NI_MAXHOST];
	char sbuf[NI_MAXSERV];
	if (getnameinfo(
		pAddr,
		addrLen,
		hbuf, sizeof(hbuf),
		sbuf, sizeof(sbuf),
		NI_NUMERICHOST | NI_NUMERICSERV) != 0) {
		return false;
	}
	*pHost = hbuf;
	*pService = sbuf;
	return true;
}

_JUNK_END
