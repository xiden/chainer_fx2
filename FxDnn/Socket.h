#pragma once
#ifndef __JUNK_SOCKET_H
#define __JUNK_SOCKET_H

#include "JunkConfig.h"

#include <string.h>
#include <time.h>

#if defined __GNUC__
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <errno.h>
#elif defined  _MSC_VER
#include <WinSock2.h>
#include <WS2tcpip.h>
#endif

#include <vector>
#include <string>

_JUNK_BEGIN

//! ソケットクラス、デストラクタではソケットクローズしない
struct SocketRef {
#ifdef __GNUC__
	typedef int Handle; // ハンドル型
#elif _MSC_VER
	typedef SOCKET Handle; // ハンドル型
#endif

	//! Shutdown() メソッドに渡す列挙値
	enum SdEnum {
		SdRead = 0, //!< 読み込みをシャットダウン
		SdRWrite = 1, //!< 書き込みをシャットダウン
		SdBoth = 2, //!< 読み書き両方をシャットダウン
	};

	//! エンドポイント情報
	struct Endpoint {
		addrinfo* pRoot;
		std::vector<addrinfo*> AddrInfos;

		Endpoint() {
			this->pRoot = NULL;
		}
		~Endpoint() {
			Delete();
		}

		//! 指定ホスト名、サービス名、ヒントからアドレス情報を取得する
		bool Create(const char* pszHost, const char* pszService, const addrinfo* pHint);

		//! 確保したメモリを破棄する
		void Delete();

		//! ホスト名とサービス名を取得する
		bool GetNames(std::vector<std::string>* pHosts, std::vector<std::string>* pServices);

		//! 指定ホスト名、サービス名、ヒントからアドレス情報を取得する
		bool Create(const char* pszHost, const char* pszService, int sockType, int family) {
			struct addrinfo hints;
			memset(&hints, 0, sizeof(hints));
			hints.ai_socktype = sockType;
			hints.ai_family = family;
			return Create(pszHost, pszService, &hints);
		}

		void Attach(addrinfo* pRoot) {
			Delete();
			this->pRoot = pRoot;
			this->AddrInfos.resize(0);
			for (struct addrinfo* adrinf = this->pRoot; adrinf != NULL; adrinf = adrinf->ai_next) {
				this->AddrInfos.push_back(adrinf);
			}
		}

		addrinfo* Detach() {
			addrinfo* p = this->pRoot;
			this->pRoot = NULL;
			return p;
		}

	private:
		// 以下実装してはいけないメソッド
		Endpoint(const Endpoint& c);
		Endpoint& operator=(const Endpoint& c);
	};

	Handle m_hSock; //!< ソケットハンドル

	//! 無効ソケットハンドルの取得
	static _FINLINE Handle InvalidHandle() {
		return -1;
	}

	//! 指定されたハンドルが無効かどうか調べる
	static _FINLINE bool IsInvalidHandle(Handle handle) {
		return handle == InvalidHandle();
	}

	//! 時間を ms 単位から timeval に変換
	static timeval MsToTimeval(int64 ms);

	//! 時間を timeval から ms 単位に変換
	static int64 TimevalToMs(timeval tv);

	//! SocketRef クラスを使用するプログラムの開始時に一度だけ呼び出す
	static bool Startup();

	//! SocketRef クラスの使用が完全に終わったら呼び出す
	static bool Cleanup();

	//! 127.0.0.1 の様なIPv4アドレス文字列をバイナリに変換する
	static uint32 IPv4StrToBin(const char* pszIPv4);

	//! 127.0.0.1 の様なIPv4アドレス文字列とポート番号からアドレス構造体を取得する
	static sockaddr_in IPv4StrToAddress(const char* pszIPv4, int port);

	//! コンストラクタ
	SocketRef() {
		m_hSock = InvalidHandle();
	}
	//! コンストラクタ、ソケットハンドル指定して初期化
	SocketRef(Handle handle) {
		m_hSock = handle;
	}

	//! 自分に無効ソケットハンドルが設定されているかどうか
	_FINLINE bool IsInvalidHandle() {
		return m_hSock == InvalidHandle();
	}

	//! ソケットハンドルを切り離し、所有者を呼び出し元に変更する
	Handle Detach() {
		Handle handle = m_hSock;
		m_hSock = InvalidHandle();
		return handle;
	}

	//! ソケットを作成
	bool Create() {
		m_hSock = socket(AF_INET, SOCK_STREAM, 0);
		return m_hSock != InvalidHandle();
	}

	//! アドレス情報でソケットを作成
	bool Create(const addrinfo* pAddrInfo) {
		m_hSock = socket(pAddrInfo->ai_family, pAddrInfo->ai_socktype, 0);
		return m_hSock != InvalidHandle();
	}

	//! ソケットを閉じる
	void Close() {
		if (m_hSock == InvalidHandle())
			return;
#ifdef __GNUC__
		close(m_hSock);
#elif defined _MSC_VER
		closesocket(m_hSock);
#endif
		m_hSock = InvalidHandle();
	}

	//! 接続
	bool Connect(const sockaddr_in* pAddr) {
		return connect(m_hSock, (const sockaddr*)pAddr, sizeof(sockaddr_in)) == 0;
	}

	//! 接続
	bool Connect(const addrinfo* pAddrInfo) {
		return connect(m_hSock, pAddrInfo->ai_addr, pAddrInfo->ai_addrlen) == 0;
	}

	//! ソケットを指定ポートにバインドする
	bool Bind(int port) {
		sockaddr_in addr;
		socklen_t addrsize = sizeof(addr);

		memset(&addr, 0, sizeof(addr));
		addr.sin_port = htons((uint16)port);
		addr.sin_family = AF_INET;
		addr.sin_addr.s_addr = htonl(INADDR_ANY);

		return bind(m_hSock, (sockaddr*)&addr, addrsize) == 0;
	}

	//! ソケットを指定アドレス情報にバインドする
	bool Bind(const addrinfo* pAddrInfo) {
		return bind(m_hSock, pAddrInfo->ai_addr, pAddrInfo->ai_addrlen) == 0;
	}

	//! リッスン開始
	bool Listen(int backlog) {
		return listen(m_hSock, backlog) == 0;
	}

	//! 接続受付
	Handle Accept(sockaddr_in* pAddr) {
		socklen_t addrsize = sizeof(sockaddr_in);
		return accept(m_hSock, (sockaddr*)pAddr, &addrsize);
	}

	//! 接続受付
	Handle Accept(sockaddr* pFromAddr, socklen_t* pFromLen) {
		return accept(m_hSock, pFromAddr, pFromLen);
	}

	//! ソケットから読み込み
	_FINLINE int Recv(void* pBuf, size_t size) {
#ifdef __GNUC__
		return recv(m_hSock, pBuf, size, 0);
#elif defined _MSC_VER
		return recv(m_hSock, (char*)pBuf, size, 0);
#endif
	}

	//! ソケットから読み込み(UDP用)
	_FINLINE int RecvFrom(void* pBuf, size_t size, sockaddr* pFromAddr, socklen_t* pFromLen) {
#ifdef __GNUC__
		return recvfrom(m_hSock, (char*)pBuf, size, 0, pFromAddr, pFromLen);
#elif defined _MSC_VER
		return recvfrom(m_hSock, (char*)pBuf, size, 0, pFromAddr, pFromLen);
#endif
	}

	//! ソケットへ書き込み
	_FINLINE int Send(const void* pBuf, size_t size) {
#ifdef __GNUC__
		return send(m_hSock, (void*)pBuf, size, 0);
#elif defined _MSC_VER
		return send(m_hSock, (char*)pBuf, size, 0);
#endif
	}

	//! 指定アドレスへ送信(UDP用)
	_FINLINE int SendTo(void* pBuf, size_t size, const addrinfo* pAddrInfo) {
#ifdef __GNUC__
		return sendto(m_hSock, (char*)pBuf, size, 0, (sockaddr*)pAddrInfo->ai_addr, pAddrInfo->ai_addrlen);
#elif defined _MSC_VER
		return sendto(m_hSock, (char*)pBuf, size, 0, pAddrInfo->ai_addr, pAddrInfo->ai_addrlen);
#endif
	}

	//! 指定アドレス情報へ送信(UDP用)
	_FINLINE int SendTo(void* pBuf, size_t size, const sockaddr_in& addr) {
#ifdef __GNUC__
		return sendto(m_hSock, (char*)pBuf, size, 0, (sockaddr*)&addr, sizeof(addr));
#elif defined _MSC_VER
		return sendto(m_hSock, (char*)pBuf, size, 0, (sockaddr*)&addr, sizeof(addr));
#endif
	}

	//! シャットダウン
	bool Shutdown(SdEnum how) {
		return shutdown(m_hSock, (int)how) == 0;
	}

	//! 読み込みタイムアウト設定
	bool SetRecvTimeout(int64 ms) {
		timeval tv = MsToTimeval(ms);
		return setsockopt(m_hSock, SOL_SOCKET, SO_RCVTIMEO, (char*)&tv, sizeof(tv)) == 0;
	}

	//! 読み込みタイムアウト取得
	int64 GetRecvTimeout() {
		timeval tv;
		socklen_t size = sizeof(tv);
		memset(&tv, 0, sizeof(tv));
		if (getsockopt(m_hSock, SOL_SOCKET, SO_RCVTIMEO, (char*)&tv, &size) != 0)
			return -1;
		return TimevalToMs(tv);
	}

	//! 書き込みタイムアウト設定
	bool SetSendTimeout(int64 ms) {
		timeval tv = MsToTimeval(ms);
		return setsockopt(m_hSock, SOL_SOCKET, SO_SNDTIMEO, (char*)&tv, sizeof(tv)) == 0;
	}

	//! 書き込みタイムアウト取得
	int64 GetSendTimeout() {
		timeval tv;
		socklen_t size = sizeof(tv);
		memset(&tv, 0, sizeof(tv));
		if (getsockopt(m_hSock, SOL_SOCKET, SO_SNDTIMEO, (char*)&tv, &size) != 0)
			return -1;
		return TimevalToMs(tv);
	}

	//! Nagleアルゴリズムの無効化の設定
	bool SetNoDelay(int flag) {
		return setsockopt(m_hSock, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(flag)) == 0;
	}

	//! Nagleアルゴリズムの無効化の取得
	int GetNoDelay() {
		int flag = 0;
		socklen_t size = sizeof(flag);
		if (getsockopt(m_hSock, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, &size) != 0)
			return -1;
		return flag;
	}

	//! ブロッキングモードの設定
	bool SetBlockingMode(bool blocking) {
#ifdef __GNUC__
		int val = blocking ? 0 : 1;
		return ioctl(m_hSock, FIONBIO, &val) == 0;
#elif defined _MSC_VER
		u_long val = blocking ? 0 : 1;
		return ioctlsocket(m_hSock, FIONBIO, &val) == 0;
#endif
	}

	//! ノンブロッキングモードで Recv() または RecvFrom() 呼び出し時に負数が返った場合にエラーでは無くデータ未取得なのかどうか調べる
	_FINLINE bool NoRecvError() {
#ifdef __GNUC__
		return errno == EAGAIN;
#elif defined _MSC_VER
		return ::WSAGetLastError() == WSAEWOULDBLOCK;
#endif
	}
};

//! ソケットクラス
struct Socket: public SocketRef {
	//! コンストラクタ
	Socket() {
	}
	//! コンストラクタ、ソケットハンドル指定して初期化
	Socket(Handle handle) :
			SocketRef(handle) {
	}
	//! デストラクタ
	Socket(const Socket& c) :
			SocketRef(c.m_hSock) {
	}
	~Socket() {
		if (m_hSock == InvalidHandle())
			return;
#ifdef __GNUC__
		close(m_hSock);
#elif defined _MSC_VER
		closesocket(m_hSock);
#endif
	}
};

_JUNK_END

#endif
