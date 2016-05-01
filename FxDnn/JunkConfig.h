#pragma once
#ifndef __JUNK_JUNKCONFIG_H
#define __JUNK_JUNKCONFIG_H

#include <assert.h>
#include <ctype.h>
#include <stddef.h>

// ネームスペース用マクロ
#define _JUNK_BEGIN namespace Junk {
#define _JUNK_END }
#define _JUNK_USING using namespace Junk;

// DLLエクスポート、インポート設定用マクロ
//  _JUNK_EXPORTS が定義されている場合はDLLエクスポート用コンパイル
//  _JUNK_IMPORTS が定義されている場合はDLLインポート用コンパイル
// になります
#ifdef _MSC_VER
#ifdef _JUNK_EXPORTS
#define JUNKAPI extern "C" __declspec(dllexport)
#define JUNKCALL __stdcall
#elif _JUNK_IMPORTS
#define JUNKAPI extern "C" __declspec(import)
#define JUNKCALL __stdcall
#else
#define JUNKAPI
#define JUNKCALL
#endif
#endif

// 強制インライン展開マクロ
#if defined __GNUC__
#define _FINLINE inline __attribute__((always_inline))
#elif defined  _MSC_VER
#define _FINLINE inline __forceinline
#endif

_JUNK_BEGIN

// ビット数を明確にした整数型宣言
typedef char int8;
typedef short int16;
typedef int int32;
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
#if defined __GNUC__
typedef signed long long int64;
typedef unsigned long long uint64;
#elif defined  _MSC_VER
typedef __int64 int64;
typedef unsigned __int64 uint64;
#endif

// ポインタサイズと同じビット数になる整数型宣言
#if defined __GNUC__
#ifdef __x86_64__
typedef long long IntPtr;
typedef unsigned long long UIntPtr;
#else
typedef int IntPtr;
typedef unsigned int UIntPtr;
#endif
#elif defined  _MSC_VER
#ifdef _WIN64
typedef __int64 IntPtr;
typedef unsigned __int64 UIntPtr;
#else
typedef int IntPtr;
typedef unsigned int UIntPtr;
#endif
#endif

// 整数型で bool の代わり
typedef IntPtr ibool;

_JUNK_END

#endif
