#pragma once
#ifndef __JUNK_JUNKCONFIG_H__
#define __JUNK_JUNKCONFIG_H__

#include <assert.h>
#include <ctype.h>
#include <stddef.h>
#include <stdint.h>

// ネームスペース用マクロ
#define _JUNK_BEGIN namespace jk {
#define _JUNK_END }
#define _JUNK_USING using namespace jk;

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

//// ビット数を明確にした整数型宣言
//#ifndef _STDINT
//typedef char int8_t;
//typedef short int16_t;
//typedef int int32_t;
//typedef unsigned char uint8_t;
//typedef unsigned short uint16_t;
//typedef unsigned int uint32_t;
//#if defined __GNUC__
//typedef signed long long int64_t;
//typedef unsigned long long uint64_t;
//#elif defined  _MSC_VER
//typedef __int64 int64_t;
//typedef unsigned __int64 uint64_t;
//#endif
//#endif
//
//// ポインタサイズと同じビット数になる整数型宣言
//#if defined __GNUC__
//#ifdef __x86_64__
//typedef long long intptr_t;
//typedef unsigned long long UIntPtr;
//#else
//typedef int intptr_t;
//typedef unsigned int UIntPtr;
//#endif
//#elif defined  _MSC_VER
//#ifdef _WIN64
//typedef __int64 intptr_t;
//typedef unsigned __int64 UIntPtr;
//#else
//typedef int intptr_t;
//typedef unsigned int UIntPtr;
//#endif
//#endif

_JUNK_BEGIN
//! bool 型を使うと遅いことがあるのでアーキテクチャのbit数に合わせたもの
typedef intptr_t ibool;
_JUNK_END

#endif
