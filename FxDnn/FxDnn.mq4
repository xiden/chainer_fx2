//+------------------------------------------------------------------+
//|                                                        FxDnn.mq4 |
//|                                                               TI |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "TI"
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict

#import "FxDnn.dll"
   int FxDnnConnect();
   int FxDnnGetInitiateInfo(int& minEvalLen, int& predictionLen);
   int FxDnnInitialize(float& pYenData[], int& pMinData[]);
   void FxDnnUninitialize();
   int FxDnnSendFxData(int count, float& pYenData[], int& pMinData[]);
   int FxDnnRecvPredictionData(float& pYenData[]);

   int Test1(int a, int b);
   int Test2(int count, int& in[], int& out[]);
#import

bool HasError = false;
int MinEvalLen;
int PredictionLen;


//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
	printf("OnInit()");
	
	// 予測サーバーへ接続
	if(FxDnnConnect() < 0) {
		printf("FxDnnConnect() failed");
		return INIT_FAILED;
	}

	// 初期化に必要な過去データ数とサーバーが送り返す予測データ数の取得	
	if(FxDnnGetInitiateInfo(MinEvalLen, PredictionLen) < 0) {
		printf("FxDnnGetInitiateInfo() failed");
		return INIT_FAILED;
	}
	printf("MinEvalLen=" + MinEvalLen);
	printf("PredictionLen=" + PredictionLen);

	// 過去データを渡して初期化する	
	float yens[];
	int mins[];
	ArraySetAsSeries(yens, false);
	ArraySetAsSeries(mins, false);
	ArrayResize(yens, MinEvalLen);
	ArrayResize(mins, MinEvalLen);
	for(int i = MinEvalLen - 1, j = 0; i != -1; i--, j++) {
		yens[j] = (float)Close[i];
		mins[j] = (int)(Time[i] / 60);
	}
	if(FxDnnInitialize(yens, mins) < 0) {
		printf("FxDnnInitialize() failed");
		return INIT_FAILED;
	}
	
	return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
	// サーバーから切断
	FxDnnUninitialize();
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
	printf("OnTick()");

	// データを送信する
	float yens[];
	int mins[];
	int count = 60;
	ArraySetAsSeries(yens, false);
	ArraySetAsSeries(mins, false);
	ArrayResize(yens, count);
	ArrayResize(mins, count);
	for(int i = count - 1, j = 0; i != -1; i--, j++) {
		yens[j] = (float)Close[i];
		mins[j] = (int)(Time[i] / 60);
	}
	if(FxDnnSendFxData(count, yens, mins) < 0) {
		printf("FxDnnSendFxData() failed");
		return;
	}
	
	// 予測する
	float pred[];
	ArraySetAsSeries(pred, true);
	ArrayResize(pred, PredictionLen);
	if(FxDnnRecvPredictionData(pred) < 0) {
		printf("FxDnnRecvPredictionData() failed");
		return;
	}
}
//+------------------------------------------------------------------+
