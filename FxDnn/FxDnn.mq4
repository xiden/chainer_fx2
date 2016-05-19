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
	int FxDnnInitialize(float& pOpenData[], float& pHighData[], float& pLowData[], float& pCloseData[], int& pMinData[]);
	void FxDnnUninitialize();
	int FxDnnSendFxData(int count, float& pOpenData[], float& pHighData[], float& pLowData[], float& pCloseData[], int& pMinData[]);
	int FxDnnRecvPredictionData(float& pPredictionData[]);
	int FxDnnSetYenAveK(double k);
	int FxDnnLog(long tickTime, long candleTime, int count, float& pData[]);
#import

input double OrderDeltaPipsAi = 1; // 注文判定PIPS AI予想値
input double OrderDeltaPipsMa = 0.0; // 注文判定PIPS 移動平均d
input double OrderDelta2PipsMa = 0.01; // 注文判定PIPS 移動平均d2
input double MinProfYen = 500.0; // 最小許容利益円
input double CloseReachRate = 100.0; // 決済判定用予測到達率%
input double CloseDeltaPipsAi = 0.0; // 最小許容利益時の決済判定反転PIPS AI予想値
input double CloseDeltaPipsMa = 0.0; // 最小許容利益時の決済判定反転PIPS 移動平均d
input double CloseDelta2PipsMa = 0.25; // 最小許容利益時の決済判定反転PIPS 移動平均d2
input double ImCloseDeltaPipsAi = 0.0; // 決済判定反転PIPS AI予想値
input double ImCloseDeltaPipsMa = 0.0; // 決済判定反転PIPS 移動平均d
input double ImCloseDelta2PipsMa = 1.0; // 決済判定反転PIPS 移動平均d2
input double MaxLossYen = 5000.0; // 最大許容損失円
input double MaxProfYen = 3000.0; // 最大利益円(この値に達したら即決済)
input double OrderVolume = 1.0; // 注文数量

enum OrderCloseMode {
	OCM_LossCut, // 損切り
	OCM_Satisfied, // 満足した
	OCM_Reached, // 当初の予測に到達した
	OCM_Prevent, // 予防
};

bool Initialized = false; // サーバーと接続できて初期化済みかどうか
int MinEvalLen; // サーバーでの評価に必要な最小データ数
int PredictionLen; // サーバーから返ってくる予測値データ数
datetime LastMinute; // 最後に注文した時間(分)
double DeltaPipsAi = 0.0; // AI予測変化量PIPS
double DeltaPipsMa = 0.0; // 移動平均での変化量PIPS
double Delta2PipsMa = 0.0; // 移動平均での変化量PIPS
double PredStartPip = 0.0; // 注文時のAsk*100
double PredEndPip = 0.0; // 注文時の予測PIPS

int BuyCount = 0; // 買い注文回数
int BuySettlLossCount = 0; // 買い注文決済時の損失回数
int BuySettlProfCount = 0; // 買い注文決済時の利益回数
int BuyLossCutCount = 0; // 買い注文時の損切り回数
int SellCount = 0; // 売り注文回数
int SellSettlLossCount = 0; // 売り注文決済時の損失回数
int SellSettlProfCount = 0; // 売り注文決済時の利益回数
int SellLossCutCount = 0; // 売り注文時の損切り回数

bool ComparePredDeltaOr(int compareType, double ai, double ma, double ma2); // サーバー予測値との比較、どれか一つでも条件を満たせば true が返る
bool ComparePredDeltaAnd(int compareType, double ai, double ma, double ma2); // サーバー予測値との比較、全てが条件を満たせば true が返る
bool MyOrderClose(OrderCloseMode mode); // 注文を決済する
string GetOrderCloseModeName(OrderCloseMode mode); // 注文決済モード名の取得
double CalcPredReachRate(); // 現在の予測到達率%を計算する


//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
	//printf("OnInit()");
	long
	
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
	printf("MinEvalLen=%d", MinEvalLen);
	printf("RetValLen=%d", PredictionLen);
	
	PredStartPip = PredEndPip = 0.0;

	// 予測値の合成係数を設定する
	//FxDnnSetYenAveK(AveK);

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
	if(!Initialized) {
		printf("Bars=%d", Bars);
		if(MinEvalLen <= Bars) {
			// 過去データを渡して初期化する	
			float opens[];
			float highs[];
			float lows[];
			float closes[];
			int mins[];
			ArraySetAsSeries(opens, false);
			ArraySetAsSeries(highs, false);
			ArraySetAsSeries(lows, false);
			ArraySetAsSeries(closes, false);
			ArraySetAsSeries(mins, false);
			ArrayResize(opens, MinEvalLen);
			ArrayResize(highs, MinEvalLen);
			ArrayResize(lows, MinEvalLen);
			ArrayResize(closes, MinEvalLen);
			ArrayResize(mins, MinEvalLen);
			for(int i = MinEvalLen - 1, j = 0; i != -1; i--, j++) {
				opens[j] = (float)Open[i];
				highs[j] = (float)High[i];
				lows[j] = (float)Low[i];
				closes[j] = (float)Close[i];
				mins[j] = Time[i];
			}
			if(FxDnnInitialize(opens, highs, lows, closes, mins) < 0) {
				printf("FxDnnInitialize() failed");
				return;
			}
			Initialized = true;
		}
	}
	if(!Initialized)
		return;

	// データを送信する
	float opens[];
	float highs[];
	float lows[];
	float closes[];
	int mins[];
	int count = 60;
	ArraySetAsSeries(opens, false);
	ArraySetAsSeries(highs, false);
	ArraySetAsSeries(lows, false);
	ArraySetAsSeries(closes, false);
	ArraySetAsSeries(mins, false);
	ArrayResize(opens, MinEvalLen);
	ArrayResize(highs, MinEvalLen);
	ArrayResize(lows, MinEvalLen);
	ArrayResize(closes, MinEvalLen);
	ArrayResize(mins, MinEvalLen);
	for(int i = count - 1, j = 0; i != -1; i--, j++) {
		opens[j] = (float)Open[i];
		highs[j] = (float)High[i];
		lows[j] = (float)Low[i];
		closes[j] = (float)Close[i];
		mins[j] = Time[i];
	}
	if(FxDnnSendFxData(count, opens, highs, lows, closes, mins) < 0) {
		printf("FxDnnSendFxData() failed");
		return;
	}
	
	// サーバーで予測する
	float pred[];
	ArraySetAsSeries(pred, false);
	ArrayResize(pred, PredictionLen);
	if(FxDnnRecvPredictionData(pred) < 0) {
		printf("FxDnnRecvPredictionData() failed");
		return;
	}

	DeltaPipsAi = pred[0];
	DeltaPipsMa = pred[1];
	Delta2PipsMa = pred[2];
	printf("DeltaPipsAi=%f : DeltaPipsMa=%f : Delta2PipsMa=%f : Reach=%.1f%%", DeltaPipsAi, DeltaPipsMa, Delta2PipsMa, CalcPredReachRate());

	// 予測結果を基に売買を行う	
	
	// 損切り判定
	// 注文時からの損失が規定値を超えたら決済する、これは最優先
	for(int i = 0; i < OrdersTotal(); i++) {
		if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {
			if(OrderMagicNumber() != 16384 || OrderSymbol() != Symbol())
				continue;
			if(-MaxLossYen < OrderProfit())
				continue; // 指定値以上損失が出ていないなら決済はしない
			MyOrderClose(OCM_LossCut);
		}
	}

	// 注文決済判定
	// 注文時と逆方向に指定PIPS動いたら決済
	bool settle = false; // 自主的に決済したかどうか
	for(int i = 0; i < OrdersTotal(); i++) {
		if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {
			if(OrderMagicNumber() != 16384 || OrderSymbol() != Symbol())
				continue;

			// 予測値へ到達したら決済
			if(OrderType() == OP_SELL || OrderType() == OP_BUY) {
				if(CloseReachRate <= CalcPredReachRate()) {
					if(MyOrderClose(OCM_Reached)) {
						settle = true;
						continue;
					}
				}
			}

			if(MinProfYen <= OrderProfit()) {
				if(OrderType() == OP_SELL) {
					// 直近の予測が上向いているか即決利益に達しているなら売りを決済
					if(MaxProfYen <= OrderProfit() || ComparePredDeltaOr(1, CloseDeltaPipsAi, CloseDeltaPipsMa, CloseDelta2PipsMa)) {
						if(MyOrderClose(OCM_Satisfied)) {
							settle = true;
						}
					}
				} else if(OrderType() == OP_BUY) {
					// 直近の予測が下向いているか即決利益に達しているなら買いを決済
					if(MaxProfYen <= OrderProfit() || ComparePredDeltaOr(-1, CloseDeltaPipsAi, CloseDeltaPipsMa, CloseDelta2PipsMa)) {
						if(MyOrderClose(OCM_Satisfied)) {
							settle = true;
						}
					}
				}
			} else {
				if(OrderType() == OP_SELL) {
					// 直近の予測が許容値を超えて上向いているなら売りを決済
					if(ComparePredDeltaOr(1, ImCloseDeltaPipsAi, ImCloseDeltaPipsMa, ImCloseDelta2PipsMa)) {
						if(MyOrderClose(OCM_Prevent)) {
							settle = true;
						}
					}
				} else if(OrderType() == OP_BUY) {
					// 直近の予測が許容値を超えて下向いているなら買いを決済
					if(ComparePredDeltaOr(-1, ImCloseDeltaPipsAi, ImCloseDeltaPipsMa, ImCloseDelta2PipsMa)) {
						if(MyOrderClose(OCM_Prevent)) {
							settle = true;
						}
					}
				}
			}
		}
	}

	// 新規注文判定
	// 予測変化量PIPSが指定値を超えたら注文
	// 新しい分足ができた瞬間に判断する　※注文回数が多すぎるとアカウント停止されるらしい
	datetime minute = Time[0] / 60;
//	if(LastMinute != minute || settle) {
	if(true) {
		double ai = DeltaPipsAi * OrderDeltaPipsAi;
		double ma = DeltaPipsMa * OrderDeltaPipsMa;
		double ma2 = Delta2PipsMa * OrderDelta2PipsMa;
		if(0.0 <= ai * ma && 0.0 <= ai * ma2 && OrderDeltaPipsAi <= fabs(DeltaPipsAi) && OrderDeltaPipsMa <= fabs(DeltaPipsMa) && OrderDelta2PipsMa <= fabs(Delta2PipsMa)) {
			int buysell = 0;
			for(int i = 0; i < OrdersTotal(); i++) {
				if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {
					if(OrderMagicNumber() != 16384 || OrderSymbol() != Symbol())
						continue;
					if(OrderType() == OP_SELL)
						buysell++;
					if(OrderType() == OP_BUY)
						buysell++;
				}
			}
			if(buysell == 0) {
				if(0.0 < DeltaPipsAi) {
					if(OrderSend(Symbol(), OP_BUY, OrderVolume, Ask, 3, Ask - 100 * Point, 0, "FxDnn", 16384,0, LightBlue)) {
						BuyCount++;
						PredStartPip = Ask * 100.0;
						PredEndPip = PredStartPip + DeltaPipsAi;
					} else {
						Print("OrderSend OP_BUY error ", GetLastError());
					}
				} else {
					if(OrderSend(Symbol(), OP_SELL, OrderVolume, Bid, 3, Bid + 100 * Point, 0, "FxDnn", 16384, 0, Blue)) {
						SellCount++;
						PredStartPip = Ask * 100.0;
						PredEndPip = PredStartPip + DeltaPipsAi;
					} else {
						Print("OrderSend OP_SELL error ", GetLastError());
					}
				}
			}
		}
		LastMinute = minute;
	}

	double buyToPer = BuyCount != 0 ? 1.0 / BuyCount : 0.0;
	double sellToPer = SellCount != 0 ? 1.0 / SellCount : 0.0;

	/*printf(
		"Buy: %d +%.2f%% -%.2f%% lc%.2f%% | Sell: %d +%.2f%% -%.2f%% lc%.2f%%",
		BuyCount,
		100.0 * BuySettlProfCount * buyToPer,
		100.0 * BuySettlLossCount * buyToPer,
		100.0 * BuyLossCutCount * buyToPer,
		SellCount,
		100.0 * SellSettlProfCount * sellToPer,
		100.0 * SellSettlLossCount * sellToPer,
		100.0 * SellLossCutCount * sellToPer);*/
}
//+------------------------------------------------------------------+

// サーバー予測値との比較、どれか一つでも条件を満たせば true が返る
bool ComparePredDeltaOr(int compareType, double ai, double ma, double ma2) {
	if(compareType > 0) {
		if(ai != 0.0 && DeltaPipsAi >= ai) return true;
		if(ma != 0.0 && DeltaPipsMa >= ma) return true;
		if(ma2 != 0.0 && Delta2PipsMa >= ma2) return true;
	} else if(compareType < 0) {
		if(ai != 0.0 && DeltaPipsAi <= -ai) return true;
		if(ma != 0.0 && DeltaPipsMa <= -ma) return true;
		if(ma2 != 0.0 && Delta2PipsMa <= -ma2) return true;
	}
	return false;
}

// サーバー予測値との比較、全てが条件を満たせば true が返る
bool ComparePredDeltaAnd(int compareType, double ai, double ma, double ma2) {
	if(compareType > 0) {
		if(ai != 0.0 && DeltaPipsAi < ai) return false;
		if(ma != 0.0 && DeltaPipsMa < ma) return false;
		if(ma2 != 0.0 && Delta2PipsMa < ma2) return false;
	} else if(compareType < 0) {
		if(ai != 0.0 && DeltaPipsAi > -ai) return false;
		if(ma != 0.0 && DeltaPipsMa > -ma) return false;
		if(ma2 != 0.0 && Delta2PipsMa > -ma2) return false;
	}
	return true;
}

// 注文を決済する
bool MyOrderClose(OrderCloseMode mode) {
	if(OrderType() == OP_SELL) {
		if(OrderClose(OrderTicket(), OrderLots(), Ask, 3, mode == OCM_LossCut ? Red : White)) {
			double prof = OrderProfit();
			printf("%s: %f", GetOrderCloseModeName(mode), prof);
			if(prof < 0.0) {
				SellSettlLossCount++;
				if(mode == OCM_LossCut)
					SellLossCutCount++;
			} else if(0.0 < prof) {
				SellSettlProfCount++;
			}
			PredStartPip = PredEndPip = 0.0;
			return true;
		} else {
			Print("OrderClose error ",GetLastError());
			return false;
		}
	} else if(OrderType() == OP_BUY) {
		if(OrderClose(OrderTicket(), OrderLots(), Bid, 3, mode == OCM_LossCut ? Red : White)) {
			double prof = OrderProfit();
			printf("%s: %f", GetOrderCloseModeName(mode), prof);
			if(prof < 0.0) {
				BuySettlLossCount++;
				if(mode == OCM_LossCut)
					BuyLossCutCount++;
			} else if(0.0 < prof) {
				BuySettlProfCount++;
			}
			PredStartPip = PredEndPip = 0.0;
			return true;
		} else {
			Print("OrderClose error ",GetLastError());
			return false;
		}
	} else {
		return false;
	}
}

// 注文決済モード名の取得
string GetOrderCloseModeName(OrderCloseMode mode) {
	switch(mode) {
	case OCM_LossCut: return "損切り";
	case OCM_Satisfied: return "もう満足";
	case OCM_Reached: return "予測値到達";
	case OCM_Prevent: return "予防";
	default: return "";
	}
}

// 現在の予測到達率%を計算する
double CalcPredReachRate() {
	if(PredStartPip == 0.0 && PredEndPip == 0.0)
		return 0.0; // まだ予測してない
	return 100.0 * (Ask * 100.0 - PredStartPip) / (PredEndPip - PredStartPip);
}
