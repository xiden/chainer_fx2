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
	int FxDnnSetYenAveK(double k);
#import

input double OrderDeltaPips = 2; // 注文判定PIPS変化量
input double SettlDeltaPips = 1; // 決済判定反転PIPS変化量
input double SettlDeltaPipsMa = 1; // 決済判定反転PIPS変化量（ガウシアン移動平均）
input double MaxLossPips = 2.0; // 最大許容損失PIPS
input double MinGainPips = 1.0; // 最小許容利益PIPS
input double OrderVolume = 1.0; // 注文数量

bool HasError = false;
bool Initialized = false;
int MinEvalLen;
int PredictionLen;
datetime LastMinute;
int BuyCount = 0; // 買い注文回数
int BuySettlLossCount = 0; // 買い注文決済時の損失回数
int BuySettlProfCount = 0; // 買い注文決済時の利益回数
int BuyLossCutCount = 0; // 買い注文時の損切り回数
int SellCount = 0; // 売り注文回数
int SellSettlLossCount = 0; // 売り注文決済時の損失回数
int SellSettlProfCount = 0; // 売り注文決済時の利益回数
int SellLossCutCount = 0; // 売り注文時の損切り回数


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
	printf("MinEvalLen=%d", MinEvalLen);
	printf("RetValLen=%d", PredictionLen);

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
		if(MinEvalLen <= Bars) {
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
				return;
			}
			Initialized = true;
		}
	}
	if(!Initialized)
		return;

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
	
	// サーバーで予測する
	float pred[];
	ArraySetAsSeries(pred, false);
	ArrayResize(pred, PredictionLen);
	if(FxDnnRecvPredictionData(pred) < 0) {
		printf("FxDnnRecvPredictionData() failed");
		return;
	}
	
	double deltaPips = pred[0];
	double deltaPipsMa = pred[1];
	//printf("deltaPips=%f : deltaPipsMa=%f", deltaPips, deltaPipsMa);

	// 予測結果を基に売買を行う	
	
	// 損切り判定
	// 注文時からの損失が規定値を超えたら決済する
	for(int i = 0; i < OrdersTotal(); i++) {
		if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {
			if(OrderMagicNumber() != 16384 || OrderSymbol() != Symbol())
				continue;

			if(OrderType() == OP_SELL) {
				if(OrderOpenPrice() + 10.0 * MaxLossPips * Point <= Ask) {
					if(!OrderClose(OrderTicket(), OrderLots(), Ask, 3, Yellow))
						Print("OrderClose error ",GetLastError());
					SellLossCutCount++;
				}
			}

			if(OrderType() == OP_BUY) {
				if(Bid <= OrderOpenPrice() - 10.0 * MaxLossPips * Point) {
					if(!OrderClose(OrderTicket(), OrderLots(), Bid, 3, Yellow))
						Print("OrderClose error ",GetLastError());
					BuyLossCutCount++;
				}
			}
		}
	}

	// 注文決済判定
	// 注文時と逆方向に指定PIPS動いたら決済
	for(int i = 0; i < OrdersTotal(); i++) {
		if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {
			if(OrderMagicNumber() != 16384 || OrderSymbol() != Symbol())
				continue;
			if(SettlDeltaPips <= deltaPips || SettlDeltaPipsMa <= deltaPipsMa) {
				// 直近の予測が上向いているなら売りを決済
				if(OrderType() == OP_SELL && Ask <= OrderOpenPrice() - 10.0 * MinGainPips * Point) {
					if(OrderClose(OrderTicket(), OrderLots(), Ask, 3, White)) {
						double prof = OrderProfit();
						if(prof < 0.0)
							SellSettlLossCount++;
						else if(0.0 < prof)
							SellSettlProfCount++;
					} else {
						Print("OrderClose error ",GetLastError());
					}
				}
			} else if(deltaPips <= -SettlDeltaPips || deltaPipsMa <= -SettlDeltaPipsMa) {
				// 直近の予測が下向いているなら買いを決済
				if(OrderType() == OP_BUY && OrderOpenPrice() + 10.0 * MinGainPips * Point <= Bid) {
					if(OrderClose(OrderTicket(), OrderLots(), Bid, 3, White)) {
						double prof = OrderProfit();
						if(prof < 0.0)
							BuySettlLossCount++;
						else if(0.0 < prof)
							BuySettlProfCount++;
					} else {
						Print("OrderClose error ",GetLastError());
					}
				}
			}
		}
	}

	// 新規注文判定
	// 予測変化量PIPSが指定値を超えたら注文
	// 新しい分足ができた瞬間に判断する　※注文回数が多すぎるとアカウント停止されるらしい
	datetime minute = Time[0] / 60;
//	if(LastMinute != minute) {
	if(true) {
		if(deltaPips <= -OrderDeltaPips || OrderDeltaPips <= deltaPips) {
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
				if(0.0 < deltaPips) {
					if(OrderSend(Symbol(), OP_BUY, OrderVolume, Ask, 3, Ask - 100 * Point, 0, "FxDnn", 16384,0, Green))
						BuyCount++;
					else
						Print("OrderSend OP_BUY error ", GetLastError());
				} else {
					if(OrderSend(Symbol(), OP_SELL, OrderVolume, Bid, 3, Bid + 100 * Point, 0, "FxDnn", 16384, 0, Red))
						SellCount++;
					else
						Print("OrderSend OP_SELL error ", GetLastError());
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
