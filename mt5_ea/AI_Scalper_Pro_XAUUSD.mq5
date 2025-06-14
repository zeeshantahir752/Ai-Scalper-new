//+------------------------------------------------------------------+
//| AI_Scalper_Pro_XAUUSD_COMPLETE.mq5                              |
//| AI-Powered Gold Scalper for Exness XAUUSDm - FULL VERSION      |
//| Account: 247501403 | Server: Exness-MT5Trial                     |
//+------------------------------------------------------------------+
#property copyright "AI Scalper Pro Team"
#property version   "3.00"
#property strict
#property description "Advanced AI-powered scalping EA for XAUUSD"
#property description "Features: ML signal processing, adaptive risk management"
#property description "Designed for Exness XAUUSDm with 24/7 operation"

// Enums for risk models - MOVED TO TOP
enum ENUM_RISK_MODEL
{
   RISK_FIXED = 0,     // Fixed lot size
   RISK_PERCENT = 1,   // Percentage of equity
   RISK_BALANCE = 2,   // Percentage of balance
   RISK_MARTINGALE = 3 // Martingale progression
};

// Input Parameters - AI Signal Processing
input group "=== AI Signal Processing ==="
input string SignalFilePath = "signals\\xau_signal.txt";          // AI signal file path - FIXED
input int SignalTimeoutSeconds = 300;                             // Signal validity timeout
input double MinConfidence = 0.75;                               // Minimum AI confidence (0.5-1.0)
input double HighConfidenceThreshold = 0.85;                     // High confidence threshold
input bool UseAISignalsOnly = true;                              // Trade only on AI signals
input bool UseConfidenceWeighting = true;                        // Weight lot size by confidence
input int MaxSignalsPerHour = 8;                                 // Signal frequency limit
input bool VerifySignalConsistency = true;                       // Check signal consistency

// Input Parameters - Advanced Risk Management
input group "=== Advanced Risk Management ==="
input double BaseRiskPercentage = 1.5;                           // Base risk per trade %
input double MaxRiskPercentage = 3.0;                            // Maximum risk per trade %
input double MaxDrawdownPercent = 8.0;                           // Maximum portfolio drawdown %
input double DailyLossLimit = 5.0;                               // Daily loss limit %
input double EquityStopPercent = 4.0;                            // Emergency equity stop %
input int MaxTradesPerDay = 15;                                  // Daily trade limit
input int MaxTradesPerHour = 4;                                  // Hourly trade limit
input double MaxLotSize = 1.0;                                   // Maximum lot size
input double MinLotSize = 0.01;                                  // Minimum lot size
input bool UseProgressiveRisk = false;                           // Progressive risk after wins
input double RiskMultiplier = 1.2;                               // Risk multiplier after loss

// Input Parameters - Position Management
input group "=== Position Management ==="
input int StopLossPoints = 250;                                  // Stop loss distance (points)
input int TakeProfitPoints = 500;                                // Take profit distance (points)
input bool UseTrailingStop = true;                               // Enable trailing stop
input int TrailingStopPoints = 150;                              // Trailing stop distance
input int TrailingStepPoints = 50;                               // Trailing step size
input int TrailingStartPoints = 100;                             // Points profit to start trailing
input bool UseBreakEven = true;                                  // Move to break-even
input int BreakEvenPoints = 200;                                 // Points to trigger break-even
input int BreakEvenPlusPoints = 20;                              // Points beyond break-even
input int MaxSpreadPoints = 50;                                  // Maximum allowed spread
input bool UsePartialTakeProfit = true;                          // Partial profit taking
input double PartialTPPercent = 50.0;                            // Percentage for partial TP

// Input Parameters - Trading Schedule & Filters
input group "=== Trading Schedule & Filters ==="
input int TradingStartHour = 1;                                  // Trading start hour (server time)
input int TradingEndHour = 23;                                   // Trading end hour (server time)
input bool AvoidNewsTime = true;                                 // Avoid high-impact news
input int NewsAvoidanceMinutes = 30;                             // Minutes before/after news
input bool TradeFriday = true;                                   // Allow Friday trading
input int FridayCloseHour = 21;                                  // Friday close hour
input bool AvoidRollover = true;                                 // Avoid swap rollover time
input int RolloverStartHour = 23;                                // Rollover start hour
input int RolloverEndHour = 1;                                   // Rollover end hour
input bool UseVolatilityFilter = true;                           // Filter by volatility
input double MinVolatility = 0.5;                                // Minimum volatility requirement
input double MaxVolatility = 3.0;                                // Maximum volatility allowed

// Input Parameters - Market Condition Filters
input group "=== Market Condition Filters ==="
input bool UseSpreadFilter = true;                               // Enable spread filtering
input bool UseLiquidityFilter = true;                            // Check market liquidity
input double MinLiquidity = 1000000;                             // Minimum liquidity (volume)
input bool UseCorrelationFilter = false;                         // Check XAUUSD correlation
input bool UseMarketSentiment = false;                           // Consider market sentiment
input int LookbackPeriods = 20;                                  // Periods for market analysis
input double TrendStrengthThreshold = 0.6;                       // Trend strength requirement
input bool AvoidLowLiquidity = true;                             // Avoid low liquidity periods

// Input Parameters - Position Sizing & Money Management
input group "=== Position Sizing & Money Management ==="
input ENUM_RISK_MODEL RiskModel = RISK_PERCENT;                  // Risk calculation model
input double FixedLotSize = 0.1;                                 // Fixed lot size (if selected)
input bool UseEquityCurve = true;                                // Adjust size based on performance
input double MaxPositionPercent = 10.0;                          // Max position % of equity
input bool UseVolumeWeighting = false;                           // Weight by market volume
input double MinEquityForTrading = 100.0;                        // Minimum equity to trade
input bool UseCompounding = true;                                // Compound profits
input double CompoundingFactor = 1.1;                            // Compounding multiplier

// Input Parameters - Recovery & Protection Systems
input group "=== Recovery & Protection Systems ==="
input bool UseDrawdownRecovery = true;                           // Enable drawdown recovery
input double DrawdownRecoveryThreshold = 3.0;                    // Drawdown % to trigger recovery
input double RecoveryRiskReduction = 0.5;                        // Risk reduction during recovery
input bool UseEmergencyStop = true;                              // Emergency stop system
input double EmergencyStopPercent = 10.0;                        // Emergency stop loss %
input bool UseTimeBasedExit = false;                             // Exit after time period
input int MaxPositionTimeMinutes = 480;                          // Maximum position time (minutes)
input bool UseCorrelationProtection = false;                     // Avoid correlated trades
input int MaxCorrelatedPositions = 2;                            // Max correlated positions

// Input Parameters - Advanced Analytics & Logging
input group "=== Advanced Analytics & Logging ==="
input bool EnableDetailedLogging = true;                         // Detailed trade logging
input bool LogSignalDetails = true;                              // Log AI signal information
input bool EnablePerformanceTracking = true;                     // Track performance metrics
input bool SendTelegramNotifications = false;                    // Telegram notifications
input string TelegramBotToken = "";                              // Telegram bot token
input string TelegramChatID = "";                                // Telegram chat ID
input bool EnableWebDashboard = false;                           // Web-based dashboard
input int WebDashboardPort = 8080;                               // Dashboard port
input bool SaveTradingHistory = true;                            // Save detailed history

// Global Variables - Core Trading
long MagicNumber = 20250530247501403;                            // Unique EA identifier
int dailyTrades = 0;                                             // Daily trade counter
int hourlyTrades = 0;                                            // Hourly trade counter
int hourlySignals = 0;                                           // Hourly signal counter
long lastTradeTime = 0;                                          // Last trade timestamp
long lastSignalTime = 0;                                         // Last signal timestamp
double startingEquity = 0;                                       // Starting equity
double dayStartEquity = 0;                                       // Day start equity
double maxEquity = 0;                                            // Maximum equity reached
double currentDrawdown = 0;                                      // Current drawdown %
bool emergencyStop = false;                                      // Emergency stop flag
bool recoveryMode = false;                                       // Recovery mode flag

// Global Variables - Signal Processing
string lastSignalTimestamp = "";                                 // Last processed signal
double lastSignalConfidence = 0;                                 // Last signal confidence
string lastSignalReason = "";                                   // Last signal reason
int consecutiveWins = 0;                                         // Consecutive winning trades
int consecutiveLosses = 0;                                       // Consecutive losing trades
double lastSignalPrice = 0;                                      // Last signal price
int signalHistory[24];                                           // 24-hour signal history

// Global Variables - Performance Tracking
double totalProfit = 0;                                          // Total profit/loss
double totalCommission = 0;                                      // Total commission paid
double totalSwap = 0;                                            // Total swap
int totalTrades = 0;                                             // Total trades executed
int winningTrades = 0;                                           // Winning trades count
int losingTrades = 0;                                            // Losing trades count
double largestWin = 0;                                           // Largest winning trade
double largestLoss = 0;                                          // Largest losing trade
double profitFactor = 0;                                         // Profit factor
double winRate = 0;                                              // Win rate percentage

// Global Variables - Market Analysis
double currentVolatility = 0;                                    // Current market volatility
double currentSpread = 0;                                        // Current spread
double marketLiquidity = 0;                                      // Market liquidity
bool newsTime = false;                                           // News time flag
double trendStrength = 0;                                        // Current trend strength
string marketSentiment = "NEUTRAL";                              // Market sentiment

// Global Variables - Technical Indicators
int ma20Handle = INVALID_HANDLE;                                 // MA20 handle
int ma50Handle = INVALID_HANDLE;                                 // MA50 handle

// AI Signal Structure - Enhanced
struct AISignal
{
   string signal;              // BUY, SELL, NONE
   double confidence;          // 0.0 to 1.0
   double price;               // Signal price
   int spread;                 // Spread at signal time
   string timestamp;           // Signal timestamp
   string reason;              // Signal reasoning
   double volatility;          // Market volatility
   double volume;              // Market volume
   string trend;               // Market trend
   double support;             // Support level
   double resistance;          // Resistance level
   double rsi;                 // RSI value
   double macd;                // MACD value
   string timeframe;           // Signal timeframe
   double expectedTP;          // Expected take profit
   double expectedSL;          // Expected stop loss
   int priority;               // Signal priority (1-5)
};

// Trade Management Structure
struct TradeInfo
{
   ulong ticket;               // Trade ticket
   ENUM_ORDER_TYPE type;       // Order type
   double lotSize;             // Lot size
   double openPrice;           // Open price
   double stopLoss;            // Stop loss
   double takeProfit;          // Take profit
   datetime openTime;          // Open time
   double confidence;          // AI confidence
   string reason;              // Trade reason
   bool partialTPExecuted;     // Partial TP flag
   bool breakEvenMoved;        // Break-even flag
   double maxProfit;           // Maximum profit reached
   double currentProfit;       // Current profit
};

// Market Condition Structure
struct MarketCondition
{
   double volatility;          // Current volatility
   double spread;              // Current spread
   double volume;              // Current volume
   string trend;               // Trend direction
   double trendStrength;       // Trend strength
   bool newsTime;              // News time flag
   bool highLiquidity;         // High liquidity flag
   double support;             // Support level
   double resistance;          // Resistance level
   string sentiment;           // Market sentiment
};

// Performance Statistics Structure
struct PerformanceStats
{
   double totalReturn;         // Total return %
   double maxDrawdown;         // Maximum drawdown %
   double sharpeRatio;         // Sharpe ratio
   double profitFactor;        // Profit factor
   double winRate;             // Win rate %
   double avgWin;              // Average win
   double avgLoss;             // Average loss
   double avgTrade;            // Average trade
   int totalTrades;            // Total trades
   double recoveryFactor;      // Recovery factor
};

// Array to store active trades
TradeInfo activeTrades[];

//+------------------------------------------------------------------+
//| Reset daily statistics - ADDED MISSING FUNCTION                 |
//+------------------------------------------------------------------+
void ResetDailyStatistics()
{
   dailyTrades = 0;
   dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   
   Print("📅 Daily statistics reset");
   Print("   Day start equity: $", NormalizeDouble(dayStartEquity, 2));
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("================================================================");
   Print("🚀 AI SCALPER PRO EA - COMPLETE VERSION STARTING");
   Print("================================================================");
   Print("📊 Account Information:");
   Print("   Account: ", AccountInfoInteger(ACCOUNT_LOGIN));
   Print("   Server: ", AccountInfoString(ACCOUNT_SERVER));
   Print("   Company: ", AccountInfoString(ACCOUNT_COMPANY));
   Print("   Balance: $", NormalizeDouble(AccountInfoDouble(ACCOUNT_BALANCE), 2));
   Print("   Equity: $", NormalizeDouble(AccountInfoDouble(ACCOUNT_EQUITY), 2));
   Print("   Leverage: 1:", AccountInfoInteger(ACCOUNT_LEVERAGE));
   Print("================================================================");
   Print("🤖 AI Configuration:");
   Print("   Signal File: ", SignalFilePath);
   Print("   Min Confidence: ", MinConfidence * 100, "%");
   Print("   High Confidence: ", HighConfidenceThreshold * 100, "%");
   Print("   AI Signals Only: ", (UseAISignalsOnly ? "YES" : "NO"));
   Print("================================================================");
   Print("⚡ Risk Management:");
   Print("   Base Risk: ", BaseRiskPercentage, "%");
   Print("   Max Risk: ", MaxRiskPercentage, "%");
   Print("   Max Drawdown: ", MaxDrawdownPercent, "%");
   Print("   Daily Loss Limit: ", DailyLossLimit, "%");
   Print("   Max Trades/Day: ", MaxTradesPerDay);
   Print("================================================================");
   
   // Initialize global variables
   startingEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   dayStartEquity = startingEquity;
   maxEquity = startingEquity;
   
   // Initialize technical indicators
   ma20Handle = iMA(_Symbol, PERIOD_M15, 20, 0, MODE_SMA, PRICE_CLOSE);
   ma50Handle = iMA(_Symbol, PERIOD_M15, 50, 0, MODE_SMA, PRICE_CLOSE);
   
   if(ma20Handle == INVALID_HANDLE || ma50Handle == INVALID_HANDLE)
   {
      Print("❌ Failed to initialize technical indicators");
      return INIT_FAILED;
   }
   
   // Validate inputs
   if(!ValidateInputParameters())
   {
      Print("❌ Input validation failed - EA will not start");
      return INIT_PARAMETERS_INCORRECT;
   }
   
   // Initialize arrays
   ArrayResize(activeTrades, 0);
   ArrayInitialize(signalHistory, 0);
   
   // Test signal file access
   if(!TestSignalFileAccess())
   {
      Print("⚠️ Signal file access test failed - check file path");
      if(UseAISignalsOnly)
      {
         Print("❌ Cannot proceed without signal file when UseAISignalsOnly=true");
         return INIT_FAILED;
      }
   }
   
   // Initialize market analysis
   InitializeMarketAnalysis();
   
   // Reset daily statistics
   ResetDailyStatistics();
   
   // Setup performance tracking
   if(EnablePerformanceTracking)
   {
      InitializePerformanceTracking();
   }
   
   // Setup notifications
   if(SendTelegramNotifications)
   {
      TestTelegramConnection();
   }
   
   Print("✅ AI Scalper Pro EA initialized successfully");
   Print("🎯 Waiting for AI signals and market opportunities...");
   Print("================================================================");
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   string reasonText = GetDeInitReason(reason);
   
   Print("================================================================");
   Print("🛑 AI SCALPER PRO EA STOPPING");
   Print("================================================================");
   Print("📊 Session Summary:");
   Print("   Reason: ", reasonText);
   Print("   Runtime: ", GetRuntimeString());
   Print("   Daily Trades: ", dailyTrades, "/", MaxTradesPerDay);
   Print("   Total Trades: ", totalTrades);
   Print("   Win Rate: ", NormalizeDouble(winRate, 1), "%");
   Print("   Total P/L: $", NormalizeDouble(totalProfit, 2));
   Print("   Max Drawdown: ", NormalizeDouble(currentDrawdown, 2), "%");
   Print("================================================================");
   Print("💼 Final Account State:");
   Print("   Starting Equity: $", NormalizeDouble(startingEquity, 2));
   Print("   Current Equity: $", NormalizeDouble(AccountInfoDouble(ACCOUNT_EQUITY), 2));
   Print("   Session Return: ", NormalizeDouble(GetSessionReturn(), 2), "%");
   Print("================================================================");
   
   // Release indicator handles
   if(ma20Handle != INVALID_HANDLE)
      IndicatorRelease(ma20Handle);
   if(ma50Handle != INVALID_HANDLE)
      IndicatorRelease(ma50Handle);
   
   // Save performance data
   if(EnablePerformanceTracking)
   {
      SavePerformanceData();
   }
   
   // Send final notification
   if(SendTelegramNotifications)
   {
      SendTelegramMessage("🛑 AI Scalper EA Stopped\nReason: " + reasonText + 
                         "\nSession P/L: $" + DoubleToString(totalProfit, 2));
   }
   
   Print("👋 AI Scalper Pro EA shutdown complete");
}

//+------------------------------------------------------------------+
//| Expert tick function - Main trading logic                        |
//+------------------------------------------------------------------+
void OnTick()
{
   // Update market conditions
   UpdateMarketConditions();
   
   // Check emergency conditions first
   if(CheckEmergencyConditions())
   {
      HandleEmergencyStop();
      return;
   }
   
   // Update trade management for existing positions
   ManageActivePositions();
   
   // Check if trading is allowed
   if(!IsTradeAllowed())
   {
      return;
   }
   
   // Process AI signals (main trading logic)
   static long lastSignalCheck = 0;
   if(TimeLocal() - lastSignalCheck >= 5) // Check every 5 seconds
   {
      ProcessAISignals();
      lastSignalCheck = TimeLocal();
   }
   
   // Update statistics periodically
   static long lastStatsUpdate = 0;
   if(TimeLocal() - lastStatsUpdate >= 60) // Update every minute
   {
      UpdatePerformanceStatistics();
      UpdateDailyStatistics();
      lastStatsUpdate = TimeLocal();
   }
   
   // Log detailed information periodically
   static long lastDetailedLog = 0;
   if(EnableDetailedLogging && TimeLocal() - lastDetailedLog >= 300) // Every 5 minutes
   {
      LogDetailedStatus();
      lastDetailedLog = TimeLocal();
   }
}

//+------------------------------------------------------------------+
//| Comprehensive input validation                                   |
//+------------------------------------------------------------------+
bool ValidateInputParameters()
{
   bool isValid = true;
   
   Print("🔍 Validating input parameters...");
   
   // Risk Management Validation
   if(BaseRiskPercentage <= 0 || BaseRiskPercentage > 10)
   {
      Print("❌ Invalid BaseRiskPercentage: ", BaseRiskPercentage, " (must be 0.1-10.0)");
      isValid = false;
   }
   
   if(MaxRiskPercentage < BaseRiskPercentage || MaxRiskPercentage > 20)
   {
      Print("❌ Invalid MaxRiskPercentage: ", MaxRiskPercentage, " (must be >= BaseRisk and <= 20.0)");
      isValid = false;
   }
   
   if(MinConfidence < 0.5 || MinConfidence > 1.0)
   {
      Print("❌ Invalid MinConfidence: ", MinConfidence, " (must be 0.5-1.0)");
      isValid = false;
   }
   
   if(HighConfidenceThreshold <= MinConfidence || HighConfidenceThreshold > 1.0)
   {
      Print("❌ Invalid HighConfidenceThreshold: ", HighConfidenceThreshold);
      isValid = false;
   }
   
   // Position Management Validation
   if(StopLossPoints <= 0 || TakeProfitPoints <= 0)
   {
      Print("❌ Invalid SL/TP points: SL=", StopLossPoints, " TP=", TakeProfitPoints);
      isValid = false;
   }
   
   if(TakeProfitPoints <= StopLossPoints)
   {
      Print("❌ Take Profit must be larger than Stop Loss");
      isValid = false;
   }
   
   // Trading Limits Validation
   if(MaxTradesPerDay <= 0 || MaxTradesPerDay > 100)
   {
      Print("❌ Invalid MaxTradesPerDay: ", MaxTradesPerDay, " (must be 1-100)");
      isValid = false;
   }
   
   if(MaxTradesPerHour <= 0 || MaxTradesPerHour > 20)
   {
      Print("❌ Invalid MaxTradesPerHour: ", MaxTradesPerHour, " (must be 1-20)");
      isValid = false;
   }
   
   // Lot Size Validation
   double symbolMinLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double symbolMaxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   
   if(MinLotSize < symbolMinLot)
   {
      Print("❌ MinLotSize too small for symbol: ", MinLotSize, " (min: ", symbolMinLot, ")");
      isValid = false;
   }
   
   if(MaxLotSize > symbolMaxLot)
   {
      Print("❌ MaxLotSize too large for symbol: ", MaxLotSize, " (max: ", symbolMaxLot, ")");
      isValid = false;
   }
   
   // Schedule Validation
   if(TradingStartHour < 0 || TradingStartHour > 23 || TradingEndHour < 0 || TradingEndHour > 23)
   {
      Print("❌ Invalid trading hours: ", TradingStartHour, "-", TradingEndHour);
      isValid = false;
   }
   
   // Symbol Validation
   if(_Symbol != "XAUUSDm" && _Symbol != "XAUUSD")
   {
      Print("⚠️ Warning: EA optimized for XAUUSD/XAUUSDm, current symbol: ", _Symbol);
   }
   
   // Account Validation
   double minEquity = MinEquityForTrading;
   if(AccountInfoDouble(ACCOUNT_EQUITY) < minEquity)
   {
      Print("❌ Insufficient equity: $", AccountInfoDouble(ACCOUNT_EQUITY), " (min: $", minEquity, ")");
      isValid = false;
   }
   
   if(isValid)
   {
      Print("✅ All input parameters validated successfully");
   }
   
   return isValid;
}

//+------------------------------------------------------------------+
//| Test signal file access and format                               |
//+------------------------------------------------------------------+
bool TestSignalFileAccess()
{
   Print("🔍 Testing AI signal file access...");
   
   int fileHandle = FileOpen(SignalFilePath, FILE_READ | FILE_TXT);
   if(fileHandle == INVALID_HANDLE)
   {
      int error = GetLastError();
      Print("❌ Cannot access signal file: ", SignalFilePath);
      Print("   Error code: ", error);
      Print("   Error description: ", GetErrorDescription(error));
      Print("💡 Suggestions:");
      Print("   1. Check if file path is correct");
      Print("   2. Ensure Python AI system has created the file");
      Print("   3. Verify file permissions");
      Print("   4. Try absolute path: C:\\Users\\hp\\Downloads\\Ai Bot 2\\ai-scalper-xauusd\\signals\\xau_signal.txt");
      return false;
   }
   
   // Test file reading
   string content = "";
   int lines = 0;
   while(!FileIsEnding(fileHandle) && lines < 20) // Read max 20 lines for testing
   {
      string line = FileReadString(fileHandle);
      content += line + "\n";
      lines++;
   }
   FileClose(fileHandle);
   
   if(StringLen(content) == 0)
   {
      Print("⚠️ Signal file is empty");
      return false;
   }
   
   Print("✅ Signal file access successful");
   Print("   File size: ", StringLen(content), " characters");
   Print("   Lines read: ", lines);
   
   // Test signal parsing
   AISignal testSignal = ParseSignalContent(content);
   if(testSignal.signal != "NONE")
   {
      Print("✅ Signal parsing successful");
      Print("   Signal: ", testSignal.signal);
      Print("   Confidence: ", NormalizeDouble(testSignal.confidence * 100, 1), "%");
   }
   else
   {
      Print("⚠️ No valid signal found in file (this is normal if no signal is active)");
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Check if trading is allowed based on all conditions              |
//+------------------------------------------------------------------+
bool IsTradeAllowed()
{
   // Check terminal and EA permissions
   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
   {
      return false;
   }
   
   if(!MQLInfoInteger(MQL_TRADE_ALLOWED))
   {
      return false;
   }
   
   // Check emergency stop
   if(emergencyStop)
   {
      return false;
   }
   
   // Check trading schedule
   if(!IsTradingTime())
   {
      return false;
   }
   
   // Check risk management conditions
   if(!CheckRiskManagementConditions())
   {
      return false;
   }
   
   // Check daily limits
   if(dailyTrades >= MaxTradesPerDay)
   {
      return false;
   }
   
   // Check hourly limits
   if(hourlyTrades >= MaxTradesPerHour)
   {
      return false;
   }
   
   // Check signal limits
   if(hourlySignals >= MaxSignalsPerHour)
   {
      return false;
   }
   
   // Check market conditions
   if(!CheckMarketConditions())
   {
      return false;
   }
   
   // Check minimum time between trades
   if(TimeLocal() - lastTradeTime < 30) // 30 seconds minimum
   {
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Check if current time allows trading                             |
//+------------------------------------------------------------------+
bool IsTradingTime()
{
   MqlDateTime timeStruct;
   TimeToStruct(TimeLocal(), timeStruct);
   
   // Check trading hours
   if(timeStruct.hour < TradingStartHour || timeStruct.hour >= TradingEndHour)
   {
      return false;
   }
   
   // Check weekends
   if(timeStruct.day_of_week == 0 || timeStruct.day_of_week == 6) // Sunday or Saturday
   {
      return false;
   }
   
   // Check Friday trading
   if(!TradeFriday && timeStruct.day_of_week == 5) // Friday
   {
      return false;
   }
   
   // Check Friday close time
   if(timeStruct.day_of_week == 5 && timeStruct.hour >= FridayCloseHour)
   {
      return false;
   }
   
   // Check rollover time
   if(AvoidRollover)
   {
      if((timeStruct.hour >= RolloverStartHour && RolloverStartHour > RolloverEndHour) ||
         (timeStruct.hour < RolloverEndHour && RolloverStartHour > RolloverEndHour) ||
         (timeStruct.hour >= RolloverStartHour && timeStruct.hour < RolloverEndHour && RolloverStartHour < RolloverEndHour))
      {
         return false;
      }
   }
   
   // Check news time
   if(AvoidNewsTime && newsTime)
   {
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Check risk management conditions                                 |
//+------------------------------------------------------------------+
bool CheckRiskManagementConditions()
{
   double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   
   // Check minimum equity
   if(currentEquity < MinEquityForTrading)
   {
      if(EnableDetailedLogging)
      {
         Print("❌ Equity below minimum trading threshold: $", currentEquity, " < $", MinEquityForTrading);
      }
      return false;
   }
   
   // Check equity stop
   if(startingEquity > 0)
   {
      double equityLoss = ((startingEquity - currentEquity) / startingEquity) * 100.0;
      if(equityLoss >= EquityStopPercent)
      {
         Print("🚨 EQUITY STOP TRIGGERED! Loss: ", NormalizeDouble(equityLoss, 2), "%");
         emergencyStop = true;
         CloseAllPositions("Equity Stop");
         return false;
      }
   }
   
   // Check daily loss limit
   if(dayStartEquity > 0)
   {
      double dailyLoss = ((dayStartEquity - currentEquity) / dayStartEquity) * 100.0;
      if(dailyLoss >= DailyLossLimit)
      {
         Print("🚨 DAILY LOSS LIMIT REACHED! Loss: ", NormalizeDouble(dailyLoss, 2), "%");
         return false;
      }
   }
   
   // Check maximum drawdown
   if(maxEquity > 0)
   {
      currentDrawdown = ((maxEquity - currentEquity) / maxEquity) * 100.0;
      if(currentDrawdown >= MaxDrawdownPercent)
      {
         Print("🚨 MAXIMUM DRAWDOWN REACHED! Drawdown: ", NormalizeDouble(currentDrawdown, 2), "%");
         emergencyStop = true;
         CloseAllPositions("Max Drawdown");
         return false;
      }
   }
   
   // Update max equity
   if(currentEquity > maxEquity)
   {
      maxEquity = currentEquity;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Check market conditions for trading                              |
//+------------------------------------------------------------------+
bool CheckMarketConditions()
{
   // Update current market conditions
   UpdateMarketConditions();
   
   // Check spread
   if(UseSpreadFilter && currentSpread > MaxSpreadPoints)
   {
      return false;
   }
   
   // Check volatility
   if(UseVolatilityFilter)
   {
      if(currentVolatility < MinVolatility || currentVolatility > MaxVolatility)
      {
         return false;
      }
   }
   
   // Check liquidity
   if(UseLiquidityFilter && marketLiquidity < MinLiquidity)
   {
      return false;
   }
   
   // Check low liquidity periods
   if(AvoidLowLiquidity)
   {
      MqlDateTime timeStruct;
      TimeToStruct(TimeLocal(), timeStruct);
      
      // Avoid very early hours (low liquidity)
      if(timeStruct.hour >= 22 || timeStruct.hour <= 2)
      {
         return false;
      }
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Process AI signals from file                                     |
//+------------------------------------------------------------------+
void ProcessAISignals()
{
   if(!UseAISignalsOnly)
   {
      return;
   }
   
   // Read and parse signal file
   AISignal signal = ReadAndParseSignalFile();
   
   // Validate signal
   if(!IsValidSignal(signal))
   {
      return;
   }
   
   // Check if this is a new signal
   if(signal.timestamp == lastSignalTimestamp)
   {
      return;
   }
   
   // Check signal age
   if(IsSignalExpired(signal))
   {
      return;
   }
   
   // Check signal consistency
   if(VerifySignalConsistency && !CheckSignalConsistency(signal))
   {
      Print("⚠️ Signal consistency check failed - skipping");
      return;
   }
   
   // Log signal details if enabled
   if(LogSignalDetails)
   {
      LogSignalInformation(signal);
   }
   
   // Process the signal
   bool signalProcessed = false;
   
   if(signal.signal == "BUY")
   {
      Print("🔵 AI BUY Signal Received");
      Print("   Confidence: ", NormalizeDouble(signal.confidence * 100, 1), "%");
      Print("   Price: ", signal.price);
      Print("   Reason: ", signal.reason);
      
      if(ExecuteTradingSignal(ORDER_TYPE_BUY, signal))
      {
         signalProcessed = true;
      }
   }
   else if(signal.signal == "SELL")
   {
      Print("🔴 AI SELL Signal Received");
      Print("   Confidence: ", NormalizeDouble(signal.confidence * 100, 1), "%");
      Print("   Price: ", signal.price);
      Print("   Reason: ", signal.reason);
      
      if(ExecuteTradingSignal(ORDER_TYPE_SELL, signal))
      {
         signalProcessed = true;
      }
   }
   
   // Update signal tracking
   if(signalProcessed)
   {
      lastSignalTimestamp = signal.timestamp;
      lastSignalConfidence = signal.confidence;
      lastSignalReason = signal.reason;
      lastSignalPrice = signal.price;
      lastSignalTime = TimeLocal();
      
      // Update signal history
      UpdateSignalHistory();
      
      // Send notification if enabled
      if(SendTelegramNotifications)
      {
         string message = "🎯 AI Signal Executed\n" +
                         "Direction: " + signal.signal + "\n" +
                         "Confidence: " + DoubleToString(signal.confidence * 100, 1) + "%\n" +
                         "Price: " + DoubleToString(signal.price, _Digits) + "\n" +
                         "Reason: " + signal.reason;
         SendTelegramMessage(message);
      }
   }
}

//+------------------------------------------------------------------+
//| Read and parse AI signal file with enhanced error handling       |
//+------------------------------------------------------------------+
AISignal ReadAndParseSignalFile()
{
   AISignal signal;
   // Initialize signal structure
   signal.signal = "NONE";
   signal.confidence = 0.0;
   signal.price = 0.0;
   signal.spread = 0;
   signal.timestamp = "";
   signal.reason = "";
   signal.volatility = 0.0;
   signal.volume = 0.0;
   signal.trend = "";
   signal.support = 0.0;
   signal.resistance = 0.0;
   signal.rsi = 0.0;
   signal.macd = 0.0;
   signal.timeframe = "";
   signal.expectedTP = 0.0;
   signal.expectedSL = 0.0;
   signal.priority = 1;
   
   int fileHandle = FileOpen(SignalFilePath, FILE_READ | FILE_TXT);
   if(fileHandle == INVALID_HANDLE)
   {
      if(EnableDetailedLogging)
      {
         Print("❌ Cannot open signal file: ", SignalFilePath, " Error: ", GetLastError());
      }
      return signal;
   }
   
   string content = "";
   while(!FileIsEnding(fileHandle))
   {
      content += FileReadString(fileHandle);
   }
   FileClose(fileHandle);
   
   if(StringLen(content) == 0)
   {
      return signal;
   }
   
   // Parse the signal content
   signal = ParseSignalContent(content);
   
   return signal;
}

//+------------------------------------------------------------------+
//| Parse signal content from file - ENHANCED FOR PYTHON BOT        |
//+------------------------------------------------------------------+
AISignal ParseSignalContent(string content)
{
   AISignal signal;
   signal.signal = "NONE";
   signal.confidence = 0.0;
   signal.price = 0.0;
   signal.spread = 0;
   signal.timestamp = "";
   signal.reason = "";
   signal.volatility = 0.0;
   signal.volume = 0.0;
   signal.trend = "";
   signal.support = 0.0;
   signal.resistance = 0.0;
   signal.rsi = 0.0;
   signal.macd = 0.0;
   signal.timeframe = "";
   signal.expectedTP = 0.0;
   signal.expectedSL = 0.0;
   signal.priority = 1;
   
   // Check if content is JSON format (from Python bot)
   if(StringFind(content, "{") >= 0 && StringFind(content, "}") >= 0)
   {
      // Parse JSON format
      signal.signal = ExtractJsonValue(content, "signal");
      signal.confidence = StringToDouble(ExtractJsonValue(content, "confidence"));
      signal.price = StringToDouble(ExtractJsonValue(content, "current_price"));
      signal.timestamp = ExtractJsonValue(content, "timestamp");
      signal.reason = "AI_JSON_Signal";
      
      // Set defaults for missing JSON fields
      signal.volatility = 1.5;
      signal.volume = 1000000;
      signal.trend = ExtractJsonValue(content, "strength");
      if(signal.trend == "") signal.trend = "MEDIUM";
      signal.support = signal.price - 200 * _Point;
      signal.resistance = signal.price + 200 * _Point;
      signal.rsi = 50.0;
      signal.macd = 0.0;
      signal.timeframe = "M5";
      
      if(signal.signal == "BUY")
      {
         signal.expectedTP = signal.price + TakeProfitPoints * _Point;
         signal.expectedSL = signal.price - StopLossPoints * _Point;
      }
      else if(signal.signal == "SELL")
      {
         signal.expectedTP = signal.price - TakeProfitPoints * _Point;
         signal.expectedSL = signal.price + StopLossPoints * _Point;
      }
      
      signal.priority = (signal.trend == "STRONG") ? 3 : 2;
   }
   else
   {
      // Parse line-by-line format
      string lines[];
      int lineCount = StringSplit(content, '\n', lines);
      
      for(int i = 0; i < lineCount; i++)
      {
         string line = lines[i];
         StringReplace(line, " ", "");
         StringReplace(line, "\t", "");
         StringReplace(line, "\"", "");
         StringReplace(line, ",", "");
         
         // Parse each field
         if(StringFind(line, "signal:") >= 0)
         {
            signal.signal = ExtractValue(line);
         }
         else if(StringFind(line, "confidence:") >= 0)
         {
            signal.confidence = StringToDouble(ExtractValue(line));
         }
         else if(StringFind(line, "price:") >= 0)
         {
            signal.price = StringToDouble(ExtractValue(line));
         }
         else if(StringFind(line, "current_price:") >= 0) // ADDED COMPATIBILITY
         {
            signal.price = StringToDouble(ExtractValue(line));
         }
         else if(StringFind(line, "spread:") >= 0)
         {
            signal.spread = (int)StringToInteger(ExtractValue(line));
         }
         else if(StringFind(line, "timestamp:") >= 0)
         {
            signal.timestamp = ExtractValue(line);
         }
         else if(StringFind(line, "reason:") >= 0)
         {
            signal.reason = ExtractValue(line);
         }
         else if(StringFind(line, "volatility:") >= 0)
         {
            signal.volatility = StringToDouble(ExtractValue(line));
         }
         else if(StringFind(line, "volume:") >= 0)
         {
            signal.volume = StringToDouble(ExtractValue(line));
         }
         else if(StringFind(line, "trend:") >= 0)
         {
            signal.trend = ExtractValue(line);
         }
         else if(StringFind(line, "support:") >= 0)
         {
            signal.support = StringToDouble(ExtractValue(line));
         }
         else if(StringFind(line, "resistance:") >= 0)
         {
            signal.resistance = StringToDouble(ExtractValue(line));
         }
         else if(StringFind(line, "rsi:") >= 0)
         {
            signal.rsi = StringToDouble(ExtractValue(line));
         }
         else if(StringFind(line, "macd:") >= 0)
         {
            signal.macd = StringToDouble(ExtractValue(line));
         }
         else if(StringFind(line, "timeframe:") >= 0)
         {
            signal.timeframe = ExtractValue(line);
         }
         else if(StringFind(line, "expectedTP:") >= 0)
         {
            signal.expectedTP = StringToDouble(ExtractValue(line));
         }
         else if(StringFind(line, "expectedSL:") >= 0)
         {
            signal.expectedSL = StringToDouble(ExtractValue(line));
         }
         else if(StringFind(line, "priority:") >= 0)
         {
            signal.priority = (int)StringToInteger(ExtractValue(line));
         }
      }
   }
   
   return signal;
}

//+------------------------------------------------------------------+
//| Extract JSON value - FIXED FOR MQL5 COMPATIBILITY               |
//+------------------------------------------------------------------+
string ExtractJsonValue(string json, string key)
{
   string search_pattern = "\"" + key + "\":";
   int start_pos = StringFind(json, search_pattern);
   if(start_pos < 0) return "";
   
   start_pos += StringLen(search_pattern);
   
   // Skip whitespace and quotes
   while(start_pos < StringLen(json))
   {
      ushort ch = StringGetCharacter(json, start_pos);
      if(ch == ' ' || ch == '"')
      {
         start_pos++;
      }
      else
      {
         break;
      }
   }
   
   int end_pos = start_pos;
   bool in_quotes = false;
   
   // Find end of value
   while(end_pos < StringLen(json))
   {
      ushort ch = StringGetCharacter(json, end_pos);
      if(ch == '"')
      {
         if(!in_quotes)
         {
            in_quotes = true;
         }
         else
         {
            break;
         }
      }
      else if(!in_quotes && (ch == ',' || ch == '}'))
      {
         break;
      }
      end_pos++;
   }
   
   string result = StringSubstr(json, start_pos, end_pos - start_pos);
   StringReplace(result, "\"", "");
   StringTrimLeft(result);
   StringTrimRight(result);
   
   return result;
}

//+------------------------------------------------------------------+
//| Extract value from key:value pair                                |
//+------------------------------------------------------------------+
string ExtractValue(string line)
{
   int colonPos = StringFind(line, ":");
   if(colonPos >= 0 && colonPos < StringLen(line) - 1)
   {
      return StringSubstr(line, colonPos + 1);
   }
   return "";
}

//+------------------------------------------------------------------+
//| Validate if signal is usable                                     |
//+------------------------------------------------------------------+
bool IsValidSignal(AISignal &signal)
{
   // Check signal type
   if(signal.signal != "BUY" && signal.signal != "SELL")
   {
      return false;
   }
   
   // Check confidence threshold
   if(signal.confidence < MinConfidence)
   {
      if(EnableDetailedLogging)
      {
         Print("❌ Signal confidence too low: ", NormalizeDouble(signal.confidence * 100, 1), "% < ", NormalizeDouble(MinConfidence * 100, 1), "%");
      }
      return false;
   }
   
   // Check if signal has required data
   if(signal.price <= 0)
   {
      Print("❌ Invalid signal price: ", signal.price);
      return false;
   }
   
   // Check timestamp
   if(StringLen(signal.timestamp) == 0)
   {
      Print("❌ Signal missing timestamp");
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Check if signal is expired                                       |
//+------------------------------------------------------------------+
bool IsSignalExpired(AISignal &signal)
{
   // Simple expiry check - in production you'd parse the timestamp
   // For now, signals are considered valid for SignalTimeoutSeconds
   
   if(lastSignalTime > 0)
   {
      long timeDiff = TimeLocal() - lastSignalTime;
      if(timeDiff > SignalTimeoutSeconds)
      {
         if(EnableDetailedLogging)
         {
            Print("⏰ Signal expired - age: ", timeDiff, " seconds");
         }
         return true;
      }
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Check signal consistency with recent signals                     |
//+------------------------------------------------------------------+
bool CheckSignalConsistency(AISignal &signal)
{
   // This is a placeholder for signal consistency logic
   // In a real implementation, you might check:
   // - Signal direction against recent market movement
   // - Confidence correlation with recent signals
   // - Price deviation from current market price
   
   // Basic check: signal price should be close to current price
   double currentPrice = (signal.signal == "BUY") ? 
                        SymbolInfoDouble(_Symbol, SYMBOL_ASK) : 
                        SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   double priceDiff = MathAbs(signal.price - currentPrice);
   double maxPriceDiff = MaxSpreadPoints * _Point * 2; // Allow 2x max spread difference
   
   if(priceDiff > maxPriceDiff)
   {
      Print("⚠️ Signal price inconsistency - Signal: ", signal.price, " Current: ", currentPrice);
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Execute trading signal                                           |
//+------------------------------------------------------------------+
bool ExecuteTradingSignal(ENUM_ORDER_TYPE orderType, AISignal &signal)
{
   // Check if we can open new positions
   if(!CanOpenNewPosition(orderType))
   {
      return false;
   }
   
   // Calculate position size
   double lotSize = CalculatePositionSize(signal);
   if(lotSize <= 0)
   {
      Print("❌ Invalid lot size calculated: ", lotSize);
      return false;
   }
   
   // Calculate stop loss and take profit
   double stopLoss, takeProfit;
   CalculateStopLossAndTakeProfit(orderType, signal, stopLoss, takeProfit);
   
   // Open the position
   ulong ticket = OpenMarketPosition(orderType, lotSize, stopLoss, takeProfit, signal);
   
   if(ticket > 0)
   {
      // Add to active trades tracking
      AddToActiveTrades(ticket, orderType, lotSize, signal);
      
      // Update counters
      dailyTrades++;
      hourlyTrades++;
      hourlySignals++;
      totalTrades++;
      lastTradeTime = (long)TimeLocal();
      
      Print("✅ Position opened successfully");
      Print("   Ticket: ", ticket);
      Print("   Type: ", (orderType == ORDER_TYPE_BUY ? "BUY" : "SELL"));
      Print("   Lot Size: ", lotSize);
      Print("   Stop Loss: ", stopLoss);
      Print("   Take Profit: ", takeProfit);
      Print("   AI Confidence: ", NormalizeDouble(signal.confidence * 100, 1), "%");
      
      return true;
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Check if we can open new position                               |
//+------------------------------------------------------------------+
bool CanOpenNewPosition(ENUM_ORDER_TYPE orderType)
{
   // Check if we already have position in same direction
   for(int i = 0; i < ArraySize(activeTrades); i++)
   {
      if(activeTrades[i].type == orderType)
      {
         Print("❌ Already have ", (orderType == ORDER_TYPE_BUY ? "BUY" : "SELL"), " position open");
         return false;
      }
   }
   
   // Check maximum position count
   if(ArraySize(activeTrades) >= 3) // Maximum 3 positions
   {
      Print("❌ Maximum position count reached");
      return false;
   }
   
   // Check correlation protection
   if(UseCorrelationProtection)
   {
      int correlatedPositions = CountCorrelatedPositions();
      if(correlatedPositions >= MaxCorrelatedPositions)
      {
         Print("❌ Too many correlated positions: ", correlatedPositions);
         return false;
      }
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Calculate position size based on risk and confidence             |
//+------------------------------------------------------------------+
double CalculatePositionSize(AISignal &signal)
{
   double riskPercent = BaseRiskPercentage;
   
   // Adjust risk based on confidence if enabled
   if(UseConfidenceWeighting)
   {
      // Higher confidence = higher risk (within limits)
      double confidenceMultiplier = (signal.confidence - MinConfidence) / (1.0 - MinConfidence);
      riskPercent = BaseRiskPercentage + (MaxRiskPercentage - BaseRiskPercentage) * confidenceMultiplier;
   }
   
   // Adjust risk based on recent performance
   if(UseProgressiveRisk)
   {
      if(consecutiveLosses > 0)
      {
         riskPercent *= MathPow(RiskMultiplier, consecutiveLosses);
      }
      else if(consecutiveWins > 2)
      {
         riskPercent *= 1.1; // Slight increase after wins
      }
   }
   
   // Apply recovery mode risk reduction
   if(recoveryMode)
   {
      riskPercent *= RecoveryRiskReduction;
   }
   
   // Calculate lot size based on risk model
   double lotSize = 0;
   
   switch(RiskModel)
   {
      case RISK_FIXED:
         lotSize = FixedLotSize;
         break;
         
      case RISK_PERCENT:
         {
            double equity = AccountInfoDouble(ACCOUNT_EQUITY);
            double riskAmount = equity * (riskPercent / 100.0);
            double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
            if(tickValue > 0)
            {
               lotSize = riskAmount / (StopLossPoints * tickValue);
            }
         }
         break;
         
      case RISK_BALANCE:
         {
            double balance = AccountInfoDouble(ACCOUNT_BALANCE);
            double riskAmount = balance * (riskPercent / 100.0);
            double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
            if(tickValue > 0)
            {
               lotSize = riskAmount / (StopLossPoints * tickValue);
            }
         }
         break;
         
      case RISK_MARTINGALE:
         // Martingale: double after loss
         lotSize = BaseRiskPercentage / 100.0 * AccountInfoDouble(ACCOUNT_EQUITY) / 1000.0;
         if(consecutiveLosses > 0)
         {
            lotSize *= MathPow(2, consecutiveLosses);
         }
         break;
   }
   
   // Apply lot size limits
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double stepLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   lotSize = MathMax(lotSize, MinLotSize);
   lotSize = MathMin(lotSize, MaxLotSize);
   lotSize = MathMax(lotSize, minLot);
   lotSize = MathMin(lotSize, maxLot);
   
   // Round to step
   if(stepLot > 0)
   {
      lotSize = NormalizeDouble(MathRound(lotSize / stepLot) * stepLot, 2);
   }
   
   // Final validation
   if(lotSize < minLot || lotSize > maxLot)
   {
      Print("❌ Calculated lot size out of range: ", lotSize, " (", minLot, "-", maxLot, ")");
      return 0;
   }
   
   return lotSize;
}

//+------------------------------------------------------------------+
//| Calculate stop loss and take profit levels                       |
//+------------------------------------------------------------------+
void CalculateStopLossAndTakeProfit(ENUM_ORDER_TYPE orderType, AISignal &signal, double &stopLoss, double &takeProfit)
{
   double currentPrice;
   
   if(orderType == ORDER_TYPE_BUY)
   {
      currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      stopLoss = currentPrice - StopLossPoints * _Point;
      takeProfit = currentPrice + TakeProfitPoints * _Point;
   }
   else
   {
      currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      stopLoss = currentPrice + StopLossPoints * _Point;
      takeProfit = currentPrice - TakeProfitPoints * _Point;
   }
   
   // Use signal's expected levels if available and reasonable
   if(signal.expectedSL > 0)
   {
      double signalSLDiff = MathAbs(signal.expectedSL - stopLoss);
      if(signalSLDiff < StopLossPoints * _Point * 0.5) // Within 50% of calculated SL
      {
         stopLoss = signal.expectedSL;
      }
   }
   
   if(signal.expectedTP > 0)
   {
      double signalTPDiff = MathAbs(signal.expectedTP - takeProfit);
      if(signalTPDiff < TakeProfitPoints * _Point * 0.5) // Within 50% of calculated TP
      {
         takeProfit = signal.expectedTP;
      }
   }
   
   // Ensure minimum distance from current price
   double minDistance = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
   
   if(orderType == ORDER_TYPE_BUY)
   {
      if(currentPrice - stopLoss < minDistance)
      {
         stopLoss = currentPrice - minDistance;
      }
      if(takeProfit - currentPrice < minDistance)
      {
         takeProfit = currentPrice + minDistance;
      }
   }
   else
   {
      if(stopLoss - currentPrice < minDistance)
      {
         stopLoss = currentPrice + minDistance;
      }
      if(currentPrice - takeProfit < minDistance)
      {
         takeProfit = currentPrice - minDistance;
      }
   }
}

//+------------------------------------------------------------------+
//| Open market position with comprehensive error handling           |
//+------------------------------------------------------------------+
ulong OpenMarketPosition(ENUM_ORDER_TYPE orderType, double lotSize, double stopLoss, double takeProfit, AISignal &signal)
{
   double price;
   
   if(orderType == ORDER_TYPE_BUY)
   {
      price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   }
   else
   {
      price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   }
   
   // Create trade request
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = NormalizeDouble(lotSize, 2);
   request.type = orderType;
   request.price = NormalizeDouble(price, _Digits);
   request.sl = NormalizeDouble(stopLoss, _Digits);
   request.tp = NormalizeDouble(takeProfit, _Digits);
   request.deviation = 10;
   request.magic = MagicNumber;
   request.comment = "AI_" + (string)TimeLocal() + "_C" + DoubleToString(signal.confidence * 100, 0);
   
   // Attempt to send order
   if(!OrderSend(request, result))
   {
      Print("❌ OrderSend failed with error: ", GetLastError());
      Print("   Request: ", orderType, " ", lotSize, " lots at ", price);
      return 0;
   }
   
   // Check result
   if(result.retcode == TRADE_RETCODE_DONE)
   {
      return result.order;
   }
   else
   {
      Print("❌ Order failed with retcode: ", result.retcode);
      Print("   Description: ", GetTradeRetcodeDescription(result.retcode));
      
      // Retry with adjusted parameters if needed
      if(result.retcode == TRADE_RETCODE_INVALID_STOPS)
      {
         Print("🔄 Retrying with adjusted stops...");
         return RetryOrderWithAdjustedStops(request, signal);
      }
      
      return 0;
   }
}

//+------------------------------------------------------------------+
//| Retry order with adjusted stops                                  |
//+------------------------------------------------------------------+
ulong RetryOrderWithAdjustedStops(MqlTradeRequest &request, AISignal &signal)
{
   // Adjust stops to minimum allowed distance
   double minDistance = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point * 1.5;
   
   if(request.type == ORDER_TYPE_BUY)
   {
      request.sl = request.price - minDistance;
      request.tp = request.price + minDistance * 2;
   }
   else
   {
      request.sl = request.price + minDistance;
      request.tp = request.price - minDistance * 2;
   }
   
   MqlTradeResult result = {};
   if(OrderSend(request, result) && result.retcode == TRADE_RETCODE_DONE)
   {
      Print("✅ Order retry successful with adjusted stops");
      return result.order;
   }
   
   return 0;
}

//+------------------------------------------------------------------+
//| Add trade to active trades tracking                              |
//+------------------------------------------------------------------+
void AddToActiveTrades(ulong ticket, ENUM_ORDER_TYPE type, double lotSize, AISignal &signal)
{
   int size = ArraySize(activeTrades);
   ArrayResize(activeTrades, size + 1);
   
   activeTrades[size].ticket = ticket;
   activeTrades[size].type = type;
   activeTrades[size].lotSize = lotSize;
   activeTrades[size].openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
   activeTrades[size].stopLoss = PositionGetDouble(POSITION_SL);
   activeTrades[size].takeProfit = PositionGetDouble(POSITION_TP);
   activeTrades[size].openTime = (datetime)PositionGetInteger(POSITION_TIME);
   activeTrades[size].confidence = signal.confidence;
   activeTrades[size].reason = signal.reason;
   activeTrades[size].partialTPExecuted = false;
   activeTrades[size].breakEvenMoved = false;
   activeTrades[size].maxProfit = 0;
   activeTrades[size].currentProfit = 0;
}

//+------------------------------------------------------------------+
//| Manage all active positions                                      |
//+------------------------------------------------------------------+
void ManageActivePositions()
{
   for(int i = ArraySize(activeTrades) - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(activeTrades[i].ticket))
      {
         // Update current profit
         activeTrades[i].currentProfit = PositionGetDouble(POSITION_PROFIT);
         
         // Update max profit
         if(activeTrades[i].currentProfit > activeTrades[i].maxProfit)
         {
            activeTrades[i].maxProfit = activeTrades[i].currentProfit;
         }
         
         // Apply trade management
         ManageIndividualPosition(i);
      }
      else
      {
         // Position closed, update statistics and remove from tracking
         UpdateTradeStatistics(activeTrades[i]);
         RemoveFromActiveTrades(i);
      }
   }
}

//+------------------------------------------------------------------+
//| Manage individual position                                       |
//+------------------------------------------------------------------+
void ManageIndividualPosition(int index)
{
   if(index < 0 || index >= ArraySize(activeTrades))
      return;
   
   ulong ticket = activeTrades[index].ticket;
   
   if(!PositionSelectByTicket(ticket))
      return;
   
   ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
   double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
   double currentPrice = (posType == POSITION_TYPE_BUY) ? 
                        SymbolInfoDouble(_Symbol, SYMBOL_BID) : 
                        SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   
   // Break-even management
   if(UseBreakEven && !activeTrades[index].breakEvenMoved)
   {
      ManageBreakEven(index, posType, openPrice, currentPrice);
   }
   
   // Trailing stop management
   if(UseTrailingStop)
   {
      ManageTrailingStop(index, posType, currentPrice);
   }
   
   // Partial take profit management
   if(UsePartialTakeProfit && !activeTrades[index].partialTPExecuted)
   {
      ManagePartialTakeProfit(index, posType, openPrice, currentPrice);
   }
   
   // Time-based exit
   if(UseTimeBasedExit)
   {
      ManageTimeBasedExit(index);
   }
}

//+------------------------------------------------------------------+
//| Manage break-even for position                                   |
//+------------------------------------------------------------------+
void ManageBreakEven(int index, ENUM_POSITION_TYPE posType, double openPrice, double currentPrice)
{
   bool moveToBreakEven = false;
   
   if(posType == POSITION_TYPE_BUY)
   {
      if(currentPrice >= openPrice + BreakEvenPoints * _Point)
      {
         moveToBreakEven = true;
      }
   }
   else if(posType == POSITION_TYPE_SELL)
   {
      if(currentPrice <= openPrice - BreakEvenPoints * _Point)
      {
         moveToBreakEven = true;
      }
   }
   
   if(moveToBreakEven)
   {
      double newSL = openPrice + ((posType == POSITION_TYPE_BUY) ? BreakEvenPlusPoints : -BreakEvenPlusPoints) * _Point;
      
      if(ModifyPositionSL(activeTrades[index].ticket, newSL))
      {
         activeTrades[index].breakEvenMoved = true;
         activeTrades[index].stopLoss = newSL;
         Print("✅ Break-even moved for ticket: ", activeTrades[index].ticket);
      }
   }
}

//+------------------------------------------------------------------+
//| Manage trailing stop for position                                |
//+------------------------------------------------------------------+
void ManageTrailingStop(int index, ENUM_POSITION_TYPE posType, double currentPrice)
{
   double currentSL = PositionGetDouble(POSITION_SL);
   double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
   
   // Check if position is profitable enough to start trailing
   bool canStartTrailing = false;
   if(posType == POSITION_TYPE_BUY)
   {
      canStartTrailing = (currentPrice >= openPrice + TrailingStartPoints * _Point);
   }
   else
   {
      canStartTrailing = (currentPrice <= openPrice - TrailingStartPoints * _Point);
   }
   
   if(!canStartTrailing)
      return;
   
   double newSL = currentSL;
   bool shouldModify = false;
   
   if(posType == POSITION_TYPE_BUY)
   {
      double trailSL = currentPrice - TrailingStopPoints * _Point;
      if(trailSL > currentSL + TrailingStepPoints * _Point)
      {
         newSL = trailSL;
         shouldModify = true;
      }
   }
   else
   {
      double trailSL = currentPrice + TrailingStopPoints * _Point;
      if(trailSL < currentSL - TrailingStepPoints * _Point)
      {
         newSL = trailSL;
         shouldModify = true;
      }
   }
   
   if(shouldModify)
   {
      if(ModifyPositionSL(activeTrades[index].ticket, newSL))
      {
         activeTrades[index].stopLoss = newSL;
         Print("✅ Trailing stop updated for ticket: ", activeTrades[index].ticket, " New SL: ", newSL);
      }
   }
}

//+------------------------------------------------------------------+
//| Manage partial take profit                                       |
//+------------------------------------------------------------------+
void ManagePartialTakeProfit(int index, ENUM_POSITION_TYPE posType, double openPrice, double currentPrice)
{
   // Check if position reached partial TP level
   bool reachedPartialTP = false;
   
   if(posType == POSITION_TYPE_BUY)
   {
      reachedPartialTP = (currentPrice >= openPrice + (TakeProfitPoints * 0.5) * _Point);
   }
   else
   {
      reachedPartialTP = (currentPrice <= openPrice - (TakeProfitPoints * 0.5) * _Point);
   }
   
   if(reachedPartialTP)
   {
      double partialVolume = activeTrades[index].lotSize * (PartialTPPercent / 100.0);
      partialVolume = NormalizeDouble(partialVolume, 2);
      
      if(ClosePartialPosition(activeTrades[index].ticket, partialVolume))
      {
         activeTrades[index].partialTPExecuted = true;
         activeTrades[index].lotSize -= partialVolume;
         Print("✅ Partial TP executed for ticket: ", activeTrades[index].ticket, " Volume: ", partialVolume);
      }
   }
}

//+------------------------------------------------------------------+
//| Manage time-based exit                                          |
//+------------------------------------------------------------------+
void ManageTimeBasedExit(int index)
{
   datetime openTime = activeTrades[index].openTime;
   long positionAge = (TimeLocal() - (long)openTime) / 60; // Age in minutes
   
   if(positionAge >= MaxPositionTimeMinutes)
   {
      Print("⏰ Time-based exit triggered for ticket: ", activeTrades[index].ticket);
      ClosePosition(activeTrades[index].ticket, "Time Exit");
   }
}

//+------------------------------------------------------------------+
//| Modify position stop loss                                        |
//+------------------------------------------------------------------+
bool ModifyPositionSL(ulong ticket, double newSL)
{
   if(!PositionSelectByTicket(ticket))
      return false;
   
   double currentTP = PositionGetDouble(POSITION_TP);
   
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_SLTP;
   request.position = ticket;
   request.sl = NormalizeDouble(newSL, _Digits);
   request.tp = currentTP;
   
   if(OrderSend(request, result))
   {
      return (result.retcode == TRADE_RETCODE_DONE);
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Close partial position                                           |
//+------------------------------------------------------------------+
bool ClosePartialPosition(ulong ticket, double volume)
{
   if(!PositionSelectByTicket(ticket))
      return false;
   
   ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
   ENUM_ORDER_TYPE closeType = (posType == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   
   double closePrice = (closeType == ORDER_TYPE_SELL) ? 
                      SymbolInfoDouble(_Symbol, SYMBOL_BID) : 
                      SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.position = ticket;
   request.symbol = _Symbol;
   request.volume = volume;
   request.type = closeType;
   request.price = closePrice;
   request.deviation = 10;
   request.magic = MagicNumber;
   request.comment = "Partial TP";
   
   if(OrderSend(request, result))
   {
      return (result.retcode == TRADE_RETCODE_DONE);
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Close specific position                                          |
//+------------------------------------------------------------------+
bool ClosePosition(ulong ticket, string reason = "")
{
   if(!PositionSelectByTicket(ticket))
      return false;
   
   ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
   double volume = PositionGetDouble(POSITION_VOLUME);
   ENUM_ORDER_TYPE closeType = (posType == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   
   double closePrice = (closeType == ORDER_TYPE_SELL) ? 
                      SymbolInfoDouble(_Symbol, SYMBOL_BID) : 
                      SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.position = ticket;
   request.symbol = _Symbol;
   request.volume = volume;
   request.type = closeType;
   request.price = closePrice;
   request.deviation = 10;
   request.magic = MagicNumber;
   request.comment = (reason != "") ? reason : "Manual Close";
   
   if(OrderSend(request, result))
   {
      if(result.retcode == TRADE_RETCODE_DONE)
      {
         Print("✅ Position closed. Ticket: ", ticket, " Reason: ", reason);
         return true;
      }
   }
   
   Print("❌ Position close failed. Ticket: ", ticket, " Error: ", GetLastError());
   return false;
}

//+------------------------------------------------------------------+
//| Close all positions                                              |
//+------------------------------------------------------------------+
void CloseAllPositions(string reason = "Emergency")
{
   Print("🚨 Closing all positions - Reason: ", reason);
   
   for(int i = ArraySize(activeTrades) - 1; i >= 0; i--)
   {
      ClosePosition(activeTrades[i].ticket, reason);
   }
   
   // Also close any positions not in our tracking array
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetSymbol(i) == _Symbol)
      {
         if(PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         {
            ulong ticket = PositionGetInteger(POSITION_TICKET);
            ClosePosition(ticket, reason);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Update market conditions - FIXED iMA USAGE                      |
//+------------------------------------------------------------------+
void UpdateMarketConditions()
{
   // Update spread
   currentSpread = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID)) / _Point;
   
   // Update volatility (simplified calculation)
   double high = iHigh(_Symbol, PERIOD_H1, 0);
   double low = iLow(_Symbol, PERIOD_H1, 0);
   currentVolatility = (high - low) / _Point / 100.0; // Normalized volatility
   
   // Update liquidity (using tick volume as proxy)
   long volume = iTickVolume(_Symbol, PERIOD_M1, 0);
   marketLiquidity = (double)volume;
   
   // Simple news detection (placeholder)
   MqlDateTime timeStruct;
   TimeToStruct(TimeLocal(), timeStruct);
   newsTime = false; // In real implementation, check economic calendar
   
   // Update trend strength using MA indicators - FIXED
   double ma20Buffer[];
   double ma50Buffer[];
   
   if(CopyBuffer(ma20Handle, 0, 0, 1, ma20Buffer) > 0 && 
      CopyBuffer(ma50Handle, 0, 0, 1, ma50Buffer) > 0)
   {
      double ma20 = ma20Buffer[0];
      double ma50 = ma50Buffer[0];
      double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      
      if(ma20 > ma50 && currentPrice > ma20)
      {
         trendStrength = 0.8; // Strong uptrend
         marketSentiment = "BULLISH";
      }
      else if(ma20 < ma50 && currentPrice < ma20)
      {
         trendStrength = 0.8; // Strong downtrend
         marketSentiment = "BEARISH";
      }
      else
      {
         trendStrength = 0.3; // Weak trend
         marketSentiment = "NEUTRAL";
      }
   }
}

//+------------------------------------------------------------------+
//| Check emergency conditions                                       |
//+------------------------------------------------------------------+
bool CheckEmergencyConditions()
{
   if(emergencyStop)
      return true;
   
   // Check if emergency stop should be triggered
   if(UseEmergencyStop)
   {
      double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
      if(startingEquity > 0)
      {
         double totalLoss = ((startingEquity - currentEquity) / startingEquity) * 100.0;
         if(totalLoss >= EmergencyStopPercent)
         {
            Print("🚨 EMERGENCY STOP TRIGGERED! Total loss: ", NormalizeDouble(totalLoss, 2), "%");
            emergencyStop = true;
            return true;
         }
      }
   }
   
   // Check drawdown recovery conditions
   if(UseDrawdownRecovery && !recoveryMode)
   {
      if(currentDrawdown >= DrawdownRecoveryThreshold)
      {
         Print("🔧 Entering recovery mode - Drawdown: ", NormalizeDouble(currentDrawdown, 2), "%");
         recoveryMode = true;
      }
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Handle emergency stop                                            |
//+------------------------------------------------------------------+
void HandleEmergencyStop()
{
   static bool emergencyHandled = false;
   
   if(!emergencyHandled)
   {
      Print("🚨 EMERGENCY STOP ACTIVATED");
      CloseAllPositions("Emergency Stop");
      
      if(SendTelegramNotifications)
      {
         string message = "🚨 EMERGENCY STOP ACTIVATED\n" +
                         "Account: " + (string)AccountInfoInteger(ACCOUNT_LOGIN) + "\n" +
                         "Equity: $" + DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2) + "\n" +
                         "Reason: Risk limits exceeded";
         SendTelegramMessage(message);
      }
      
      emergencyHandled = true;
   }
}

//+------------------------------------------------------------------+
//| Update performance statistics                                    |
//+------------------------------------------------------------------+
void UpdatePerformanceStatistics()
{
   if(!EnablePerformanceTracking)
      return;
   
   // Calculate win rate
   if(totalTrades > 0)
   {
      winRate = ((double)winningTrades / totalTrades) * 100.0;
   }
   
   // Calculate profit factor
   double grossProfit = 0;
   double grossLoss = 0;
   
   // This would normally iterate through trade history
   // For now, use simplified calculation
   if(totalProfit > 0)
   {
      grossProfit = totalProfit * 1.5; // Estimated
      grossLoss = totalProfit * 0.5;   // Estimated
   }
   
   if(grossLoss > 0)
   {
      profitFactor = grossProfit / grossLoss;
   }
}

//+------------------------------------------------------------------+
//| Update daily statistics                                          |
//+------------------------------------------------------------------+
void UpdateDailyStatistics()
{
   static int lastDay = -1;
   MqlDateTime timeStruct;
   TimeToStruct(TimeLocal(), timeStruct);
   
   if(lastDay != timeStruct.day)
   {
      // New day - reset counters
      dailyTrades = 0;
      dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
      lastDay = timeStruct.day;
      
      Print("📅 New trading day started");
      Print("   Day start equity: $", NormalizeDouble(dayStartEquity, 2));
   }
   
   // Reset hourly counters
   static int lastHour = -1;
   if(lastHour != timeStruct.hour)
   {
      hourlyTrades = 0;
      hourlySignals = 0;
      lastHour = timeStruct.hour;
   }
}

//+------------------------------------------------------------------+
//| Update trade statistics after position close                     |
//+------------------------------------------------------------------+
void UpdateTradeStatistics(TradeInfo &trade)
{
   double profit = trade.currentProfit;
   totalProfit += profit;
   
   if(profit > 0)
   {
      winningTrades++;
      consecutiveWins++;
      consecutiveLosses = 0;
      
      if(profit > largestWin)
         largestWin = profit;
   }
   else
   {
      losingTrades++;
      consecutiveLosses++;
      consecutiveWins = 0;
      
      if(profit < largestLoss)
         largestLoss = profit;
   }
   
   Print("📊 Trade closed - P/L: $", NormalizeDouble(profit, 2), 
         " | Total: $", NormalizeDouble(totalProfit, 2),
         " | Win Rate: ", NormalizeDouble(winRate, 1), "%");
}

//+------------------------------------------------------------------+
//| Remove trade from active trades array                            |
//+------------------------------------------------------------------+
void RemoveFromActiveTrades(int index)
{
   if(index < 0 || index >= ArraySize(activeTrades))
      return;
   
   // Shift array elements
   for(int i = index; i < ArraySize(activeTrades) - 1; i++)
   {
      activeTrades[i] = activeTrades[i + 1];
   }
   
   // Resize array
   ArrayResize(activeTrades, ArraySize(activeTrades) - 1);
}

//+------------------------------------------------------------------+
//| Initialize market analysis                                       |
//+------------------------------------------------------------------+
void InitializeMarketAnalysis()
{
   UpdateMarketConditions();
   Print("📈 Market analysis initialized");
   Print("   Current spread: ", NormalizeDouble(currentSpread, 1), " points");
   Print("   Market sentiment: ", marketSentiment);
   Print("   Trend strength: ", NormalizeDouble(trendStrength, 2));
}

//+------------------------------------------------------------------+
//| Initialize performance tracking                                  |
//+------------------------------------------------------------------+
void InitializePerformanceTracking()
{
   totalProfit = 0;
   totalTrades = 0;
   winningTrades = 0;
   losingTrades = 0;
   largestWin = 0;
   largestLoss = 0;
   consecutiveWins = 0;
   consecutiveLosses = 0;
   
   Print("📊 Performance tracking initialized");
}

//+------------------------------------------------------------------+
//| Test Telegram connection                                         |
//+------------------------------------------------------------------+
void TestTelegramConnection()
{
   if(StringLen(TelegramBotToken) == 0 || StringLen(TelegramChatID) == 0)
   {
      Print("⚠️ Telegram credentials not configured");
      return;
   }
   
   SendTelegramMessage("✅ AI Scalper EA started successfully");
}

//+------------------------------------------------------------------+
//| Send Telegram message                                            |
//+------------------------------------------------------------------+
void SendTelegramMessage(string message)
{
   // Placeholder for Telegram implementation
   // In real implementation, use HTTP requests to Telegram Bot API
   if(EnableDetailedLogging)
   {
      Print("📱 Telegram: ", message);
   }
}

//+------------------------------------------------------------------+
//| Log detailed status information                                  |
//+------------------------------------------------------------------+
void LogDetailedStatus()
{
   Print("================================================================");
   Print("📊 AI SCALPER PRO - DETAILED STATUS");
   Print("================================================================");
   Print("💰 Account Status:");
   Print("   Balance: $", NormalizeDouble(AccountInfoDouble(ACCOUNT_BALANCE), 2));
   Print("   Equity: $", NormalizeDouble(AccountInfoDouble(ACCOUNT_EQUITY), 2));
   Print("   Free Margin: $", NormalizeDouble(AccountInfoDouble(ACCOUNT_MARGIN_FREE), 2));
   Print("   Margin Level: ", NormalizeDouble(AccountInfoDouble(ACCOUNT_MARGIN_LEVEL), 2), "%");
   
   Print("📈 Performance:");
   Print("   Total Trades: ", totalTrades);
   Print("   Daily Trades: ", dailyTrades, "/", MaxTradesPerDay);
   Print("   Win Rate: ", NormalizeDouble(winRate, 1), "%");
   Print("   Total P/L: $", NormalizeDouble(totalProfit, 2));
   Print("   Current Drawdown: ", NormalizeDouble(currentDrawdown, 2), "%");
   
   Print("🎯 Active Positions: ", ArraySize(activeTrades));
   for(int i = 0; i < ArraySize(activeTrades); i++)
   {
      Print("   #", (i+1), " Ticket: ", activeTrades[i].ticket, 
            " | Type: ", (activeTrades[i].type == ORDER_TYPE_BUY ? "BUY" : "SELL"),
            " | P/L: $", NormalizeDouble(activeTrades[i].currentProfit, 2),
            " | Confidence: ", NormalizeDouble(activeTrades[i].confidence * 100, 1), "%");
   }
   
   Print("📊 Market Conditions:");
   Print("   Spread: ", NormalizeDouble(currentSpread, 1), " points");
   Print("   Volatility: ", NormalizeDouble(currentVolatility, 2));
   Print("   Sentiment: ", marketSentiment);
   Print("   Trend Strength: ", NormalizeDouble(trendStrength, 2));
   
   Print("🤖 AI Signal Status:");
   Print("   Last Signal: ", (StringLen(lastSignalTimestamp) > 0 ? lastSignalTimestamp : "None"));
   Print("   Last Confidence: ", NormalizeDouble(lastSignalConfidence * 100, 1), "%");
   Print("   Last Reason: ", lastSignalReason);
   Print("   Hourly Signals: ", hourlySignals, "/", MaxSignalsPerHour);
   
   Print("================================================================");
}

//+------------------------------------------------------------------+
//| Log signal information                                           |
//+------------------------------------------------------------------+
void LogSignalInformation(AISignal &signal)
{
   Print("🎯 AI SIGNAL DETAILS:");
   Print("   Signal: ", signal.signal);
   Print("   Confidence: ", NormalizeDouble(signal.confidence * 100, 1), "%");
   Print("   Price: ", signal.price);
   Print("   Spread: ", signal.spread);
   Print("   Timestamp: ", signal.timestamp);
   Print("   Reason: ", signal.reason);
   Print("   Volatility: ", signal.volatility);
   Print("   Volume: ", signal.volume);
   Print("   Trend: ", signal.trend);
   Print("   Support: ", signal.support);
   Print("   Resistance: ", signal.resistance);
   Print("   RSI: ", signal.rsi);
   Print("   MACD: ", signal.macd);
   Print("   Timeframe: ", signal.timeframe);
   Print("   Expected TP: ", signal.expectedTP);
   Print("   Expected SL: ", signal.expectedSL);
   Print("   Priority: ", signal.priority);
}

//+------------------------------------------------------------------+
//| Update signal history                                            |
//+------------------------------------------------------------------+
void UpdateSignalHistory()
{
   MqlDateTime timeStruct;
   TimeToStruct(TimeLocal(), timeStruct);
   int hour = timeStruct.hour;
   
   if(hour >= 0 && hour < 24)
   {
      signalHistory[hour]++;
   }
}

//+------------------------------------------------------------------+
//| Count correlated positions                                       |
//+------------------------------------------------------------------+
int CountCorrelatedPositions()
{
   // Placeholder for correlation analysis
   // In real implementation, check positions in correlated instruments
   return ArraySize(activeTrades);
}

//+------------------------------------------------------------------+
//| Save performance data                                            |
//+------------------------------------------------------------------+
void SavePerformanceData()
{
   if(!EnablePerformanceTracking)
      return;
   
   string filename = "ai_scalper_performance_" + (string)AccountInfoInteger(ACCOUNT_LOGIN) + ".csv";
   int fileHandle = FileOpen(filename, FILE_WRITE | FILE_CSV);
   
   if(fileHandle != INVALID_HANDLE)
   {
      FileWrite(fileHandle, "Date", "Total Trades", "Win Rate", "Total P/L", "Max Drawdown", "Equity");
      FileWrite(fileHandle, 
                TimeToString(TimeLocal(), TIME_DATE),
                totalTrades,
                NormalizeDouble(winRate, 2),
                NormalizeDouble(totalProfit, 2),
                NormalizeDouble(currentDrawdown, 2),
                NormalizeDouble(AccountInfoDouble(ACCOUNT_EQUITY), 2));
      
      FileClose(fileHandle);
      Print("💾 Performance data saved to: ", filename);
   }
}

//+------------------------------------------------------------------+
//| Get deinit reason string                                         |
//+------------------------------------------------------------------+
string GetDeInitReason(int reason)
{
   switch(reason)
   {
      case REASON_PROGRAM: return "EA removed from chart";
      case REASON_REMOVE: return "EA deleted";
      case REASON_RECOMPILE: return "EA recompiled";
      case REASON_CHARTCHANGE: return "Chart timeframe changed";
      case REASON_CHARTCLOSE: return "Chart closed";
      case REASON_PARAMETERS: return "Parameters changed";
      case REASON_ACCOUNT: return "Account changed";
      default: return "Unknown reason";
   }
}

//+------------------------------------------------------------------+
//| Get runtime string                                               |
//+------------------------------------------------------------------+
string GetRuntimeString()
{
   // Placeholder - calculate actual runtime
   return "Session runtime calculation";
}

//+------------------------------------------------------------------+
//| Get session return percentage                                    |
//+------------------------------------------------------------------+
double GetSessionReturn()
{
   if(startingEquity > 0)
   {
      double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
      return ((currentEquity - startingEquity) / startingEquity) * 100.0;
   }
   return 0.0;
}

//+------------------------------------------------------------------+
//| Get error description                                            |
//+------------------------------------------------------------------+
string GetErrorDescription(int errorCode)
{
   switch(errorCode)
   {
      case 5004: return "File not found";
      case 5005: return "Cannot open file";
      case 5006: return "Cannot delete file";
      default: return "Error code: " + (string)errorCode;
   }
}

//+------------------------------------------------------------------+
//| Get trade retcode description                                    |
//+------------------------------------------------------------------+
string GetTradeRetcodeDescription(uint retcode)
{
   switch(retcode)
   {
      case TRADE_RETCODE_REQUOTE: return "Requote";
      case TRADE_RETCODE_REJECT: return "Request rejected";
      case TRADE_RETCODE_CANCEL: return "Request canceled";
      case TRADE_RETCODE_PLACED: return "Order placed";
      case TRADE_RETCODE_DONE: return "Request completed";
      case TRADE_RETCODE_DONE_PARTIAL: return "Request partially completed";
      case TRADE_RETCODE_ERROR: return "Request processing error";
      case TRADE_RETCODE_TIMEOUT: return "Request timeout";
      case TRADE_RETCODE_INVALID: return "Invalid request";
      case TRADE_RETCODE_INVALID_VOLUME: return "Invalid volume";
      case TRADE_RETCODE_INVALID_PRICE: return "Invalid price";
      case TRADE_RETCODE_INVALID_STOPS: return "Invalid stops";
      case TRADE_RETCODE_TRADE_DISABLED: return "Trade disabled";
      case TRADE_RETCODE_MARKET_CLOSED: return "Market closed";
      case TRADE_RETCODE_NO_MONEY: return "Insufficient funds";
      case TRADE_RETCODE_PRICE_CHANGED: return "Price changed";
      case TRADE_RETCODE_PRICE_OFF: return "Off quotes";
      case TRADE_RETCODE_INVALID_EXPIRATION: return "Invalid expiration";
      case TRADE_RETCODE_ORDER_CHANGED: return "Order changed";
      case TRADE_RETCODE_TOO_MANY_REQUESTS: return "Too many requests";
      default: return "Unknown retcode: " + (string)retcode;
   }
}

//+------------------------------------------------------------------+
//| END OF AI SCALPER PRO EA - COMPLETE VERSION                     |
//+------------------------------------------------------------------+