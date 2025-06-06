#!/usr/bin/env python3
"""
Advanced XGBoost Model Training for AI Scalper XAUUSDm System
Handles data collection, feature engineering, model training, and evaluation
"""

import os
import sys
import time
import logging
import warnings
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# Set proper encoding for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Suppress warnings
warnings.filterwarnings('ignore')

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    import MetaTrader5 as mt5
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import talib
    import joblib
    print("[OK] All required libraries imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages:")
    print("pip install MetaTrader5 pandas numpy xgboost scikit-learn talib-binary joblib")
    sys.exit(1)

class AIModelTrainer:
    def __init__(self, config_file="config.json"):
        """Initialize the AI Model Trainer"""
        self.config_file = config_file
        self.model = None
        self.scaler = None
        self.anomaly_detector = None
        self.feature_names = None
        self.mt5_connected = False
        
        self.setup_logging()
        self.load_config()
        self.setup_directories()
        
    def setup_logging(self):
        """Setup logging with proper encoding"""
        os.makedirs("logs", exist_ok=True)
        
        class UTF8StreamHandler(logging.StreamHandler):
            def emit(self, record):
                try:
                    msg = self.format(record)
                    # Replace Unicode characters with ASCII equivalents
                    msg = msg.replace('✓', '[OK]').replace('✗', '[ERROR]').replace('⚡', '[INFO]')
                    msg = msg.replace('🎯', '[TARGET]').replace('📊', '[DATA]').replace('🤖', '[AI]')
                    self.stream.write(msg + self.terminator)
                    self.flush()
                except Exception:
                    self.handleError(record)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/model_training.log', encoding='utf-8'),
                UTF8StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AIModelTrainer')
        
    def load_config(self):
        """Load training configuration"""
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {self.config_file} not found, using defaults")
            self.config = {
                "symbol": "XAUUSDm",
                "training_settings": {
                    "lookback_days": 30,
                    "timeframes": ["M5", "M15", "H1"],
                    "min_data_points": 1000
                }
            }
            
    def setup_directories(self):
        """Create necessary directories"""
        directories = ['models', 'logs', 'data']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            print(f"[OK] Directory ready: {directory}")
            
    def connect_mt5(self):
        """Connect to MetaTrader 5"""
        try:
            self.logger.info("Attempting to connect to MT5...")
            
            if not mt5.initialize():
                error = mt5.last_error()
                self.logger.warning(f"MT5 initialization failed: {error}")
                return False
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.warning("Could not get MT5 account information")
                return False
            
            self.mt5_connected = True
            self.logger.info(f"[OK] Connected to MT5 - Account: {account_info.login}")
            self.logger.info(f"[OK] Server: {account_info.server}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {e}")
            return False
            
    def get_historical_data(self, symbol, timeframe, bars=10000):
        """Get historical data from MT5 or generate simulated data"""
        try:
            if self.mt5_connected:
                # Get real data from MT5
                self.logger.info(f"[DATA] Downloading {symbol} data from MT5...")
                
                # Convert timeframe string to MT5 constant
                tf_map = {
                    "M1": mt5.TIMEFRAME_M1,
                    "M5": mt5.TIMEFRAME_M5,
                    "M15": mt5.TIMEFRAME_M15,
                    "H1": mt5.TIMEFRAME_H1,
                    "H4": mt5.TIMEFRAME_H4,
                    "D1": mt5.TIMEFRAME_D1
                }
                
                mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
                
                # Get rates
                rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
                
                if rates is None or len(rates) == 0:
                    self.logger.warning(f"No data received for {symbol} {timeframe}")
                    return self.generate_simulated_data(bars)
                
                # Convert to DataFrame
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                
                self.logger.info(f"[OK] Downloaded {len(df)} bars of real data")
                return df
                
            else:
                # Generate simulated data
                self.logger.warning("MT5 not connected, generating simulated data...")
                return self.generate_simulated_data(bars)
                
        except Exception as e:
            self.logger.error(f"Data download error: {e}")
            return self.generate_simulated_data(bars)
            
    def generate_simulated_data(self, bars=10000):
        """Generate realistic simulated XAUUSD data"""
        try:
            self.logger.info(f"[DATA] Generating {bars} bars of simulated data...")
            
            # Create time index
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=bars)
            time_index = pd.date_range(start=start_time, end=end_time, freq='5T')[:bars]
            
            # Generate realistic gold price data
            base_price = 2000.0
            volatility = 20.0
            
            # Random walk with trend
            returns = np.random.normal(0, volatility/100, bars)
            trend = np.linspace(-0.1, 0.1, bars)
            
            prices = []
            current_price = base_price
            
            for i in range(bars):
                # Add trend and random movement
                change = (returns[i] + trend[i]) * current_price / 100
                current_price += change
                
                # Keep price in realistic range
                current_price = max(1800, min(2500, current_price))
                prices.append(current_price)
            
            # Create OHLC data
            df = pd.DataFrame(index=time_index[:len(prices)])
            df['close'] = prices
            
            # Generate OHLC from close prices
            for i in range(len(df)):
                close = df['close'].iloc[i]
                volatility_factor = np.random.uniform(0.5, 1.5)
                
                high = close + np.random.uniform(0, volatility_factor * 5)
                low = close - np.random.uniform(0, volatility_factor * 5)
                
                if i == 0:
                    open_price = close + np.random.uniform(-2, 2)
                else:
                    open_price = df['close'].iloc[i-1] + np.random.uniform(-1, 1)
                
                df.loc[df.index[i], 'open'] = open_price
                df.loc[df.index[i], 'high'] = max(open_price, high, close)
                df.loc[df.index[i], 'low'] = min(open_price, low, close)
                df.loc[df.index[i], 'tick_volume'] = np.random.randint(100, 1000)
                df.loc[df.index[i], 'spread'] = np.random.randint(10, 30)
                df.loc[df.index[i], 'real_volume'] = np.random.randint(50, 500)
            
            self.logger.info(f"[OK] Generated {len(df)} bars of simulated data")
            return df
            
        except Exception as e:
            self.logger.error(f"Simulated data generation error: {e}")
            raise
            
    def calculate_technical_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        try:
            self.logger.info("[AI] Calculating technical indicators...")
            
            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'tick_volume']
            for col in required_cols:
                if col not in df.columns:
                    self.logger.error(f"Missing required column: {col}")
                    return df
            
            # Convert to numpy arrays for TA-Lib
            open_prices = df['open'].values.astype(float)
            high_prices = df['high'].values.astype(float)
            low_prices = df['low'].values.astype(float)
            close_prices = df['close'].values.astype(float)
            volume = df['tick_volume'].values.astype(float)
            
            # Moving Averages
            df['sma_10'] = talib.SMA(close_prices, timeperiod=10)
            df['sma_20'] = talib.SMA(close_prices, timeperiod=20)
            df['sma_50'] = talib.SMA(close_prices, timeperiod=50)
            df['ema_10'] = talib.EMA(close_prices, timeperiod=10)
            df['ema_20'] = talib.EMA(close_prices, timeperiod=20)
            
            # Trend Indicators
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close_prices)
            df['adx'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
            df['cci'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
            df['aroon_up'], df['aroon_down'] = talib.AROON(high_prices, low_prices, timeperiod=14)
            
            # Momentum Indicators
            df['rsi'] = talib.RSI(close_prices, timeperiod=14)
            df['stoch_k'], df['stoch_d'] = talib.STOCH(high_prices, low_prices, close_prices)
            df['williams_r'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
            df['momentum'] = talib.MOM(close_prices, timeperiod=10)
            df['roc'] = talib.ROC(close_prices, timeperiod=10)
            
            # Volatility Indicators
            df['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close_prices, timeperiod=20)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (close_prices - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume Indicators
            df['ad'] = talib.AD(high_prices, low_prices, close_prices, volume)
            df['obv'] = talib.OBV(close_prices, volume)
            df['mfi'] = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=14)
            
            # Pattern Recognition
            df['doji'] = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
            df['hammer'] = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
            df['engulfing'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
            
            # Price-based features
            df['price_change'] = close_prices - open_prices
            df['price_range'] = high_prices - low_prices
            df['body_size'] = abs(close_prices - open_prices)
            df['upper_shadow'] = high_prices - np.maximum(open_prices, close_prices)
            df['lower_shadow'] = np.minimum(open_prices, close_prices) - low_prices
            
            # Time-based features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_session_start'] = ((df['hour'] >= 0) & (df['hour'] <= 1)).astype(int)
            df['is_session_end'] = ((df['hour'] >= 22) & (df['hour'] <= 23)).astype(int)
            
            # Market structure features
            df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
            df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
            df['inside_bar'] = ((df['high'] <= df['high'].shift(1)) & 
                               (df['low'] >= df['low'].shift(1))).astype(int)
            
            self.logger.info(f"[OK] Calculated {len([col for col in df.columns if col not in required_cols])} technical indicators")
            return df
            
        except Exception as e:
            self.logger.error(f"Technical indicator calculation error: {e}")
            return df
            
    def create_labels(self, df, lookahead=3, threshold=0.001):
        """Create labels for supervised learning"""
        try:
            self.logger.info("[AI] Creating trading labels...")
            
            # Calculate future returns
            future_close = df['close'].shift(-lookahead)
            current_close = df['close']
            future_return = (future_close - current_close) / current_close
            
            # Create labels based on threshold
            labels = np.where(future_return > threshold, 1,  # BUY
                             np.where(future_return < -threshold, -1,  # SELL
                                     0))  # HOLD
            
            df['label'] = labels
            df['future_return'] = future_return
            
            # Convert to classification labels
            label_map = {-1: 0, 0: 1, 1: 2}  # SELL: 0, HOLD: 1, BUY: 2
            df['target'] = df['label'].map(label_map)
            
            # Count labels
            label_counts = df['target'].value_counts().sort_index()
            self.logger.info(f"[OK] Label distribution - SELL: {label_counts.get(0, 0)}, "
                           f"HOLD: {label_counts.get(1, 0)}, BUY: {label_counts.get(2, 0)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Label creation error: {e}")
            return df
            
    def prepare_training_data(self):
        """Prepare complete training dataset"""
        try:
            self.logger.info("="*50)
            self.logger.info("PREPARING TRAINING DATA")
            self.logger.info("="*50)
            
            # Connect to MT5
            if not self.connect_mt5():
                self.logger.warning("Proceeding with simulated data")
            
            # Get symbol from config
            symbol = self.config.get('symbol', 'XAUUSDm')
            
            # Collect data from multiple timeframes
            timeframes = self.config.get('training_settings', {}).get('timeframes', ['M5', 'M15', 'H1'])
            all_data = []
            
            for tf in timeframes:
                self.logger.info(f"[DATA] Processing {symbol} {tf} data...")
                
                # Get historical data
                df = self.get_historical_data(symbol, tf, bars=5000)
                
                if df is not None and len(df) > 100:
                    # Calculate technical indicators
                    df = self.calculate_technical_indicators(df)
                    
                    # Create labels
                    df = self.create_labels(df)
                    
                    # Add timeframe identifier
                    df['timeframe'] = tf
                    
                    all_data.append(df)
                    self.logger.info(f"[OK] {tf} data prepared: {len(df)} rows")
                else:
                    self.logger.warning(f"Insufficient data for {tf}")
            
            if not all_data:
                raise ValueError("No data available for training")
            
            # Combine all timeframes
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Remove rows with NaN values
            initial_rows = len(combined_df)
            combined_df = combined_df.dropna()
            final_rows = len(combined_df)
            
            self.logger.info(f"[OK] Combined dataset: {final_rows} rows (removed {initial_rows - final_rows} NaN rows)")
            
            # Select features for training
            feature_columns = [col for col in combined_df.columns 
                             if col not in ['label', 'target', 'future_return', 'timeframe', 
                                          'open', 'high', 'low', 'close', 'tick_volume', 
                                          'spread', 'real_volume']]
            
            X = combined_df[feature_columns].values
            y = combined_df['target'].values
            
            # Remove any remaining NaN or infinite values
            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X = X[mask]
            y = y[mask]
            
            self.feature_names = feature_columns
            
            self.logger.info(f"[OK] Final training data: {X.shape[0]} samples, {X.shape[1]} features")
            self.logger.info(f"[OK] Feature columns: {len(feature_columns)}")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Data preparation error: {e}")
            raise
        finally:
            # Close MT5 connection
            if self.mt5_connected:
                mt5.shutdown()
                self.mt5_connected = False
                
    def train_model(self, X, y):
        """Train XGBoost model with cross-validation"""
        try:
            self.logger.info("="*50)
            self.logger.info("TRAINING XGBOOST MODEL")
            self.logger.info("="*50)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.logger.info(f"[OK] Training set: {X_train_scaled.shape[0]} samples")
            self.logger.info(f"[OK] Test set: {X_test_scaled.shape[0]} samples")
            
            # Train XGBoost model
            self.logger.info("[AI] Training XGBoost model...")
            
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
            
            # Train the model
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            self.logger.info(f"[OK] Model training completed!")
            self.logger.info(f"[OK] Test accuracy: {accuracy:.4f}")
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
            self.logger.info(f"[OK] Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Feature importance
            feature_importance = self.model.feature_importances_
            
            if self.feature_names:
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                
                self.logger.info("[OK] Top 10 most important features:")
                for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                    self.logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
            
            # Train anomaly detector
            self.logger.info("[AI] Training anomaly detector...")
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            )
            self.anomaly_detector.fit(X_train_scaled)
            self.logger.info("[OK] Anomaly detector trained")
            
            return accuracy, feature_importance
            
        except Exception as e:
            self.logger.error(f"Model training error: {e}")
            raise
            
    def save_models(self):
        """Save trained models to disk"""
        try:
            self.logger.info("[AI] Saving trained models...")
            
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            # Save XGBoost model
            if self.model is not None:
                model_path = models_dir / "xgboost_model.pkl"
                
                # Try saving as pickle first
                try:
                    with open(model_path, 'wb') as f:
                        pickle.dump(self.model, f)
                    self.logger.info(f"[OK] XGBoost model saved to: {model_path}")
                except Exception as pickle_error:
                    # Fallback to JSON format
                    json_path = models_dir / "xgboost_model.pkl.json"
                    self.model.save_model(str(json_path))
                    self.logger.info(f"[OK] XGBoost model saved to: {json_path}")
            
            # Save scaler
            if self.scaler is not None:
                scaler_path = models_dir / "scaler.pkl"
                joblib.dump(self.scaler, scaler_path)
                self.logger.info(f"[OK] Scaler saved to: {scaler_path}")
            
            # Save anomaly detector
            if self.anomaly_detector is not None:
                anomaly_path = models_dir / "anomaly_detector.pkl"
                joblib.dump(self.anomaly_detector, anomaly_path)
                self.logger.info(f"[OK] Anomaly detector saved to: {anomaly_path}")
            
            # Save feature names
            if self.feature_names is not None:
                features_path = models_dir / "feature_names.json"
                with open(features_path, 'w') as f:
                    json.dump(self.feature_names, f, indent=2)
                self.logger.info(f"[OK] Feature names saved to: {features_path}")
            
            # Save model metadata
            metadata = {
                "training_date": datetime.now().isoformat(),
                "model_type": "XGBoost",
                "n_features": len(self.feature_names) if self.feature_names else 0,
                "symbol": self.config.get('symbol', 'XAUUSDm'),
                "version": "2.0"
            }
            
            metadata_path = models_dir / "model_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"[OK] Model metadata saved to: {metadata_path}")
            
            self.logger.info("[OK] All models saved successfully!")
            
        except Exception as e:
            self.logger.error(f"Model saving error: {e}")
            raise
            
    def test_model_prediction(self):
        """Test the trained model with a sample prediction"""
        try:
            self.logger.info("[AI] Testing model prediction...")
            
            if self.model is None or self.scaler is None:
                self.logger.warning("Model not trained yet")
                return
            
            # Create dummy input data
            n_features = len(self.feature_names) if self.feature_names else 50
            dummy_input = np.random.randn(1, n_features)
            
            # Scale the input
            dummy_scaled = self.scaler.transform(dummy_input)
            
            # Make prediction
            prediction = self.model.predict(dummy_scaled)[0]
            probabilities = self.model.predict_proba(dummy_scaled)[0]
            
            # Map prediction to signal
            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            signal = signal_map.get(prediction, 'HOLD')
            confidence = probabilities.max()
            
            self.logger.info(f"[OK] Test prediction: {signal} (confidence: {confidence:.3f})")
            
        except Exception as e:
            self.logger.error(f"Model test error: {e}")
            
    def run_training(self):
        """Run the complete training pipeline"""
        try:
            start_time = time.time()
            
            self.logger.info("="*50)
            self.logger.info("AI SCALPER MODEL TRAINING STARTED")
            self.logger.info("="*50)
            
            # Prepare training data
            X, y = self.prepare_training_data()
            
            # Train model
            accuracy, feature_importance = self.train_model(X, y)
            
            # Save models
            self.save_models()
            
            # Test model
            self.test_model_prediction()
            
            # Training complete
            training_time = time.time() - start_time
            
            self.logger.info("="*50)
            self.logger.info("Training completed successfully!")
            self.logger.info(f"Model accuracy: {accuracy:.4f}")
            self.logger.info(f"Training time: {training_time:.2f} seconds")
            self.logger.info("="*50)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Training pipeline error: {e}")
            return False

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='AI Scalper XGBoost Model Training')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--test', action='store_true', help='Test model after training')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    try:
        # Create trainer
        trainer = AIModelTrainer(args.config)
        
        if args.verbose:
            trainer.logger.setLevel(logging.DEBUG)
        
        # Run training
        success = trainer.run_training()
        
        if success:
            print("\n[OK] Model training completed successfully!")
            print("You can now run the AI Scalper system: python run_ai_system.py")
            return 0
        else:
            print("\n[ERROR] Model training failed!")
            return 1
            
    except Exception as e:
        print(f"\n[ERROR] Training error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)