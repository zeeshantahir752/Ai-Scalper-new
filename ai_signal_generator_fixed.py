# ai_signal_generator_unicode_fix.py - Fixed Unicode logging issues

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Fix Windows console encoding
if sys.platform.startswith('win'):
    import locale
    try:
        # Try to set UTF-8 encoding
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        pass

# Import our path manager (assuming it's in the same directory)
try:
    from train_model_xgboost_fixed import ModelPathManager, FeatureEngineering
except ImportError:
    print("Error: Make sure train_model_xgboost_fixed.py is in the same directory")
    sys.exit(1)

# XGBoost and ML libraries
try:
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    print("[OK] ML libraries imported successfully")
except ImportError as e:
    print(f"[ERROR] Error importing ML libraries: {e}")
    sys.exit(1)

# Technical Analysis
try:
    import talib
    print("[OK] TA-Lib imported successfully")
except ImportError as e:
    print(f"[ERROR] Error importing TA-Lib: {e}")
    sys.exit(1)

# MetaTrader 5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    print("[OK] MetaTrader5 imported successfully")
except ImportError:
    MT5_AVAILABLE = False
    print("[WARN] MetaTrader5 not available - running in simulation mode")

warnings.filterwarnings('ignore')

class SafeLogger:
    """Unicode-safe logger for Windows compatibility"""
    
    def __init__(self, name, log_file=None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler with safe encoding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Set encoding for console handler
        if hasattr(console_handler.stream, 'reconfigure'):
            try:
                console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
            except:
                pass
        
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            try:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                print(f"[WARN] Could not create file handler: {e}")
    
    def _safe_message(self, msg):
        """Convert Unicode characters to safe ASCII alternatives"""
        replacements = {
            '✓': '[OK]',
            '❌': '[ERROR]',
            '⚠️': '[WARN]',
            '🚀': '[START]',
            '🎯': '[TARGET]',
            '🛑': '[STOP]',
            '🤖': '[AI]',
            '📊': '[DATA]',
            '⚙️': '[CONFIG]'
        }
        
        safe_msg = str(msg)
        for unicode_char, ascii_replacement in replacements.items():
            safe_msg = safe_msg.replace(unicode_char, ascii_replacement)
        
        return safe_msg
    
    def info(self, msg):
        try:
            self.logger.info(self._safe_message(msg))
        except UnicodeEncodeError:
            # Fallback: encode as ASCII with replacement
            safe_msg = str(msg).encode('ascii', errors='replace').decode('ascii')
            self.logger.info(safe_msg)
    
    def error(self, msg):
        try:
            self.logger.error(self._safe_message(msg))
        except UnicodeEncodeError:
            safe_msg = str(msg).encode('ascii', errors='replace').decode('ascii')
            self.logger.error(safe_msg)
    
    def warning(self, msg):
        try:
            self.logger.warning(self._safe_message(msg))
        except UnicodeEncodeError:
            safe_msg = str(msg).encode('ascii', errors='replace').decode('ascii')
            self.logger.warning(safe_msg)

class ConfigManager:
    """Handles configuration loading with proper error handling"""
    
    def __init__(self, config_path=None):
        if config_path is None:
            self.config_path = Path(__file__).parent / "config.json"
        else:
            self.config_path = Path(config_path)
        
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration with defaults"""
        default_config = {
            "trading": {
                "symbol": "XAUUSDm",
                "signal_threshold": 0.70,
                "max_daily_signals": 20,
                "risk_per_trade": 0.02
            },
            "paths": {
                "models_dir": "models",
                "signals_dir": "signals",
                "logs_dir": "logs",
                "signal_file": "signals/xau_signal.txt"
            },
            "model": {
                "min_confidence_threshold": 0.65,
                "anomaly_detection_threshold": 0.1
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                return {**default_config, **loaded_config}
            except Exception as e:
                print(f"[WARN] Error loading config: {e}, using defaults")
                return default_config
        else:
            print(f"[WARN] Config file not found at {self.config_path}, using defaults")
            return default_config
    
    def get(self, key_path, default=None):
        """Get nested configuration value"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value

class MT5Manager:
    """Handles MT5 connection and data retrieval"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.symbol = config.get('trading.symbol', 'XAUUSDm')
        self.is_connected = False
        
    def initialize(self):
        """Initialize MT5 connection"""
        if not MT5_AVAILABLE:
            return False
            
        try:
            if not mt5.initialize():
                self.logger.error("MT5 initialization failed")
                return False
            
            # Check if symbol is available
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                self.logger.error(f"Symbol {self.symbol} not found")
                return False
            
            if not symbol_info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    self.logger.error(f"Failed to select symbol {self.symbol}")
                    return False
            
            self.is_connected = True
            self.logger.info(f"MT5 connected, symbol: {self.symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {e}")
            return False
    
    def get_current_data(self, timeframe=mt5.TIMEFRAME_M5, count=1000):
        """Get current market data"""
        if not self.is_connected:
            return None
            
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, count)
            if rates is None:
                self.logger.error("No data received from MT5")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting MT5 data: {e}")
            return None
    
    def shutdown(self):
        """Shutdown MT5 connection"""
        if MT5_AVAILABLE and self.is_connected:
            mt5.shutdown()
            self.is_connected = False

class AISignalGenerator:
    """Main AI signal generator with Unicode-safe logging"""
    
    def __init__(self, config_path=None, base_dir=None):
        # Load configuration
        self.config = ConfigManager(config_path)
        
        # Setup path manager
        self.path_manager = ModelPathManager(base_dir)
        
        # Setup safe logging
        log_file = self.path_manager.logs_dir / 'ai_signals.log'
        self.logger = SafeLogger(__name__, log_file)
        
        # Setup feature engineering
        self.feature_engineer = FeatureEngineering()
        
        # Initialize MT5 manager
        self.mt5_manager = MT5Manager(self.config, self.logger)
        
        # Model components
        self.model = None
        self.scaler = None
        self.anomaly_detector = None
        self.feature_names = []
        
        # Signal tracking
        self.last_signal_time = None
        self.daily_signal_count = 0
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        self.logger.info("Loading AI models...")
        
        try:
            # Load XGBoost model
            self.model = self.path_manager.load_model("xgboost")
            if self.model is None:
                self.logger.error("Failed to load XGBoost model")
                return False
            
            # Load scaler
            self.scaler = self.path_manager.load_model("scaler")
            if self.scaler is None:
                self.logger.error("Failed to load scaler")
                return False
            
            # Load anomaly detector
            self.anomaly_detector = self.path_manager.load_model("anomaly_detector")
            if self.anomaly_detector is None:
                self.logger.warning("Anomaly detector not found, continuing without it")
            
            # Load feature names
            feature_info_path = self.path_manager.models_dir / 'feature_info.json'
            if feature_info_path.exists():
                with open(feature_info_path, 'r', encoding='utf-8') as f:
                    feature_info = json.load(f)
                    self.feature_names = feature_info.get('feature_names', [])
                    self.logger.info(f"Loaded {len(self.feature_names)} feature names")
            else:
                self.logger.warning("Feature info not found, using default features")
                self.feature_names = self.get_default_features()
            
            self.logger.info("All models loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False
    
    def get_default_features(self):
        """Get default feature names if feature_info.json is missing"""
        return [
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'bb_width', 'bb_position', 'rsi', 'macd', 'macd_signal', 'macd_hist',
            'stoch_k', 'stoch_d', 'atr', 'price_change', 'high_low_ratio',
            'close_open_ratio', 'volatility', 'trend_sma', 'trend_ema'
        ]
    
    def get_market_data(self):
        """Get current market data"""
        if self.mt5_manager.is_connected:
            return self.mt5_manager.get_current_data()
        else:
            # Return simulated data for testing
            self.logger.warning("Using simulated data - MT5 not connected")
            from train_model_xgboost_fixed import DataGenerator
            return DataGenerator.generate_sample_ohlcv(periods=1000)
    
    def prepare_features(self, df):
        """Prepare features for prediction"""
        try:
            # Add technical indicators
            df_with_indicators = self.feature_engineer.add_technical_indicators(df)
            
            # Select only the features used in training
            available_features = []
            for feature in self.feature_names:
                if feature in df_with_indicators.columns:
                    available_features.append(feature)
                else:
                    self.logger.warning(f"Feature {feature} not available in current data")
            
            if len(available_features) == 0:
                self.logger.error("No features available for prediction")
                return None
            
            # Get the latest row of features
            features_df = df_with_indicators[available_features].iloc[-1:].fillna(0)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return None
    
    def generate_signal(self):
        """Generate trading signal"""
        try:
            # Get market data
            df = self.get_market_data()
            if df is None or len(df) < 100:
                self.logger.error("Insufficient market data")
                return None
            
            # Prepare features
            features_df = self.prepare_features(df)
            if features_df is None:
                return None
            
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Check for anomalies
            is_anomaly = False
            if self.anomaly_detector is not None:
                anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
                is_anomaly = anomaly_score < self.config.get('model.anomaly_detection_threshold', 0.1)
            
            # Make prediction
            prediction_proba = self.model.predict_proba(features_scaled)[0]
            prediction = self.model.predict(features_scaled)[0]
            confidence = max(prediction_proba)
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Create signal
            signal = {
                'timestamp': datetime.now().isoformat(),
                'symbol': self.config.get('trading.symbol', 'XAUUSDm'),
                'prediction': int(prediction),
                'confidence': float(confidence),
                'signal': 'BUY' if prediction == 1 and confidence > self.config.get('trading.signal_threshold', 0.70) else 'HOLD',
                'current_price': float(current_price),
                'is_anomaly': bool(is_anomaly),
                'features_count': len(features_df.columns),
                'model_ready': True
            }
            
            # Add signal strength
            if signal['signal'] == 'BUY':
                if confidence > 0.85:
                    signal['strength'] = 'STRONG'
                elif confidence > 0.75:
                    signal['strength'] = 'MEDIUM'
                else:
                    signal['strength'] = 'WEAK'
            else:
                signal['strength'] = 'NONE'
            
            # Log signal with safe characters
            self.logger.info(f"Signal generated: {signal['signal']} | Confidence: {confidence:.3f} | Price: {current_price:.2f}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def save_signal(self, signal):
        """Save signal to file for MT5 EA"""
        try:
            signal_file = self.path_manager.signal_file_path
            
            with open(signal_file, 'w', encoding='utf-8') as f:
                json.dump(signal, f, indent=2)
            
            self.logger.info(f"Signal saved to {signal_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving signal: {e}")
            return False
    
    def run_continuous(self, interval_seconds=30):
        """Run signal generation continuously"""
        self.logger.info("Starting continuous AI signal generation...")
        self.logger.info(f"Update interval: {interval_seconds} seconds")
        
        # Initialize MT5 if available
        if MT5_AVAILABLE:
            self.mt5_manager.initialize()
        
        try:
            while True:
                # Generate signal
                signal = self.generate_signal()
                
                if signal is not None:
                    # Save signal
                    self.save_signal(signal)
                    
                    # Check daily limits
                    if signal['signal'] == 'BUY':
                        self.daily_signal_count += 1
                        max_daily = self.config.get('trading.max_daily_signals', 20)
                        
                        if self.daily_signal_count >= max_daily:
                            self.logger.warning(f"Daily signal limit reached: {max_daily}")
                
                # Wait for next update
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            self.logger.info("Signal generation stopped by user")
        except Exception as e:
            self.logger.error(f"Error in continuous mode: {e}")
        finally:
            # Cleanup
            self.mt5_manager.shutdown()
    
    def run_single(self):
        """Generate a single signal"""
        self.logger.info("Generating single signal...")
        
        # Initialize MT5 if available
        if MT5_AVAILABLE:
            self.mt5_manager.initialize()
        
        try:
            signal = self.generate_signal()
            
            if signal is not None:
                self.save_signal(signal)
                print(f"\n[SUCCESS] Signal generated:")
                print(f"   Action: {signal['signal']}")
                print(f"   Confidence: {signal['confidence']:.3f}")
                print(f"   Strength: {signal['strength']}")
                print(f"   Price: {signal['current_price']:.2f}")
                print(f"   Time: {signal['timestamp']}")
                return True
            else:
                print("[ERROR] Failed to generate signal")
                return False
                
        finally:
            self.mt5_manager.shutdown()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Signal Generator for XAUUSD')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--base-dir', type=str, help='Base directory for the project')
    parser.add_argument('--continuous', action='store_true', help='Run continuously')
    parser.add_argument('--interval', type=int, default=30, help='Update interval in seconds')
    parser.add_argument('--single', action='store_true', help='Generate single signal')
    
    args = parser.parse_args()
    
    print("[AI] AI Scalper XAUUSD - Signal Generator")
    print("=" * 50)
    
    # Create signal generator
    generator = AISignalGenerator(
        config_path=args.config,
        base_dir=args.base_dir
    )
    
    # Check if models are loaded
    if generator.model is None:
        print("[ERROR] Models not loaded. Please run training first:")
        print("   python train_model_xgboost_fixed.py")
        return 1
    
    # Run based on mode
    if args.single:
        success = generator.run_single()
        return 0 if success else 1
    else:
        generator.run_continuous(args.interval)
        return 0

if __name__ == "__main__":
    exit_code = main()