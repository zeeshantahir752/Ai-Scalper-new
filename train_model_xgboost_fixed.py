# train_model_xgboost_fixed.py - Fixed version with proper path handling

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging

# XGBoost and ML libraries
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.ensemble import IsolationForest
    print("✓ XGBoost and sklearn imported successfully")
except ImportError as e:
    print(f"❌ Error importing ML libraries: {e}")
    print("Run: pip install xgboost scikit-learn")
    sys.exit(1)

# Technical Analysis libraries
try:
    import talib
    print("✓ TA-Lib imported successfully")
except ImportError as e:
    print(f"❌ Error importing TA-Lib: {e}")
    print("Please install TA-Lib following the instructions:")
    print("Windows: Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
    print("Linux: sudo apt-get install libta-lib-dev && pip install TA-Lib")
    print("MacOS: brew install ta-lib && pip install TA-Lib")
    sys.exit(1)

# MetaTrader 5 (optional for training)
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    print("✓ MetaTrader5 imported successfully")
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️ MetaTrader5 not available - using sample data for training")

warnings.filterwarnings('ignore')

class ModelPathManager:
    """Handles all model path operations with proper error checking"""
    
    def __init__(self, base_dir=None):
        if base_dir is None:
            # Get the directory where this script is located
            self.base_dir = Path(__file__).parent.absolute()
        else:
            self.base_dir = Path(base_dir)
        
        # Define all necessary directories
        self.models_dir = self.base_dir / "models"
        self.signals_dir = self.base_dir / "signals"
        self.logs_dir = self.base_dir / "logs"
        self.data_dir = self.base_dir / "data"
        
        # Create directories if they don't exist
        self.create_directories()
        
        # Define model file paths
        self.xgboost_model_path = self.models_dir / "xgboost_model.pkl"
        self.scaler_path = self.models_dir / "scaler.pkl"
        self.anomaly_detector_path = self.models_dir / "anomaly_detector.pkl"
        self.signal_file_path = self.signals_dir / "xau_signal.txt"
        
        print(f"✓ Model directory: {self.models_dir}")
        print(f"✓ Base directory: {self.base_dir}")
    
    def create_directories(self):
        """Create all necessary directories"""
        directories = [self.models_dir, self.signals_dir, self.logs_dir, self.data_dir]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                print(f"✓ Directory created/verified: {directory}")
            except Exception as e:
                print(f"❌ Error creating directory {directory}: {e}")
                raise
    
    def save_model(self, model, model_type="xgboost"):
        """Save model with proper error handling"""
        try:
            if model_type == "xgboost":
                if hasattr(model, 'save_model'):
                    # Native XGBoost save (preferred)
                    model.save_model(str(self.xgboost_model_path).replace('.pkl', '.json'))
                    print(f"✓ XGBoost model saved to: {self.xgboost_model_path}.json")
                else:
                    # Pickle save as fallback
                    with open(self.xgboost_model_path, 'wb') as f:
                        pickle.dump(model, f)
                    print(f"✓ Model pickled to: {self.xgboost_model_path}")
            
            elif model_type == "scaler":
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"✓ Scaler saved to: {self.scaler_path}")
            
            elif model_type == "anomaly_detector":
                with open(self.anomaly_detector_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"✓ Anomaly detector saved to: {self.anomaly_detector_path}")
            
            return True
        except Exception as e:
            print(f"❌ Error saving {model_type} model: {e}")
            return False
    
    def load_model(self, model_type="xgboost"):
        """Load model with proper error handling"""
        try:
            if model_type == "xgboost":
                # Try to load native XGBoost format first
                json_path = str(self.xgboost_model_path).replace('.pkl', '.json')
                if os.path.exists(json_path):
                    model = xgb.XGBClassifier()
                    model.load_model(json_path)
                    print(f"✓ XGBoost model loaded from: {json_path}")
                    return model
                elif os.path.exists(self.xgboost_model_path):
                    with open(self.xgboost_model_path, 'rb') as f:
                        model = pickle.load(f)
                    print(f"✓ Model loaded from: {self.xgboost_model_path}")
                    return model
                else:
                    print(f"❌ No model found at {self.xgboost_model_path}")
                    return None
            
            elif model_type == "scaler":
                if os.path.exists(self.scaler_path):
                    with open(self.scaler_path, 'rb') as f:
                        model = pickle.load(f)
                    print(f"✓ Scaler loaded from: {self.scaler_path}")
                    return model
                else:
                    print(f"❌ No scaler found at {self.scaler_path}")
                    return None
            
            elif model_type == "anomaly_detector":
                if os.path.exists(self.anomaly_detector_path):
                    with open(self.anomaly_detector_path, 'rb') as f:
                        model = pickle.load(f)
                    print(f"✓ Anomaly detector loaded from: {self.anomaly_detector_path}")
                    return model
                else:
                    print(f"❌ No anomaly detector found at {self.anomaly_detector_path}")
                    return None
        
        except Exception as e:
            print(f"❌ Error loading {model_type} model: {e}")
            return None

class DataGenerator:
    """Generate sample data for training when MT5 is not available"""
    
    @staticmethod
    def generate_sample_ohlcv(periods=10000):
        """Generate realistic OHLCV data for XAUUSD"""
        np.random.seed(42)  # For reproducibility
        
        # Start with a base price around 2000 (typical for XAUUSD)
        base_price = 2000.0
        
        # Generate price movements
        returns = np.random.normal(0, 0.002, periods)  # 0.2% daily volatility
        prices = [base_price]
        
        for i in range(1, periods):
            # Add some trending behavior
            trend = 0.0001 * np.sin(i / 100)  # Slight trending
            new_price = prices[-1] * (1 + returns[i] + trend)
            prices.append(max(new_price, 1500))  # Minimum price floor
        
        prices = np.array(prices)
        
        # Generate OHLCV from prices
        data = []
        for i in range(len(prices)):
            if i == 0:
                open_price = close_price = high_price = low_price = prices[i]
            else:
                open_price = prices[i-1]
                close_price = prices[i]
                
                # High and low based on some random variation
                volatility = abs(open_price - close_price) * np.random.uniform(1, 3)
                high_price = max(open_price, close_price) + volatility * np.random.uniform(0, 1)
                low_price = min(open_price, close_price) - volatility * np.random.uniform(0, 1)
            
            volume = np.random.uniform(1000, 10000)
            
            data.append({
                'time': datetime.now() - timedelta(minutes=periods-i),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': int(volume)
            })
        
        return pd.DataFrame(data)

class FeatureEngineering:
    """Generate technical indicators and features"""
    
    @staticmethod
    def add_technical_indicators(df):
        """Add comprehensive technical indicators"""
        df = df.copy()
        
        # Price-based indicators
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
        
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
        
        # ATR
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Volume indicators (if volume is available)
        if 'volume' in df.columns:
            df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price action features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Volatility features
        df['volatility'] = df['price_change'].rolling(window=20).std()
        
        # Trend features
        df['trend_sma'] = np.where(df['close'] > df['sma_20'], 1, -1)
        df['trend_ema'] = np.where(df['ema_12'] > df['ema_26'], 1, -1)
        
        return df
    
    @staticmethod
    def create_target_variable(df, future_periods=5, threshold=0.001):
        """Create target variable for classification"""
        df = df.copy()
        
        # Calculate future return
        df['future_close'] = df['close'].shift(-future_periods)
        df['future_return'] = (df['future_close'] - df['close']) / df['close']
        
        # Create binary target: 1 for significant upward movement, 0 otherwise
        df['target'] = np.where(df['future_return'] > threshold, 1, 0)
        
        # Remove rows with NaN target
        df = df.dropna(subset=['target'])
        
        return df

class AIModelTrainer:
    """Main class for training the AI model"""
    
    def __init__(self, base_dir=None):
        self.path_manager = ModelPathManager(base_dir)
        self.feature_engineer = FeatureEngineering()
        self.model = None
        self.scaler = None
        self.anomaly_detector = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.path_manager.logs_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_training_data(self):
        """Get training data from MT5 or generate sample data"""
        if MT5_AVAILABLE:
            try:
                if not mt5.initialize():
                    raise Exception("MT5 initialization failed")
                
                # Get historical data
                symbol = "XAUUSDm"  # or "XAUUSD" depending on your broker
                timeframe = mt5.TIMEFRAME_M5
                
                # Get data for the last 30 days
                from_date = datetime.now() - timedelta(days=30)
                rates = mt5.copy_rates_from(symbol, timeframe, from_date, 10000)
                
                if rates is None or len(rates) == 0:
                    raise Exception("No data received from MT5")
                
                # Convert to DataFrame
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                
                mt5.shutdown()
                self.logger.info(f"Retrieved {len(df)} records from MT5")
                return df
                
            except Exception as e:
                self.logger.warning(f"Failed to get MT5 data: {e}")
                self.logger.info("Falling back to generated sample data")
                return DataGenerator.generate_sample_ohlcv()
        else:
            self.logger.info("Using generated sample data for training")
            return DataGenerator.generate_sample_ohlcv()
    
    def prepare_features(self, df):
        """Prepare features for training"""
        self.logger.info("Generating technical indicators...")
        
        # Add technical indicators
        df = self.feature_engineer.add_technical_indicators(df)
        
        # Create target variable
        df = self.feature_engineer.create_target_variable(df)
        
        # Select features for training
        feature_columns = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'bb_width', 'bb_position', 'rsi', 'macd', 'macd_signal', 'macd_hist',
            'stoch_k', 'stoch_d', 'atr', 'price_change', 'high_low_ratio',
            'close_open_ratio', 'volatility', 'trend_sma', 'trend_ema'
        ]
        
        # Add volume features if available
        if 'volume_ratio' in df.columns:
            feature_columns.extend(['volume_ratio'])
        
        # Remove any columns that don't exist
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Remove rows with NaN values
        df_clean = df[available_features + ['target']].dropna()
        
        X = df_clean[available_features]
        y = df_clean['target']
        
        self.logger.info(f"Prepared {len(X)} samples with {len(available_features)} features")
        self.logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, available_features
    
    def train_model(self, X, y):
        """Train the XGBoost model"""
        self.logger.info("Starting model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train anomaly detector
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_detector.fit(X_train_scaled)
        
        # Train XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        # Train with early stopping
        eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=eval_set,
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.logger.info(f"Model trained successfully!")
        self.logger.info(f"Test Accuracy: {accuracy:.4f}")
        self.logger.info(f"Best iteration: {self.model.best_iteration}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def save_models(self):
        """Save all trained models"""
        self.logger.info("Saving models...")
        
        success = True
        success &= self.path_manager.save_model(self.model, "xgboost")
        success &= self.path_manager.save_model(self.scaler, "scaler")
        success &= self.path_manager.save_model(self.anomaly_detector, "anomaly_detector")
        
        if success:
            self.logger.info("✓ All models saved successfully!")
        else:
            self.logger.error("❌ Some models failed to save")
        
        return success
    
    def run_training(self):
        """Run the complete training pipeline"""
        try:
            self.logger.info("="*50)
            self.logger.info("Starting AI Model Training Pipeline")
            self.logger.info("="*50)
            
            # Step 1: Get data
            df = self.get_training_data()
            
            # Step 2: Prepare features
            X, y, feature_names = self.prepare_features(df)
            
            # Step 3: Train model
            accuracy = self.train_model(X, y)
            
            # Step 4: Save models
            self.save_models()
            
            # Step 5: Save feature names for later use
            feature_info = {
                'feature_names': feature_names,
                'training_date': datetime.now().isoformat(),
                'accuracy': accuracy,
                'n_samples': len(X)
            }
            
            with open(self.path_manager.models_dir / 'feature_info.json', 'w') as f:
                json.dump(feature_info, f, indent=2)
            
            self.logger.info("="*50)
            self.logger.info("Training completed successfully!")
            self.logger.info(f"Model accuracy: {accuracy:.4f}")
            self.logger.info("="*50)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

def main():
    """Main function to run training"""
    print("🚀 AI Scalper XAUUSD - Model Training")
    print("="*50)
    
    # You can specify a custom base directory here if needed
    # trainer = AIModelTrainer(base_dir="/path/to/your/project")
    trainer = AIModelTrainer()
    
    success = trainer.run_training()
    
    if success:
        print("\n✅ Training completed successfully!")
        print("You can now run the AI signal generator.")
    else:
        print("\n❌ Training failed. Check the logs for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()