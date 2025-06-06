
#!/usr/bin/env python3
"""
Complete AI Scalper XAUUSDm System with Better Error Handling
"""

import os
import sys
import time
import logging
import signal
import threading
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# Set proper encoding for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Check if required files exist
def check_required_files():
    """Check if all required files exist"""
    required_files = {
        'ai_signal_generator_fixed.py': 'AI Signal Generator',
        'train_model_xgboost_fixed.py': 'Model Trainer'
    }
    
    missing_files = []
    for file, description in required_files.items():
        if not Path(file).exists():
            missing_files.append(f"{file} ({description})")
    
    if missing_files:
        print(f"[ERROR] Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print(f"\n[INFO] Current directory: {os.getcwd()}")
        print(f"[INFO] Python files found: {list(Path('.').glob('*.py'))}")
        return False
    
    return True

# Import required libraries
try:
    import MetaTrader5 as mt5
    import pandas as pd
    import numpy as np
    print("[OK] MetaTrader5 imported successfully")
    print("[OK] Required libraries imported successfully")
except ImportError as e:
    print(f"Library import error: {e}")
    print("Please install requirements: pip install MetaTrader5 pandas numpy")
    sys.exit(1)

# Check required files before importing
if not check_required_files():
    print("\n[FIX] Please ensure all required Python files are in the current directory")
    sys.exit(1)

# Import AI modules with detailed error handling
AISignalGenerator = None
AIModelTrainer = None

try:
    print("[INFO] Importing AI Signal Generator...")
    from ai_signal_generator_fixed import AISignalGenerator
    print("[OK] AISignalGenerator imported successfully")
except ImportError as e:
    print(f"[ERROR] Cannot import AISignalGenerator: {e}")
    print("[FIX] Check ai_signal_generator_fixed.py file")
    sys.exit(1)
except SyntaxError as e:
    print(f"[ERROR] Syntax error in ai_signal_generator_fixed.py: {e}")
    sys.exit(1)

try:
    print("[INFO] Importing Model Trainer...")
    from train_model_xgboost_fixed import AIModelTrainer
    print("[OK] AIModelTrainer imported successfully")
except ImportError as e:
    print(f"[ERROR] Cannot import AIModelTrainer: {e}")
    print("[FIX] Check train_model_xgboost_fixed.py file")
    sys.exit(1)
except SyntaxError as e:
    print(f"[ERROR] Syntax error in train_model_xgboost_fixed.py: {e}")
    sys.exit(1)

print("[OK] All AI modules imported successfully")

class ExnessAIScalperSystem:
    def __init__(self, config_file="config.json"):
        """Initialize the AI Scalper System"""
        self.config_file = config_file
        self.generator = None
        self.running = False
        self.mt5_connected = False
        self.signal_path = None
        self.account_info = None
        
        self.system_stats = {
            'start_time': None,
            'signals_generated': 0,
            'last_signal_time': None,
            'errors': 0,
            'uptime': 0
        }
        
        self.setup_logging()
        self.setup_signal_handlers()
        self.load_config()
        self.initialize_mt5_connection()
        
    def setup_logging(self):
        """Setup logging with proper encoding"""
        os.makedirs("logs", exist_ok=True)
        
        class SafeStreamHandler(logging.StreamHandler):
            def emit(self, record):
                try:
                    msg = self.format(record)
                    # Replace problematic Unicode characters
                    msg = msg.replace('✓', '[OK]').replace('✗', '[ERROR]')
                    msg = msg.replace('⚡', '[INFO]').replace('🎯', '[TARGET]')
                    self.stream.write(msg + self.terminator)
                    self.flush()
                except Exception:
                    self.handleError(record)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/ai_system.log', encoding='utf-8'),
                SafeStreamHandler()
            ]
        )
        self.logger = logging.getLogger('ExnessAIScaler')
        
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
    def load_config(self):
        """Load system configuration"""
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
                print(f"[OK] Configuration loaded from {self.config_file}")
        except FileNotFoundError:
            print(f"[WARNING] Config file {self.config_file} not found, creating default")
            self.config = {
                "broker": "Exness",
                "account_type": "trial",
                "account_number": "247501403",
                "symbol": "XAUUSDm",
                "signal_threshold": 0.70,
                "max_daily_signals": 10,
                "risk_per_trade": 0.01,
                "system": {
                    "check_interval": 60,
                    "max_errors": 10,
                    "restart_on_error": True
                }
            }
            # Save default config
            try:
                with open(self.config_file, 'w') as f:
                    json.dump(self.config, f, indent=2)
                print(f"[OK] Default configuration saved to {self.config_file}")
            except Exception as e:
                print(f"[WARNING] Could not save config: {e}")
                
    def initialize_mt5_connection(self):
        """Initialize MT5 connection"""
        try:
            print("\n" + "="*60)
            print("INITIALIZING EXNESS MT5 CONNECTION")
            print("="*60)
            
            # Try to initialize MT5
            if not mt5.initialize():
                error_info = mt5.last_error()
                print(f"[WARNING] MT5 initialization failed: {error_info}")
                print("[INFO] System will run with simulated data")
                self.mt5_connected = False
                return False
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                print(f"[WARNING] Could not get account information")
                print("[INFO] MT5 connected but no account info available")
                self.mt5_connected = False
                return False
            
            # Store account info and set connected flag
            self.account_info = account_info
            self.mt5_connected = True
            
            # Display connection info
            print(f"[OK] Connected to Exness MT5")
            print(f"[OK] Account: {account_info.login}")
            print(f"[OK] Server: {account_info.server}")
            print(f"[OK] Balance: ${account_info.balance:,.2f}")
            print(f"[OK] Currency: {account_info.currency}")
            
            # Check account type
            server_name = account_info.server.lower()
            if any(keyword in server_name for keyword in ['demo', 'trial', 'contest']):
                print(f"[OK] Account Type: TRIAL/DEMO")
                self.config['account_type'] = 'trial'
            else:
                print(f"[WARNING] Account Type: LIVE")
            
            # Setup signal path
            self.setup_signal_file_path()
            return True
            
        except Exception as e:
            print(f"[ERROR] MT5 connection error: {e}")
            self.mt5_connected = False
            return False
            
    def setup_signal_file_path(self):
        """Setup signal file path"""
        try:
            if self.mt5_connected:
                # Get MT5 data path
                terminal_info = mt5.terminal_info()
                if terminal_info:
                    data_path = Path(terminal_info.data_path)
                    files_dir = data_path / "MQL5" / "Files"
                    files_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Use account-specific filename
                    account_number = self.config.get('account_number', '247501403')
                    signal_filename = f"xau_signal_{account_number}.txt"
                    self.signal_path = files_dir / signal_filename
                    
                    print(f"[OK] Signal file: {self.signal_path}")
                    
                    # Test file creation
                    test_data = {"status": "READY", "timestamp": datetime.now().isoformat()}
                    with open(self.signal_path, 'w') as f:
                        json.dump(test_data, f)
                    print(f"[OK] Signal file test successful")
                    return
            
            # Fallback to local directory
            self.signal_path = Path("signals/xau_signal.txt")
            self.signal_path.parent.mkdir(exist_ok=True)
            print(f"[OK] Using local signal path: {self.signal_path}")
            
        except Exception as e:
            print(f"[ERROR] Signal path setup error: {e}")
            self.signal_path = Path("signals/xau_signal.txt")
            self.signal_path.parent.mkdir(exist_ok=True)
            
    def check_model_availability(self):
        """Check if trained models exist"""
        model_files = [
            "models/xgboost_model.pkl",
            "models/scaler.pkl"
        ]
        
        missing_files = []
        for file_path in model_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            print(f"[WARNING] Missing model files: {missing_files}")
            return False
            
        print(f"[OK] All model files found")
        return True
        
    def train_model_interactive(self):
        """Interactive model training"""
        print("\n" + "="*60)
        print("MODEL TRAINING REQUIRED")
        print("="*60)
        
        print("\nThe AI model needs to be trained before use.")
        print("This will download data and train the XGBoost model.")
        print(f"MT5 Connected: {'YES' if self.mt5_connected else 'NO (will use simulated data)'}")
        print("Estimated time: 5-15 minutes")
        
        response = input("\nTrain the model now? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            try:
                print(f"[INFO] Starting model training...")
                
                # Try using the imported trainer class
                if AIModelTrainer:
                    trainer = AIModelTrainer()
                    success = trainer.run_training()
                    
                    if success:
                        print(f"[OK] Model training completed successfully!")
                        return True
                    else:
                        print(f"[ERROR] Model training failed")
                
                # Fallback: try running the training script directly
                print(f"[INFO] Trying alternative training method...")
                result = subprocess.run([
                    sys.executable, "train_model_xgboost_fixed.py"
                ], capture_output=True, text=True, timeout=900)  # 15 minute timeout
                
                if result.returncode == 0:
                    print(f"[OK] Model trained successfully using subprocess")
                    return True
                else:
                    print(f"[ERROR] Training subprocess failed: {result.stderr}")
                    return False
                    
            except Exception as e:
                print(f"[ERROR] Training error: {e}")
                return False
        
        print("\nTo train manually later, run: python train_model_xgboost_fixed.py")
        return False
        
    def save_signal_to_file(self, signal_data):
        """Save signal to file"""
        try:
            timestamp = datetime.now()
            
            # Prepare signal data
            enhanced_signal = {
                "broker": "Exness",
                "account_type": self.config.get('account_type', 'trial'),
                "account_number": self.config.get('account_number', '247501403'),
                "timestamp": timestamp.isoformat(),
                "mt5_connected": self.mt5_connected,
                "signal": signal_data.get('signal', 'HOLD') if isinstance(signal_data, dict) else str(signal_data),
                "confidence": signal_data.get('confidence', 0.0) if isinstance(signal_data, dict) else 0.0,
                "price": signal_data.get('price', 0.0) if isinstance(signal_data, dict) else 0.0,
                "symbol": self.config.get('symbol', 'XAUUSDm')
            }
            
            # Add account info if available
            if self.account_info:
                enhanced_signal["account_balance"] = float(self.account_info.balance)
                enhanced_signal["account_equity"] = float(self.account_info.equity)
            
            # Save to signal file
            signal_json = json.dumps(enhanced_signal, indent=2, default=str)
            
            if self.signal_path:
                with open(self.signal_path, 'w', encoding='utf-8') as f:
                    f.write(signal_json)
                return True
            
            return False
            
        except Exception as e:
            print(f"[ERROR] Signal save error: {e}")
            return False
            
    def start_signal_generation(self):
        """Start signal generation"""
        try:
            print(f"\n[INFO] Initializing AI Signal Generator...")
            
            # Initialize signal generator
            if AISignalGenerator:
                # Pass configuration to generator
                if isinstance(self.config, dict):
                    self.generator = AISignalGenerator(self.config_file)
                else:
                    self.generator = AISignalGenerator()
                    
                print(f"[OK] AI Signal Generator initialized")
            else:
                raise Exception("AISignalGenerator class not available")
            
            # Update system stats
            self.system_stats['start_time'] = datetime.now()
            
            # Main signal generation loop
            print("\n" + "-"*60)
            print("EXNESS AI SCALPER - LIVE SIGNAL GENERATION")
            print("-"*60)
            print(f"MT5 Connected: {'YES' if self.mt5_connected else 'NO (Using Simulated Data)'}")
            print(f"Signal File: {self.signal_path}")
            print("Press Ctrl+C to stop")
            print("-"*60)
            
            interval = self.config.get('system', {}).get('check_interval', 60)
            
            while self.running:
                try:
                    # Generate signal
                    signal_data = self.generator.generate_signal()
                    
                    # Save signal
                    self.save_signal_to_file(signal_data)
                    
                    # Update stats
                    self.system_stats['signals_generated'] += 1
                    self.system_stats['last_signal_time'] = datetime.now()
                    
                    # Display signal
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    if isinstance(signal_data, dict):
                        confidence = signal_data.get('confidence', 0) * 100
                        signal_type = signal_data.get('signal', 'HOLD')
                        price = signal_data.get('price', 0)
                        
                        if signal_type == 'BUY':
                            print(f"[{timestamp}] [BUY] Signal: {signal_type} - Confidence: {confidence:.1f}% - Price: {price:.2f}")
                        elif signal_type == 'SELL':
                            print(f"[{timestamp}] [SELL] Signal: {signal_type} - Confidence: {confidence:.1f}% - Price: {price:.2f}")
                        else:
                            print(f"[{timestamp}] [HOLD] Signal: {signal_type} - Confidence: {confidence:.1f}%")
                    else:
                        print(f"[{timestamp}] [INFO] Signal: {signal_data}")
                    
                    # Show connection status
                    if not self.mt5_connected:
                        print(f"[WARNING] Using simulated data - MT5 not connected")
                    
                    time.sleep(interval)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.system_stats['errors'] += 1
                    print(f"[ERROR] Signal generation error: {e}")
                    
                    max_errors = self.config.get('system', {}).get('max_errors', 10)
                    if self.system_stats['errors'] >= max_errors:
                        print(f"[ERROR] Too many errors ({max_errors}), stopping")
                        break
                    
                    time.sleep(30)
                    
        except Exception as e:
            print(f"[ERROR] Fatal signal generation error: {e}")
            
    def display_startup_banner(self):
        """Display startup banner"""
        banner = """
        ================================================================
                    EXNESS AI SCALPER XAUUSD SYSTEM                     
                      Advanced Trading System v2.0                     
                                                                      
          AI-Powered Signal Generation for Exness MetaTrader 5      
          Trial Account Support with Risk Management                 
          Real-time Signal Integration                               
        ================================================================
        """
        print(banner)
        
    def display_system_info(self):
        """Display system information"""
        print("\n" + "="*60)
        print("SYSTEM INFORMATION")
        print("="*60)
        
        print(f"Python Version: {sys.version.split()[0]}")
        print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Account info
        if self.account_info:
            print(f"\nExness Account:")
            print(f"  Account: {self.account_info.login}")
            print(f"  Server: {self.account_info.server}")
            print(f"  Balance: ${self.account_info.balance:,.2f}")
            print(f"  Leverage: 1:{self.account_info.leverage}")
        
        # Configuration
        print(f"\nTrading Configuration:")
        print(f"  Symbol: {self.config.get('symbol', 'XAUUSDm')}")
        print(f"  Signal Threshold: {self.config.get('signal_threshold', 0.7)}")
        print(f"  Risk Per Trade: {self.config.get('risk_per_trade', 0.02) * 100}%")
        
        # Status
        print(f"\nSystem Status:")
        print(f"  MT5 Connection: {'CONNECTED' if self.mt5_connected else 'DISCONNECTED'}")
        print(f"  AI Model: {'AVAILABLE' if self.check_model_availability() else 'MISSING'}")
        print(f"  Signal File: {'READY' if self.signal_path else 'NOT CONFIGURED'}")
        
        print("="*60)
        
    def start_system(self):
        """Start the complete system"""
        try:
            # Display banner
            self.display_startup_banner()
            
            # Check model availability
            if not self.check_model_availability():
                if not self.train_model_interactive():
                    print(f"[ERROR] Cannot proceed without trained model")
                    return False
            
            # Display system info
            self.display_system_info()
            
            # Start signal generation
            self.running = True
            self.start_signal_generation()
            
        except KeyboardInterrupt:
            print(f"\n[INFO] System stopped by user")
        except Exception as e:
            print(f"[ERROR] System error: {e}")
        finally:
            self.shutdown()
            
    def shutdown(self):
        """Graceful shutdown"""
        print(f"\n[INFO] Shutting down system...")
        self.running = False
        
        # Close MT5
        try:
            if self.mt5_connected:
                mt5.shutdown()
                print(f"[OK] MT5 connection closed")
        except:
            pass
        
        # Display stats
        print(f"\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        print(f"Signals Generated: {self.system_stats['signals_generated']}")
        print(f"Errors: {self.system_stats['errors']}")
        print(f"MT5 Connection: {'Active' if self.mt5_connected else 'Inactive'}")
        print("="*60)
        print(f"\nThank you for using Exness AI Scalper!")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Exness AI Scalper System')
    parser.add_argument('--config', default='config.json', help='Config file path')
    parser.add_argument('--test-mt5', action='store_true', help='Test MT5 connection only')
    parser.add_argument('--train-only', action='store_true', help='Train model only')
    args = parser.parse_args()
    
    if args.test_mt5:
        # Quick MT5 test
        print("Testing MT5 connection...")
        if mt5.initialize():
            account = mt5.account_info()
            if account:
                print(f"[OK] Connected to: {account.login}")
                print(f"[OK] Server: {account.server}")
                print(f"[OK] Balance: ${account.balance:,.2f}")
            mt5.shutdown()
        else:
            print("[ERROR] Could not connect to MT5")
        return
    
    if args.train_only:
        # Train model only
        if AIModelTrainer:
            trainer = AIModelTrainer()
            success = trainer.run_training()
            if success:
                print("[OK] Training completed successfully!")
            else:
                print("[ERROR] Training failed!")
        else:
            print("[ERROR] AIModelTrainer not available")
        return
    
    # Start full system
    system = ExnessAIScalperSystem(args.config)
    system.start_system()

if __name__ == "__main__":
    main()