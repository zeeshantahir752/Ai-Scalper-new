# Create test_imports.py
import os
from pathlib import Path

print(f"Current directory: {os.getcwd()}")
print(f"Python files: {list(Path('.').glob('*.py'))}")

try:
    from train_model_xgboost_fixed import AIModelTrainer
    print("✅ AIModelTrainer import successful!")
except Exception as e:
    print(f"❌ Import failed: {e}")

try:
    from ai_signal_generator_fixed import AISignalGenerator  
    print("✅ AISignalGenerator import successful!")
except Exception as e:
    print(f"❌ Import failed: {e}")