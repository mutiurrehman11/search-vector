import pickle
import sys

# Add the Pipeline directory to the Python path
sys.path.append('d:\\Pycharm\\Work\\search-vector\\Pipeline')

# Now you can import the MLReRanker class
from MLReRanker import MLReRanker

try:
    with open('models/reranker.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    scaler = model_data.get('scaler')
    
    if scaler:
        print("Scaler found in model file.")
        if hasattr(scaler, 'mean_'):
            print("Scaler appears to be fitted.")
            print(f"Scaler mean: {scaler.mean_}")
        else:
            print("Scaler is NOT fitted. It's missing the 'mean_' attribute.")
    else:
        print("Scaler not found in model file.")
        
    model = model_data.get('model')
    if model:
        print("Model found in model file.")
    else:
        print("Model not found in model file.")

except Exception as e:
    print(f"An error occurred: {e}")