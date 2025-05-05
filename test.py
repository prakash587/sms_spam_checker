import pickle
from sklearn.utils.validation import check_is_fitted
from sklearn.naive_bayes import MultinomialNB

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Check if it's fitted
try:
    check_is_fitted(model)
    print("✅ Model is fitted and ready to use.")
except:
    print("❌ Model is NOT fitted. You must retrain or obtain the correct model file.")
