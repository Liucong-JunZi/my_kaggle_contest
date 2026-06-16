
import pickle
with open('feat_cols.pkl', 'rb') as f:
    feat_cols = pickle.load(f)
print(f"Number of features: {len(feat_cols)}")
for i, feat in enumerate(feat_cols):
    print(f"{i+1:2d}. {feat}")
