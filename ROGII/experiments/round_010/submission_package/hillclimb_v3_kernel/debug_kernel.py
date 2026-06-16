
import os
import pickle

print("=== Debug Kaggle Input ===")

# Check kaggle input directory structure
print("\n1. /kaggle/input directory contents:")
if os.path.exists('/kaggle/input'):
    for item in os.listdir('/kaggle/input'):
        item_path = os.path.join('/kaggle/input', item)
        print(f"   {item} -> {item_path}")
        if os.path.isdir(item_path):
            try:
                subitems = os.listdir(item_path)
                for subitem in subitems[:20]:
                    subitem_path = os.path.join(item_path, subitem)
                    size_str = f" ({os.path.getsize(subitem_path)} bytes)" if os.path.isfile(subitem_path) else ""
                    print(f"     - {subitem}{size_str}")
            except Exception as e:
                print(f"     [ERROR listing {item_path}: {e}]")

# Check our dataset specifically
print("\n2. Checking dataset paths:")
dataset_paths = [
    '/kaggle/input/rogii-pretrained-models-v1',
    '/kaggle/input/smartorz/rogii-pretrained-models-v1',
]
for path in dataset_paths:
    exists = os.path.exists(path)
    is_dir = os.path.isdir(path) if exists else False
    print(f"   {path}: exists={exists}, is_dir={is_dir}")
    if exists and is_dir:
        try:
            files = os.listdir(path)
            for f in files:
                fpath = os.path.join(path, f)
                size = os.path.getsize(fpath)
                print(f"     - {f} ({size} bytes)")
        except Exception as e:
            print(f"     [ERROR listing {path}: {e}]")

# Check if files are readable
print("\n3. Checking if model files are readable:")
for dirname in dataset_paths:
    if os.path.exists(dirname) and os.path.isdir(dirname):
        for fname in ['lgb_model.txt', 'cat_model.cbm', 'feat_cols.pkl']:
            fpath = os.path.join(dirname, fname)
            exists = os.path.exists(fpath)
            is_file = os.path.isfile(fpath) if exists else False
            readable = os.access(fpath, os.R_OK) if exists else False
            size = os.path.getsize(fpath) if exists else -1
            print(f"   {fname} in {dirname}: exists={exists}, is_file={is_file}, readable={readable}, size={size}")

# Check competition data
print("\n4. Checking competition data:")
competition_slugs = [
    'rogii-wellbore-geology-prediction',
    'competitions/rogii-wellbore-geology-prediction',
]
for slug in competition_slugs:
    path = f"/kaggle/input/{slug}"
    if os.path.exists(path) and os.path.isdir(path):
        print(f"   Found competition data at: {path}")
        items = os.listdir(path)
        for item in items:
            itempath = os.path.join(path, item)
            size_str = f" ({os.path.getsize(itempath)} bytes)" if os.path.isfile(itempath) else ""
            print(f"     - {item}{size_str}")

print("\n=== Debug Complete ===")

# Create a dummy submission to ensure kernel outputs something
import pandas as pd
dummy_sub = pd.DataFrame({
    'id': ['dummy_1'],
    'tvt': [1234.5]
})
dummy_sub.to_csv('/kaggle/working/submission.csv', index=False)
print("Created dummy submission.csv")
