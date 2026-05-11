import os
import numpy as np
import pandas as pd
from urllib import request
import zipfile
import json
from generate_mask import generate_mask

# NOTE: stroke dataset harus didownload manual dari Kaggle karena butuh autentikasi.
# Letakkan file 'healthcare-dataset-stroke-data.csv' di folder datasets/stroke/
# sebelum menjalankan script ini.
# Link: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

DATA_DIR = 'datasets'

NAME_URL_DICT = {
    'coupon': 'https://archive.ics.uci.edu/static/public/603/in+vehicle+coupon+recommendation.zip',
    'churn':  'https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip',
}

# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────

def unzip_file(zip_filepath, dest_path):
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_path)


def download_dataset(name, url):
    print(f'Start processing dataset {name}.')
    save_dir = f'{DATA_DIR}/{name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        request.urlretrieve(url, f'{save_dir}/{name}.zip')
        print(f'Finish downloading from {url}.')
        unzip_file(f'{save_dir}/{name}.zip', save_dir)
        print(f'Finish unzipping {name}.')
    else:
        print(f'Already downloaded: {name}.')


# ─────────────────────────────────────────────
# Processing functions
# ─────────────────────────────────────────────

def process_coupon():
    """
    Dataset: In-Vehicle Coupon Recommendation (UCI id=603)
    Preprocessing:
      - Hapus kolom 'car' dan 'bar'
      - Hapus baris yang mengandung missing value
    """
    # File CSV ada di dalam zip langsung
    raw_path = f'{DATA_DIR}/coupon/in-vehicle-coupon-recommendation.csv'
    save_path = f'{DATA_DIR}/coupon/data.csv'

    data_df = pd.read_csv(raw_path)

    # Hapus kolom 'car' dan 'bar' (case-insensitive fallback)
    cols_to_drop = [c for c in data_df.columns if c.lower() in ('car', 'bar')]
    data_df = data_df.drop(columns=cols_to_drop)
    print(f'[coupon] Dropped columns: {cols_to_drop}')

    # Hapus baris dengan missing value
    before = len(data_df)
    data_df = data_df.dropna()
    after = len(data_df)
    print(f'[coupon] Dropped {before - after} rows with missing values. Remaining: {after}')

    data_df.to_csv(save_path, index=False)
    print(f'[coupon] Saved to {save_path}')
    return data_df


def process_stroke():
    """
    Dataset: Stroke Prediction Dataset (Kaggle - fedesoriano)
    File harus diletakkan manual di: datasets/stroke/healthcare-dataset-stroke-data.csv
    Preprocessing:
      - Hapus baris yang mengandung missing value
    """
    raw_path = f'{DATA_DIR}/stroke/healthcare-dataset-stroke-data.csv'
    save_path = f'{DATA_DIR}/stroke/data.csv'

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f'\n[stroke] File tidak ditemukan di {raw_path}\n'
            'Download manual dari: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset\n'
            f'Lalu letakkan file CSV-nya di folder: {DATA_DIR}/stroke/'
        )

    data_df = pd.read_csv(raw_path)

    # Hapus kolom 'id' jika ada (bukan fitur)
    if 'id' in data_df.columns:
        data_df = data_df.drop(columns=['id'])
        print('[stroke] Dropped column: id')

    # Hapus baris dengan missing value
    before = len(data_df)
    data_df = data_df.dropna()
    after = len(data_df)
    print(f'[stroke] Dropped {before - after} rows with missing values. Remaining: {after}')

    data_df.to_csv(save_path, index=False)
    print(f'[stroke] Saved to {save_path}')
    return data_df


def process_churn():
    """
    Dataset: Iranian Churn Dataset (UCI id=563)
    Preprocessing:
      - Tidak ada missing value (dataset sudah bersih)
      - Hapus kolom 'Customer ID' jika ada (bukan fitur)
    """
    # Cari file CSV di dalam folder churn
    churn_dir = f'{DATA_DIR}/churn'
    save_path = f'{churn_dir}/data.csv'

    # Cari file CSV apapun di folder tersebut
    csv_files = [f for f in os.listdir(churn_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f'[churn] Tidak ada file CSV di {churn_dir}')
    raw_path = f'{churn_dir}/{csv_files[0]}'
    print(f'[churn] Reading from {raw_path}')

    data_df = pd.read_csv(raw_path)

    # Hapus kolom ID jika ada
    id_cols = [c for c in data_df.columns if 'id' in c.lower()]
    if id_cols:
        data_df = data_df.drop(columns=id_cols)
        print(f'[churn] Dropped columns: {id_cols}')

    # Tidak ada missing value, tapi tetap drop just in case
    before = len(data_df)
    data_df = data_df.dropna()
    after = len(data_df)
    print(f'[churn] Dropped {before - after} rows with missing values. Remaining: {after}')

    data_df.to_csv(save_path, index=False)
    print(f'[churn] Saved to {save_path}')
    return data_df


# ─────────────────────────────────────────────
# Auto-generate Info JSON (seperti shoppers.json)
# ─────────────────────────────────────────────

def generate_info_json(dataname, data_df, target_col):
    """
    Generate file JSON info (num_col_idx, cat_col_idx, target_col_idx)
    mengikuti format yang sama dengan shoppers.json.
    Kolom numerik murni → num_col_idx
    Kolom kategorikal (object / non-numeric) → cat_col_idx
    Target → target_col_idx
    """
    info_dir = f'{DATA_DIR}/Info'
    os.makedirs(info_dir, exist_ok=True)

    cols = data_df.columns.tolist()
    target_idx = [cols.index(target_col)]

    num_col_idx = []
    cat_col_idx = []

    for i, col in enumerate(cols):
        if col == target_col:
            continue
        if pd.api.types.is_numeric_dtype(data_df[col]):
            unique_vals = set(data_df[col].dropna().unique())
            # Kolom binary 0/1 secara semantik adalah kategorikal,
            # bukan numerik kontinu → masukkan ke cat agar tidak std=0
            if unique_vals <= {0, 1}:
                cat_col_idx.append(i)
            else:
                num_col_idx.append(i)
        else:
            cat_col_idx.append(i)

    info = {
        "name": dataname,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": target_idx
    }

    info_path = f'{info_dir}/{dataname}.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)

    print(f'[{dataname}] Info JSON saved to {info_path}')
    print(f'  num_col_idx  ({len(num_col_idx)}): {num_col_idx}')
    print(f'  cat_col_idx  ({len(cat_col_idx)}): {cat_col_idx}')
    print(f'  target_col_idx: {target_idx}')
    return info


# ─────────────────────────────────────────────
# Train/test split  (sama persis dengan kode asli)
# ─────────────────────────────────────────────

def train_test_split(dataname, ratio=0.7):
    data_dir  = f'{DATA_DIR}/{dataname}'
    path      = f'{DATA_DIR}/{dataname}/data.csv'
    info_path = f'{DATA_DIR}/Info/{dataname}.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    cat_idx = info['cat_col_idx']
    num_idx = info['num_col_idx']

    data_df   = pd.read_csv(path)
    total_num = data_df.shape[0]

    if len(cat_idx) == 0:
        data_values = data_df.values[:, :-1].astype(np.float32)
        nan_idx  = np.isnan(data_values).nonzero()[0]
        keep_idx = list(set(np.arange(data_values.shape[0])) - set(list(nan_idx)))
        keep_idx = np.array(keep_idx)
    else:
        keep_idx = np.arange(total_num)

    num_train = int(keep_idx.shape[0] * ratio)
    num_test  = total_num - num_train
    seed      = 1234

    np.random.seed(seed)
    np.random.shuffle(keep_idx)

    train_idx = keep_idx[:num_train]
    test_idx  = keep_idx[-num_test:]

    train_df = data_df.loc[train_idx]
    test_df  = data_df.loc[test_idx]

    train_path = f'{data_dir}/train.csv'
    test_path  = f'{data_dir}/test.csv'

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)

    print(f'[{dataname}] Train shape: {train_df.shape}, Test shape: {test_df.shape}')
    print(f'[{dataname}] Train saved: {train_path}')
    print(f'[{dataname}] Test  saved: {test_path}')


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':

    # ── 1. Download & unzip dataset dari UCI ──────────────────────────────────
    for name, url in NAME_URL_DICT.items():
        download_dataset(name, url)

    # ── 2. Stroke: buat folder jika belum ada ─────────────────────────────────
    os.makedirs(f'{DATA_DIR}/stroke', exist_ok=True)

    # ── 3. Preprocessing ──────────────────────────────────────────────────────
    # coupon_df = process_coupon()
    # stroke_df = process_stroke()     # akan raise error jika file belum ada
    churn_df  = process_churn()

    # ── 4. Generate Info JSON ─────────────────────────────────────────────────
    # generate_info_json('coupon', coupon_df, target_col='Y')
    # generate_info_json('stroke', stroke_df, target_col='stroke')

    # churn: JSON dibuat manual (tidak pakai generate_info_json)
    churn_info = {
        "name": "churn",
        "num_col_idx": [0, 2, 3, 4, 5, 6, 7, 8, 11, 12],
        "cat_col_idx": [1, 9, 10],
        "target_col_idx": [13]
    }
    os.makedirs(f'{DATA_DIR}/Info', exist_ok=True)
    with open(f'{DATA_DIR}/Info/churn.json', 'w') as f:
        json.dump(churn_info, f, indent=4)
    print(f'[churn] Info JSON saved to {DATA_DIR}/Info/churn.json')

    # ── 5. Train / Test split ─────────────────────────────────────────────────
    for name in ['churn']:
        train_test_split(name, ratio=0.7)

    # ── 6. Generate masks (MCAR, MAR, MNAR) ──────────────────────────────────
    for name in ['churn']:
        for mask_type in ['MCAR', 'MAR', 'MNAR_logistic_T2']:
            for mask_p in [0.3]:
                generate_mask(dataname=name,
                              mask_type=mask_type,
                              mask_num=10,
                              p=mask_p)

    print('\nDone! Semua dataset sudah diproses.')
    print('Struktur folder yang dihasilkan:')
    print('  datasets/')
    print('  ├── Info/')
    print('  │   ├── coupon.json')
    print('  │   ├── stroke.json')
    print('  │   └── churn.json')
    print('  ├── coupon/')
    print('  │   ├── data.csv, train.csv, test.csv')
    print('  ├── stroke/')
    print('  │   ├── data.csv, train.csv, test.csv')
    print('  └── churn/')
    print('      ├── data.csv, train.csv, test.csv')