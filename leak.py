# Codigo que a llm gerou para verificar vazamentos nos dados de treino/teste
# Achei estranho que a precisão ta MTO alta, mas nao achei vazamento no codigo principal
# Entao criei esse script separado para analisar os dados
# Aparentemente existem linhas duplicadas entre treino e teste, o que pode causar vazamento, mas ao remover, a precisão AUMENTOU!
# Posso estar comento deslizes em algum lugar... Por favor validar 
# Ass. GabrielC

import kagglehub
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif


def find_csvs():
    dataset_path = kagglehub.dataset_download("rafaelsaraivacampos/25d-indoor-positioning-using-wifi-signals")
    csv_files = sorted(list(Path(dataset_path).rglob('*.csv')))
    return csv_files


def load_and_prepare(csv_files):
    points = pd.read_csv(csv_files[0])
    train = pd.read_csv(csv_files[2])
    test = pd.read_csv(csv_files[1])

    # normalize column names
    points.columns = [c.strip() for c in points.columns]
    train.columns = [c.strip() for c in train.columns]
    test.columns = [c.strip() for c in test.columns]

    # ensure point column
    if 'scanId' in points.columns and 'point' not in points.columns:
        points = points.rename(columns={'scanId': 'point'})

    # ensure floor column
    if 'floor' in points.columns and 'FLOOR' not in points.columns:
        points = points.rename(columns={'floor': 'FLOOR'})

    # merge floor into train/test if possible
    if 'point' in train.columns and 'point' in points.columns and 'FLOOR' in points.columns:
        train = train.merge(points[['point', 'FLOOR']], on='point', how='left')
    if 'point' in test.columns and 'point' in points.columns and 'FLOOR' in points.columns:
        test = test.merge(points[['point', 'FLOOR']], on='point', how='left')

    return points, train, test


def identify_rss_columns(df):
    return [c for c in df.columns if c.upper().startswith('RSS')]


def main():
    print('Running leakage checks...')
    csv_files = find_csvs()
    print('CSV files found:')
    for i, f in enumerate(csv_files):
        print(f'  {i+1}: {f.name}')

    points, train, test = load_and_prepare(csv_files)

    print('\nPoints file shape:', points.shape)
    print('Train shape:', train.shape)
    print('Test shape:', test.shape)

    # point -> FLOOR mapping
    if 'point' in points.columns and 'FLOOR' in points.columns:
        grp = points.groupby('point')['FLOOR'].nunique()
        multi_floor = grp[grp > 1]
        print('\npoint -> FLOOR uniqueness:')
        print('  unique points:', grp.shape[0])
        print('  points mapping to multiple FLOORs:', multi_floor.shape[0])
        if not multi_floor.empty:
            print('  examples (point -> #floors):')
            print(multi_floor.head().to_string())
    else:
        print('\nNo point->FLOOR mapping available in points file.')

    # ID overlap
    id_cols = [c for c in ['point', 'scanId', 'scan_id', 'id'] if c in train.columns and c in test.columns]
    print('\nID columns present in both train and test:', id_cols)
    for c in id_cols:
        train_ids = set(train[c].unique())
        test_ids = set(test[c].unique())
        inter = train_ids.intersection(test_ids)
        print(f"  Column '{c}': train unique={len(train_ids)}, test unique={len(test_ids)}, intersection={len(inter)}")

    # RSS columns
    rss_cols = identify_rss_columns(train)
    print(f'\nDetected {len(rss_cols)} RSS columns (example: {rss_cols[:5]})')

    # Check for constant features
    const_feats = [c for c in rss_cols if train[c].nunique() <= 1]
    print('Constant RSS features in train (if any):', const_feats)

    # Check identical RSS vectors between train and test (fast hash)
    def row_hashes(df, cols):
        # use bytes view of float32 for speed
        sub = df[cols].fillna(0).astype(np.float32)
        return pd.util.hash_pandas_object(sub, index=False)

    if rss_cols:
        train_hashes = row_hashes(train, rss_cols)
        test_hashes = row_hashes(test, rss_cols)
        inter_hashes = set(train_hashes).intersection(set(test_hashes))
        print(f'Identical RSS rows between train/test: {len(inter_hashes)}')
    else:
        print('No RSS columns detected to compare vectors.')

    # Check if x/y exist in train -> could leak FLOOR
    for coord in ['x', 'y', 'X', 'Y']:
        if coord in train.columns:
            print(f"Coordinate column present in train: {coord} (nunique={train[coord].nunique()})")

    # Mutual information of RSS -> FLOOR (if FLOOR present)
    if 'FLOOR' in train.columns:
        print('\nComputing mutual information (RSS -> FLOOR) for top features...')
        X = train[rss_cols].fillna(0).astype(float)
        y = train['FLOOR']
        # if y is string/object convert to category codes
        if y.dtype == object:
            y = y.astype('category').cat.codes
        try:
            mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
            mi_series = pd.Series(mi, index=rss_cols).sort_values(ascending=False)
            print(mi_series.head(15).to_string())
        except Exception as e:
            print('MI computation failed:', e)

    # Check for exact duplicate samples (same RSS and same point) across train/test
    if 'point' in train.columns and 'point' in test.columns and rss_cols:
        train_keys = train[rss_cols + ['point']].fillna(0).astype(str).agg('|'.join, axis=1)
        test_keys = test[rss_cols + ['point']].fillna(0).astype(str).agg('|'.join, axis=1)
        inter = set(train_keys).intersection(set(test_keys))
        print(f'Exact samples (RSS + point) present in both train/test: {len(inter)}')

    print('\nLeakage check completed.')


if __name__ == '__main__':
    main()
