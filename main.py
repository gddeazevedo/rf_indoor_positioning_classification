import kagglehub
#import joblib
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plot

def normalize_cols(df):
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == 'x' and col != 'x':
            rename_map[col] = 'x'
        elif col_lower == 'y' and col != 'y':
            rename_map[col] = 'y'
        elif col_lower == 'floor' and col != 'FLOOR':
            rename_map[col] = 'FLOOR'
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

# Prepara X e y
def prepare_data(df, target):
    if isinstance(target, (list, tuple)):
        y = df[target]
        X = df.select_dtypes(include=[np.number]).drop(columns=target, errors='ignore')
    else:
        y = df[target]
        X = df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
    X = X.fillna(0)
    return X, y

# Função utilitária para treinar e avaliar modelo dado X,y
def train_eval(X_tr, y_tr, X_te, y_te, run_name='model'):
    scaler_local = StandardScaler()
    X_tr_s = scaler_local.fit_transform(X_tr)
    X_te_s = scaler_local.transform(X_te)
    clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
    clf.fit(X_tr_s, y_tr)
    preds = clf.predict(X_te_s)
    acc = accuracy_score(y_te, preds)
    return clf, scaler_local, preds, acc

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)

    print('Baixando dataset')
    dataset_path = kagglehub.dataset_download("rafaelsaraivacampos/25d-indoor-positioning-using-wifi-signals")

    # Procura todos os CSVs no dataset
    csv_files = sorted(list(Path(dataset_path).rglob('*.csv')))

    # Carrega arquivos - assume: [1º=algo, 2º=treino, 3º=teste]
    train_file = csv_files[2]  # 3º arquivo: uerj_wifi_indoorLoc_train.csv (é o treino)
    test_file = csv_files[1]   # 2º arquivo: uerj_wifi_indoorLoc_test.csv (é o teste)

    print(f'\ntreino: {train_file.name}')
    df_train = pd.read_csv(train_file)
    print(f'teste: {test_file.name}')
    df_test = pd.read_csv(test_file)

    df_train = normalize_cols(df_train)
    df_test = normalize_cols(df_test)

    # Usa o arquivo de pontos (mapeamento point -> FLOOR) quando disponível
    points_file = csv_files[0] if len(csv_files) > 0 else None
    if points_file:
        print(f'Carregando mapeamento de pontos: {points_file.name}')
        points_df = pd.read_csv(points_file)
        points_df = normalize_cols(points_df)
        # garante coluna 'point' em points_df
        if 'point' not in points_df.columns and 'scanId' in points_df.columns:
            # tenta detectar coluna equivalente
            points_df = points_df.rename(columns={'scanId': 'point'})

        # garante coluna FLOOR em points_df
        if 'FLOOR' not in points_df.columns and 'floor' in points_df.columns:
            points_df = points_df.rename(columns={'floor': 'FLOOR'})

        if 'point' in df_train.columns and 'point' in points_df.columns and 'FLOOR' in points_df.columns:
            # junta info de andar nos datasets de treino/teste
            df_train = df_train.merge(points_df[['point', 'FLOOR']], on='point', how='left')
            df_test = df_test.merge(points_df[['point', 'FLOOR']], on='point', how='left')

    # Detecta target (agora preferimos explicitamente FLOOR)
    target = 'FLOOR'

    X_train, y_train = prepare_data(df_train, target)
    X_test, y_test = prepare_data(df_test, target)

    # Detectar possíveis identificadores que causam vazamento
    id_cols = [c for c in ['point', 'scanId', 'scan_id', 'id'] if c in X_train.columns or c in X_test.columns]
    if id_cols:
        print(f'Colunas IDs: {id_cols}')

    # Remove colunas identificadoras (se existirem) para evitar vazamento
    if id_cols:
        print(f'Removendo colunas identificadoras: {id_cols}')
        X_train = X_train.drop(columns=[c for c in id_cols if c in X_train.columns], errors='ignore')
        X_test = X_test.drop(columns=[c for c in id_cols if c in X_test.columns], errors='ignore')

    # Scaling e treino
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    print(f'Acurácia no teste: {acc:.4f} ({acc*100:.2f}%)')

    print(classification_report(y_test, preds))

    plot.plot_matriz_confusao(y_test, preds, model.classes_)
    # Salva modelo final [pode ser descartada/nao necessario]
    #joblib.dump({'model': model, 'scaler': scaler}, 'outputs/model.joblib')
    #print('Modelo salvo em outputs/model.joblib')
