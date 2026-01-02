import os
from pathlib import Path

import kagglehub
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

import os
import sys

# Garante que a pasta raiz (onde está main.py) entra no sys.path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from demos.main import normalize_cols, prepare_data

def load_train_test():
    """
    Carrega o mesmo dataset usado no main.py,
    aplica normalize_cols e junta o FLOOR via points_file.
    Retorna df_train, df_test prontos para chamar prepare_data.
    """
    print("Baixando dataset (grid search)...")
    dataset_path = kagglehub.dataset_download(
        "rafaelsaraivacampos/25d-indoor-positioning-using-wifi-signals"
    )

    csv_files = sorted(list(Path(dataset_path).rglob("*.csv")))
    if len(csv_files) < 3:
        raise RuntimeError(f"Esperava pelo menos 3 CSVs, encontrei: {len(csv_files)}")

    # Mesmo mapeamento que você usou no main.py
    train_file = csv_files[2]  # uerj_wifi_indoorLoc_train.csv
    test_file = csv_files[1]   # uerj_wifi_indoorLoc_test.csv
    points_file = csv_files[0] # uerj_wifi_indoorLoc_points.csv

    print(f"treino (GS): {train_file.name}")
    print(f"teste  (GS): {test_file.name}")
    print(f"pontos (GS): {points_file.name}")

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    points_df = pd.read_csv(points_file)

    # Normaliza colunas
    df_train = normalize_cols(df_train)
    df_test = normalize_cols(df_test)
    points_df = normalize_cols(points_df)

    # Garante colunas point / FLOOR em points_df (mesma lógica do main)
    if "point" not in points_df.columns and "scanId" in points_df.columns:
        points_df = points_df.rename(columns={"scanId": "point"})
    if "FLOOR" not in points_df.columns and "floor" in points_df.columns:
        points_df = points_df.rename(columns={"floor": "FLOOR"})

    # Junta FLOOR em treino e teste
    if "point" in df_train.columns and "point" in points_df.columns and "FLOOR" in points_df.columns:
        df_train = df_train.merge(points_df[["point", "FLOOR"]], on="point", how="left")
        df_test  = df_test.merge(points_df[["point", "FLOOR"]], on="point", how="left")
    else:
        raise RuntimeError("Não foi possível juntar 'FLOOR' usando a coluna 'point'.")

    return df_train, df_test


def build_feature_matrices(df_train, df_test, target_col="FLOOR"):
    """
    A partir de df_train/df_test já com FLOOR,
    usa prepare_data para criar X_train, y_train, X_test, y_test
    e remove colunas de ID numéricas.
    """
    X_train, y_train = prepare_data(df_train, target_col)
    X_test, y_test = prepare_data(df_test, target_col)

    # Identifica colunas de ID (mesma ideia do main)
    id_cols = [c for c in ["point", "scanId", "scan_id", "id"]
               if c in X_train.columns or c in X_test.columns]
    if id_cols:
        print(f"Colunas IDs detectadas: {id_cols}")
        X_train = X_train.drop(columns=[c for c in id_cols if c in X_train.columns], errors="ignore")
        X_test  = X_test.drop(columns=[c for c in id_cols if c in X_test.columns], errors="ignore")

    # (Opcional) restringir explicitamente às colunas RSS*
    # rss_cols = [c for c in X_train.columns if str(c).upper().startswith("RSS")]
    # X_train = X_train[rss_cols]
    # X_test  = X_test[rss_cols]

    return X_train, y_train, X_test, y_test


def run_baseline_and_grid_search(X_train, y_train, X_test, y_test):
    # 1) Baseline: igual ao MLP anterior
    baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(100, 50),
                              max_iter=300,
                              random_state=42))
    ])

    print("\nTreinando MLP baseline (mesmo do main.py)...")
    baseline.fit(X_train, y_train)
    baseline_preds = baseline.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_preds)
    print(f"Acurácia baseline (teste): {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print("Relatório baseline:")
    print(classification_report(y_test, baseline_preds))

    # 2) Grid Search em cima de um Pipeline (StandardScaler + MLP)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(max_iter=300, random_state=42))
    ])

    param_grid = {
        "mlp__hidden_layer_sizes": [
            (50,),
            (100,),
            (100, 50),   # igual ao baseline
            (150, 75)
        ],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [1e-4, 1e-3, 1e-2],
        "mlp__learning_rate_init": [1e-3, 1e-2]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=2
    )

    print("\nIniciando Grid Search para MLP...")
    grid.fit(X_train, y_train)

    print("\nMelhores hiperparâmetros (validação cruzada):")
    print(grid.best_params_)
    print(f"Melhor acurácia média em CV: {grid.best_score_:.4f}")

    best_model = grid.best_estimator_
    best_preds = best_model.predict(X_test)
    best_acc = accuracy_score(y_test, best_preds)
    print(f"\nAcurácia no teste com melhor MLP (GridSearch): {best_acc:.4f} ({best_acc*100:.2f}%)")
    print("Relatório do melhor modelo:")
    print(classification_report(y_test, best_preds))

    print("\nComparação final:")
    print(f"Baseline  (main.py)  ~ MLP(100,50):  {baseline_acc:.4f}")
    print(f"GridSearch melhor MLP:              {best_acc:.4f}")

    return baseline, best_model


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    df_train, df_test = load_train_test()
    X_train, y_train, X_test, y_test = build_feature_matrices(df_train, df_test)
    baseline_model, best_model = run_baseline_and_grid_search(X_train, y_train, X_test, y_test)
