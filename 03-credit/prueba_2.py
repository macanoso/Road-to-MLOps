"""
Script para optimización de hiperparámetros con LightGBM y MLflow

Uso:
    python prueba_2.py -m MODULE -n N_TRIALS -file FILE_PATH [-exp EXPERIMENT_NAME]

Parámetros:
    -m, --module: Módulo de información a seleccionar
    -n, --n_trials: Número de trials para optimización
    -file, --file: Ruta del archivo de datos
    -exp, --experiment_name: Nombre personalizado del experimento MLflow (opcional)
    
Si no se especifica experiment_name, se genera automáticamente con timestamp para evitar sobrescribir experimentos.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname('_file_'), '..')))

import argparse
import pickle
from datetime import datetime

import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from loss_func import funcion_costo
from model_reviews import ModelReviewTotal
from utils import get_or_create_experiment, use_flag, use_x, use_y


# Simple trial logger for CSV fallback and MLflow logging after completion
def log_trial_results(study, csv_path, mlflow_enabled=True):
    """Log all completed trials to MLflow and save CSV backup"""
    trials_data = []
    
    for trial in study.trials:
        trial_dict = {
            'trial_number': trial.number,
            'objective_value': trial.value,
            'trial_state': trial.state.name,
            **trial.params
        }
        trials_data.append(trial_dict)
        
        # Log each trial to MLflow as separate run (not nested)
        if mlflow_enabled and trial.state == optuna.trial.TrialState.COMPLETE:
            try:
                with mlflow.start_run(run_name=f"trial_{trial.number:04d}", nested=False):
                    # Log parameters
                    mlflow.log_params(trial.params)
                    # Log metrics
                    if trial.value is not None:
                        mlflow.log_metric("objective_value", trial.value)
                    # Log trial info as parameters
                    mlflow.log_param("trial_state", trial.state.name)
                    mlflow.log_param("trial_number", trial.number)
            except Exception as e:
                print(f"Warning: MLflow logging failed for trial {trial.number}: {e}")
                continue
    
    # Always save CSV backup
    try:
        trials_df = pd.DataFrame(trials_data)
        trials_df.to_csv(csv_path, index=False)
        print(f"Trial results saved to CSV: {csv_path}")
    except Exception as e:
        print(f"Error: Failed to save CSV backup: {e}")
    
    return trials_data

parser = argparse.ArgumentParser(description='Seleccion de variable')
parser.add_argument('-m', '--module' ,
                    type = str, help = 'modulo de informacion a seleccionar')
parser.add_argument('-n', '--n_trials' ,
                    type = int, help = 'numero de trials para sacar variables optimas')
parser.add_argument('-file', '--file' ,
                    type = str, help = 'ruta donde se encuentra el archivo')
parser.add_argument('-exp', '--experiment_name' ,
                    type = str, help = 'nombre del experimento de MLflow (opcional)', default=None)
args = parser.parse_args()

df = pd.read_csv(args.file, compression='gzip', delimiter=';')
df_total = df.copy()
del df

def optimize_lgb(trial):
    boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf'])
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    learning_rate = trial.suggest_float('learning_rate', 0.05, 0.25)
    num_leaves = trial.suggest_int('num_leaves', 2, 200)
    max_depth = trial.suggest_int('max_depth', 2,20)
    min_child_weight = trial.suggest_float('min_child_weight', 0.05, 0.1)
    min_child_samples = trial.suggest_int('min_child_samples', 5,50)
    feature_fraction = trial.suggest_float('feature_fraction', 0.4, 0.9)
    random_state = 3418270147
    reg_alpha = trial.suggest_float('reg_alpha', 0.1, 1.0)
    reg_lambda = trial.suggest_float('reg_lambda', 0.1, 1.0)
    max_bin = trial.suggest_int('max_bin', 50, 300)
    min_split_gain = trial.suggest_float('min_split_gain', 0.0, 0.5)
    
    if boosting_type == 'rf':
        bagging_fraction = trial.suggest_float('bagging_fraction', 0.4, 0.9)
        bagging_freq = trial.suggest_int('bagging_freq', 1, 2)
        subsample = 1.0
        subsample_freq = 0
    else:
        subsample = trial.suggest_float('subsample', 0.4, 0.9)
        subsample_freq = trial.suggest_int('subsample_freq', 1, 10)
        bagging_fraction = None
        bagging_freq = None

    colsample_bytree = trial.suggest_float('colsample_bytree', 0.4, 0.9)

    model = LGBMClassifier(
        boosting_type=boosting_type,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        min_child_samples=min_child_samples,
        feature_fraction=feature_fraction,
        random_state=random_state,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        max_bin=max_bin,
        min_split_gain=min_split_gain,
        objective='binary',
        verbose=-1,
        subsample=subsample,
        subsample_freq=subsample_freq,
        colsample_bytree=colsample_bytree,
        bagging_fraction=bagging_fraction,
        bagging_freq=bagging_freq
    )

    model.fit(  
                use_x(use_flag(df_total,'train'),use_flag=False), 
                use_y(use_flag(df_total,'train'),use_flag=False)
                )
    
    costo = funcion_costo(
                            use_x(df_total,), 
                            use_y(df_total,), 
                            model
                            )

    return costo

# Configuración inicial
nombre_id = os.path.basename(__file__)
nombre_id = os.path.splitext(nombre_id)[0]

# Generar nombre de experimento único
if args.experiment_name:
    # Usar nombre proporcionado por el usuario
    experiment_name = args.experiment_name
else:
    # Generar nombre automático con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    module_suffix = f"_{args.module}" if args.module else ""
    experiment_name = f"hyperparameter_search_{nombre_id}_{timestamp}{module_suffix}"

print(f"Using experiment name: {experiment_name}")

# Configurar experimento de MLflow
mlflow.set_experiment(experiment_name)

# Ejecutar optimización sin MLflow callbacks (logging después)
sampler = optuna.samplers.RandomSampler(seed=42)
study_lgb = optuna.create_study(sampler=sampler, direction="maximize")

# Ejecutar optimización sin callback (pure Optuna)
study_lgb.optimize(optimize_lgb, n_trials=args.n_trials, 
                  show_progress_bar=True)

# Obtener mejores resultados
best_params_lgb = study_lgb.best_params
best_score_lgb = study_lgb.best_value

# CSV path for backup (include timestamp to avoid overwrites)
timestamp_for_files = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f"optuna_trials_{nombre_id}_{timestamp_for_files}.csv"

# Log all trials to MLflow AFTER completion and save CSV backup
print("Logging trials to MLflow and saving CSV backup...")
try:
    trials_data = log_trial_results(study_lgb, csv_path, mlflow_enabled=True)
    print(f"Successfully logged {len([t for t in study_lgb.trials if t.state == optuna.trial.TrialState.COMPLETE])} trials to MLflow")
except Exception as e:
    print(f"MLflow logging failed, but CSV backup was saved: {e}")
    # Fallback: just save CSV manually
    trials_data = []
    for trial in study_lgb.trials:
        trial_dict = {
            'trial_number': trial.number,
            'objective_value': trial.value,
            'trial_state': trial.state.name,
            **trial.params
        }
        trials_data.append(trial_dict)
    
    trials_df = pd.DataFrame(trials_data)
    trials_df.to_csv(csv_path, index=False)
    print(f"Fallback: Trial results saved to CSV: {csv_path}")



print('LGBMClassifier Best Params:', best_params_lgb)
print('LGBMClassifier Best Score:', best_score_lgb)

# Entrenar y guardar modelo final
with mlflow.start_run(run_name=f"final_model_{nombre_id}_{timestamp_for_files}"):
    # Entrenar modelo final con mejores parámetros
    lgbm = LGBMClassifier(**best_params_lgb)
    lgbm.fit(  
                use_x(use_flag(df_total,'train'),use_flag=False), 
                use_y(use_flag(df_total,'train'),use_flag=False)
            )

    # Crear estructura de carpetas (include timestamp to avoid overwrites)
    nombre_carpeta = f'lgbm_{nombre_id}_{timestamp_for_files}'
    os.makedirs(f'./MODELS/{nombre_carpeta}', exist_ok=True)

    # Guardar modelo localmente
    filename = f'./MODELS/{nombre_carpeta}/lgbm_{nombre_id}.pkl'
    pickle.dump(lgbm, open(filename, 'wb'))

    # Generar review del modelo
    review = ModelReviewTotal(
                                modelo=lgbm,
                                x=use_x(df_total),
                                y=use_y(df_total),
                                df_completo=df_total,
                                fa_column='fi',
                                id_column='cis',
                                bases=['train','test','oot1','oot2']
                                )

    # Generar gráficos
    review.graficar_mapeo_deciles(clase=30,
                                save_path=f"./MODELS/{nombre_carpeta}/percentil_30_modelo_{nombre_id}.png")
    review.graficar_mapeo_deciles(clase=50,
                                save_path=f"./MODELS/{nombre_carpeta}/percentil_50_modelo_{nombre_id}.png")
    
    # Obtener métricas del review (aunque no se están usando actualmente)
    review.auc_gini(f"./MODELS/{nombre_carpeta}/")
    review.revisar_tl(f"./MODELS/{nombre_carpeta}/")
    
    # Log información del modelo final
    mlflow.log_params({f"final_{k}": v for k, v in best_params_lgb.items()})
    mlflow.log_metric("final_best_score", best_score_lgb)
    
    # Log modelo usando LightGBM flavor
    mlflow.lightgbm.log_model(lgbm, "model")
    
    # Log artifacts del modelo
    mlflow.log_artifacts(f"./MODELS/{nombre_carpeta}", "model_outputs")

print(f"Experimento '{experiment_name}' completado.")
print(f"Resultados guardados en: {csv_path}")
print(f"Modelo guardado en: ./MODELS/{nombre_carpeta}/")
print(f"Mejores parámetros: {best_params_lgb}")
print(f"Mejor score: {best_score_lgb}")