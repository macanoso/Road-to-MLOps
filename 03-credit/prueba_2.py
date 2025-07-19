import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname('_file_'), '..')))

import argparse
import pickle

import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from loss_func import funcion_costo
from model_reviews import ModelReviewTotal
from utils import get_or_create_experiment, use_flag, use_x, use_y


# MLflow callback following official best practices with parent-child runs
class MLflowCallback:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        # Create CSV with header
        pd.DataFrame(columns=['trial_number', 'objective_value', 'trial_state']).to_csv(csv_path, index=False)
    
    def __call__(self, study, trial):
        # Log to MLflow as CHILD run (nested under parent)
        try:
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                mlflow.log_params(trial.params)
                if trial.value is not None:
                    mlflow.log_metric("objective_value", trial.value)
                mlflow.log_param("trial_number", trial.number)
                mlflow.log_param("trial_state", trial.state.name)
        except:
            pass
        
        # Append to CSV immediately  
        trial_data = {'trial_number': trial.number, 'objective_value': trial.value, 'trial_state': trial.state.name, **trial.params}
        pd.DataFrame([trial_data]).to_csv(self.csv_path, mode='a', header=False, index=False)

parser = argparse.ArgumentParser(description='Seleccion de variable')
parser.add_argument('-m', '--module' ,
                    type = str, help = 'modulo de informacion a seleccionar')
parser.add_argument('-n', '--n_trials' ,
                    type = int, help = 'numero de trials para sacar variables optimas')
parser.add_argument('-file', '--file' ,
                    type = str, help = 'ruta donde se encuentra el archivo')
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

# Configurar experimento de MLflow
mlflow.set_experiment("Búsqueda de hiperparametros clean final")

# CSV path
csv_path = f"optuna_trials_{nombre_id}.csv"

# Setup callback
callback = MLflowCallback(csv_path)

# Run optimization within PARENT MLflow run (official pattern)
with mlflow.start_run(run_name=f"hyperparameter_optimization_{nombre_id}"):
    # Log parent run info
    mlflow.log_params({
        "n_trials": args.n_trials,
        "module": args.module if args.module else "default",
        "data_file": args.file,
        "script_name": __file__
    })
    
    # Run optimization with callback (child runs will be created)
    sampler = optuna.samplers.RandomSampler(seed=42)
    study_lgb = optuna.create_study(sampler=sampler, direction="maximize")
    study_lgb.optimize(optimize_lgb, n_trials=args.n_trials, 
                      show_progress_bar=True, callbacks=[callback])
    
    # Get results
    best_params_lgb = study_lgb.best_params
    best_score_lgb = study_lgb.best_value
    
    # Log summary results to parent run
    mlflow.log_metrics({
        "best_score": best_score_lgb,
        "n_trials_completed": len(study_lgb.trials),
        "n_trials_pruned": len([t for t in study_lgb.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "n_trials_failed": len([t for t in study_lgb.trials if t.state == optuna.trial.TrialState.FAIL])
    })
    mlflow.log_params({f"best_{k}": v for k, v in best_params_lgb.items()})
    
    # Log CSV as artifact to parent run
    mlflow.log_artifact(csv_path, "optimization_data")

print(f"\nBest Score: {best_score_lgb}")
print(f"Best Params: {best_params_lgb}")
print(f"Results saved to: {csv_path}")

# Entrenar y guardar modelo final
with mlflow.start_run(run_name=f"final_model_{nombre_id}"):
    # Entrenar modelo final con mejores parámetros
    lgbm = LGBMClassifier(**best_params_lgb)
    lgbm.fit(  
                use_x(use_flag(df_total,'train'),use_flag=False), 
                use_y(use_flag(df_total,'train'),use_flag=False)
            )

    # Crear estructura de carpetas
    nombre_carpeta = f'lgbm_{nombre_id}'
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

print(f"Experiment completed! Final model and results saved.")