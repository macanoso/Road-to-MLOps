import argparse
import logging
import os
import pickle
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import mlflow
import mlflow.lightgbm
import optuna
import pandas as pd
from lightgbm import LGBMClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loss_func import funcion_costo
from model_reviews import ModelReviewTotal
from utils import use_flag, use_x, use_y


class TrialLogger:
    """Handles logging of Optuna trials to MLflow and CSV with fallback mechanisms."""
    
    def __init__(self, experiment_name: str, script_name: str):
        self.experiment_name = experiment_name
        self.script_name = script_name
        self.trial_results = []
        self.mlflow_enabled = True
        self.csv_path = f"optuna_trials_{os.path.splitext(os.path.basename(script_name))[0]}.csv"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize MLflow
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow experiment with error handling."""
        try:
            mlflow.set_experiment(self.experiment_name)
            self.logger.info(f"MLflow experiment '{self.experiment_name}' initialized")
        except Exception as e:
            self.logger.warning(f"MLflow setup failed: {e}. Using CSV fallback only.")
            self.mlflow_enabled = False
    
    def log_trial(self, trial: optuna.Trial, trial_params: Dict[str, Any], 
                  additional_info: Optional[Dict[str, Any]] = None):
        """Log a single trial result to MLflow and store for CSV backup."""
        
        # Prepare trial data
        trial_data = {
            'trial_number': trial.number,
            'objective_value': trial.value,
            'trial_state': trial.state.name,
            'timestamp': datetime.now().isoformat(),
            **trial_params
        }
        
        if additional_info:
            trial_data.update(additional_info)
        
        # Store for CSV backup
        self.trial_results.append(trial_data.copy())
        
        # Log to MLflow if enabled
        if self.mlflow_enabled:
            self._log_to_mlflow(trial_data)
        
        # Save CSV after each trial (as backup)
        self._save_csv()
    
    def _log_to_mlflow(self, trial_data: Dict[str, Any]):
        """Log trial data to MLflow with error handling."""
        try:
            # Create a unique run for each trial
            run_name = f"trial_{trial_data['trial_number']:04d}"
            
            with mlflow.start_run(run_name=run_name, nested=False):
                # Separate params and metrics
                params = {k: v for k, v in trial_data.items() 
                         if k not in ['objective_value', 'timestamp']}
                metrics = {'objective_value': trial_data['objective_value']}
                
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                
        except Exception as e:
            self.logger.warning(f"MLflow logging failed for trial {trial_data['trial_number']}: {e}")
            # Continue execution, data is still saved in CSV
    
    def _save_csv(self):
        """Save trial results to CSV."""
        try:
            df = pd.DataFrame(self.trial_results)
            df.to_csv(self.csv_path, index=False)
        except Exception as e:
            self.logger.error(f"Failed to save CSV backup: {e}")
    
    def log_optimization_summary(self, study: optuna.Study, n_trials: int, 
                                module: str, data_file: str):
        """Log final optimization summary."""
        summary_data = {
            "script_name": self.script_name,
            "n_trials": n_trials,
            "module": module,
            "data_file": data_file,
            "best_score": study.best_value,
            "n_trials_completed": len(study.trials),
            "n_trials_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "n_trials_failed": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        }
        
        # Add best parameters
        best_params = {f"best_{k}": v for k, v in study.best_params.items()}
        summary_data.update(best_params)
        
        if self.mlflow_enabled:
            try:
                with mlflow.start_run(run_name=f"optimization_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                    params = {k: v for k, v in summary_data.items() if k.startswith(('script_name', 'module', 'data_file', 'best_'))}
                    metrics = {k: v for k, v in summary_data.items() if k.startswith(('best_score', 'n_trials'))}
                    
                    mlflow.log_params(params)
                    mlflow.log_metrics(metrics)
                    
                    # Log CSV as artifact
                    mlflow.log_artifact(self.csv_path, "optimization_data")
                    
            except Exception as e:
                self.logger.warning(f"Failed to log optimization summary to MLflow: {e}")
        
        return summary_data
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get all trial results as a DataFrame."""
        return pd.DataFrame(self.trial_results)


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

# Initialize logger
script_name = __file__
logger = TrialLogger(
    experiment_name="Búsqueda de hiperparametros clean final",
    script_name=script_name
)

def optimize_lgb(trial):
    boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf'])
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    learning_rate = trial.suggest_float('learning_rate', 0.05, 0.25)
    num_leaves = trial.suggest_int('num_leaves', 2, 200)
    max_depth = trial.suggest_int('max_depth', 2, 20)
    min_child_weight = trial.suggest_float('min_child_weight', 0.05, 0.1)
    min_child_samples = trial.suggest_int('min_child_samples', 5, 50)
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

    # Log trial after completion
    logger.log_trial(trial, trial.params, {"costo": costo})
    
    return costo


def optimize_and_log():
    """Run optimization with proper logging."""
    print("Starting hyperparameter optimization...")
    
    # Create Optuna study
    sampler = optuna.samplers.RandomSampler(seed=42)
    study_lgb = optuna.create_study(sampler=sampler, direction="maximize")
    
    # Run optimization without callbacks (logging handled inside objective function)
    study_lgb.optimize(optimize_lgb, n_trials=args.n_trials, show_progress_bar=True)
    
    # Log optimization summary
    summary = logger.log_optimization_summary(
        study=study_lgb,
        n_trials=args.n_trials,
        module=args.module if args.module else "default",
        data_file=args.file
    )
    
    print('LGBMClassifier Best Params:', study_lgb.best_params)
    print('LGBMClassifier Best Score:', study_lgb.best_value)
    print(f"Results saved to: {logger.csv_path}")
    
    return study_lgb, summary

def train_and_save_final_model(study: optuna.Study):
    """Train and save the final model with best parameters."""
    print("Training final model...")
    
    # Get best parameters
    best_params_lgb = study.best_params
    best_score_lgb = study.best_value
    nombre_id = os.path.splitext(os.path.basename(script_name))[0]
    
    if logger.mlflow_enabled:
        try:
            with mlflow.start_run(run_name=f"final_model_{nombre_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Train final model
                lgbm = LGBMClassifier(**best_params_lgb)
                lgbm.fit(  
                            use_x(use_flag(df_total,'train'),use_flag=False), 
                            use_y(use_flag(df_total,'train'),use_flag=False)
                        )

                # Create directory structure
                nombre_carpeta = f'lgbm_{nombre_id}'
                os.makedirs(f'./MODELS/{nombre_carpeta}', exist_ok=True)

                # Save model locally
                filename = f'./MODELS/{nombre_carpeta}/lgbm_{nombre_id}.pkl'
                pickle.dump(lgbm, open(filename, 'wb'))

                # Generate model review
                review = ModelReviewTotal(
                                            modelo=lgbm,
                                            x=use_x(df_total),
                                            y=use_y(df_total),
                                            df_completo=df_total,
                                            fa_column='fi',
                                            id_column='cis',
                                            bases=['train','test','oot1','oot2']
                                            )

                # Generate plots
                review.graficar_mapeo_deciles(clase=30,
                                            save_path=f"./MODELS/{nombre_carpeta}/percentil_30_modelo_{nombre_id}.png")
                review.graficar_mapeo_deciles(clase=50,
                                            save_path=f"./MODELS/{nombre_carpeta}/percentil_50_modelo_{nombre_id}.png")
                
                # Get metrics from review
                review.auc_gini(f"./MODELS/{nombre_carpeta}/")
                review.revisar_tl(f"./MODELS/{nombre_carpeta}/")
                
                # Log final model information
                mlflow.log_params({f"final_{k}": v for k, v in best_params_lgb.items()})
                mlflow.log_metric("final_best_score", best_score_lgb)
                
                # Log model using LightGBM flavor
                mlflow.lightgbm.log_model(lgbm, "model")
                
                # Log model artifacts
                mlflow.log_artifacts(f"./MODELS/{nombre_carpeta}", "model_outputs")
                
                print(f"Final model logged to MLflow and saved locally")
                
        except Exception as e:
            logger.logger.warning(f"Failed to log final model to MLflow: {e}")
            # Continue with local saving even if MLflow fails
            _save_model_locally(best_params_lgb, nombre_id)
    else:
        _save_model_locally(best_params_lgb, nombre_id)

def _save_model_locally(best_params_lgb, nombre_id):
    """Save model locally when MLflow is not available."""
    lgbm = LGBMClassifier(**best_params_lgb)
    lgbm.fit(  
                use_x(use_flag(df_total,'train'),use_flag=False), 
                use_y(use_flag(df_total,'train'),use_flag=False)
            )
    
    nombre_carpeta = f'lgbm_{nombre_id}'
    os.makedirs(f'./MODELS/{nombre_carpeta}', exist_ok=True)
    filename = f'./MODELS/{nombre_carpeta}/lgbm_{nombre_id}.pkl'
    pickle.dump(lgbm, open(filename, 'wb'))
    print(f"Model saved locally to {filename}")

# Main execution
if __name__ == "__main__":
    # Run optimization
    study, summary = optimize_and_log()
    
    # Train and save final model
    train_and_save_final_model(study)
    
    print(f"Optimization completed! Check {logger.csv_path} for detailed results.")