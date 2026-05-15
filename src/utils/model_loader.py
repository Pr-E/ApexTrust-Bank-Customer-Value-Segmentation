# 5 OR 6 
import logging
import mlflow
from mlflow.tracking import MlflowClient

from src.utils.mlflow_config import setup_mlflow
from config.constant import APEX_MODEL_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_model_and_params():
    try:
        # =========================
        # INITIALIZE MLFLOW
        # =========================
        setup_mlflow()

        client = MlflowClient()

        model_name = APEX_MODEL_NAME

        # =========================
        # FETCH LATEST MODEL VERSIONS
        # =========================
        latest_versions = client.get_latest_versions(
            name=model_name,
            stages=["None", "Staging", "Production"]
        )

        if not latest_versions:
            raise ValueError(
                f"No registered model found for: {model_name}"
            )

        # =========================
        # GET MOST RECENT VERSION
        # =========================
        latest_version = max(
            latest_versions,
            key=lambda x: int(x.version)
        )

        model_uri = f"models:/{model_name}/{latest_version.version}"

        # =========================
        # LOAD MODEL
        # =========================
        model = mlflow.sklearn.load_model(model_uri)

        # =========================
        # LOAD RUN PARAMETERS
        # =========================
        run_id = latest_version.run_id

        run = client.get_run(run_id)

        optimal_k = int(
            run.data.params.get("optimal_k")
        )

        logging.info(
            f"Model loaded successfully: "
            f"{model_name} v{latest_version.version}"
        )

        logging.info(f"Run ID: {run_id}")
        logging.info(f"Optimal K: {optimal_k}")

        return model, optimal_k

    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


def is_new_model_better(
    new_score,
    experiment_name="Default"
):
    try:
        client = MlflowClient()

        # =========================
        # FETCH EXPERIMENT
        # =========================
        experiment = client.get_experiment_by_name(
            experiment_name
        )

        if experiment is None:
            logging.info(
                "No existing experiment found. "
                "New model will be accepted."
            )
            return True, None

        # =========================
        # FETCH PREVIOUS RUNS
        # =========================
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.silhouette_score DESC"]
        )

        if not runs:
            logging.info(
                "No previous runs found. "
                "New model will be accepted."
            )
            return True, None

        # =========================
        # BEST PREVIOUS SCORE
        # =========================
        previous_best = runs[0].data.metrics.get(
            "silhouette_score"
        )

        if previous_best is None:
            return True, None

        logging.info(
            f"Previous Best Silhouette Score: "
            f"{previous_best}"
        )

        logging.info(
            f"New Silhouette Score: {new_score}"
        )

        return new_score > previous_best, previous_best

    except Exception as e:
        logging.error(
            f"Error comparing model performance: {e}"
        )
        raise