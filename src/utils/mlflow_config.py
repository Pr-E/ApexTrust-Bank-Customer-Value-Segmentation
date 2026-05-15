import dagshub
import mlflow

def setup_mlflow():
    dagshub.init(
        repo_owner="ejirogoro27",
        repo_name= "ApexTrust-Bank-Customer-Value-Segmentation",
        #"ApexTrust-Customer-Segmentation",
        mlflow=True
    )

    mlflow.set_experiment("customer_value_segmentation")
    