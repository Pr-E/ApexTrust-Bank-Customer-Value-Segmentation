import dagshub
import mlflow
import os
from dotenv import load_dotenv
load_dotenv(override = True)

# def setup_mlflow():
  #  dagshub.init(
      #  repo_owner="ejirogoro27",
       # repo_name= "ApexTrust-Bank-Customer-Value-Segmentation",
        #"ApexTrust-Customer-Segmentation",
   #     mlflow=True
    #)

    #mlflow.set_experiment("customer_value_segmentation")
    


def setup_mlflow():
    dagshub_token = os.getenv("MLFLOW_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("The Mlflow token cant't be accessed...")
    
    os.environ["MLFLOE_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    repo_owner = "ejirogoro27"
    repo_name =  "ApexTrust-Bank-Customer-Value-Segmentation"

    tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("customer_value_segmentation")
