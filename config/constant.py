import os
from dotenv import load_dotenv

load_dotenv(override = True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



apex_model_name = "Apex_Trust_model"





import os
from dotenv import load_dotenv

load_dotenv(override=True)

# =========================
# BASE DIRECTORY
# =========================
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# MONGODB CONFIG

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DATABASE = "Apex_trust_customer_transactions"
MONGO_COLLECTION = "Apex_Customer"


# DATA PATHS

INPUT_DATA_PATH = os.path.join(BASE_DIR, "dataset", "apex_transactions.csv")
# Cleaned_Data = os.path.join(BASE_DIR, "Data", "cleaned_data.csv")

# =========================
# MODEL CONFIG
# =========================
APEX_MODEL_NAME = "Apex_Trust_Model"