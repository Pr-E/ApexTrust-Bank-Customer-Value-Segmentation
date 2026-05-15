# =========================================
# APEX TRUST MAIN APPLICATION
# =========================================

from fastapi import FastAPI

from src.api.routes import router

# =========================================
# FASTAPI APPLICATION
# =========================================
app = FastAPI(

    title="ApexTrust Bank Customer Intelligence API",

    version="1.0.0"
)

# =========================================
# REGISTER ROUTES
# =========================================
app.include_router(router)