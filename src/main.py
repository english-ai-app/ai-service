from fastapi import FastAPI

from src.api.routes import router


app = FastAPI(title="English App AI Service")
app.include_router(router)
