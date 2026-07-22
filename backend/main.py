from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.models.database import init_db
from backend.routers import predict, history, feedback


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup logic
    init_db()
    yield
    # shutdown logic (nothing to clean up yet)


app = FastAPI(title="Smart Waste Robot API", lifespan=lifespan)

app.include_router(predict.router)
app.include_router(history.router)
app.include_router(feedback.router)


@app.get("/")
def root():
    return {"status": "ok"}
