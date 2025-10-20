from fastapi import FastAPI

from app.routers import data, backtest, plots

app = FastAPI(title="DMA Strategy Backtest API", version="1.0.0")

@app.get('/health')
def health():
    return {"status": "ok"}

app.include_router(data.router, prefix="/data", tags=["data"])
app.include_router(backtest.router, prefix="/backtest", tags=["backtest"])
app.include_router(plots.router, prefix="/plots", tags=["plots"])
