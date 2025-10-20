from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..core.strategy import fetch_stock_data
from ..schemas import FetchRequest

router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
@router.post('/fetch')
def fetch(req: FetchRequest):
    try:
        df = fetch_stock_data(
            symbol=req.symbol,
            start_date=req.start_date,
            end_date=req.end_date,
            period=req.period or '3y',
            save_local=req.save_local,
            data_dir=str(DATA_DIR),
            use_adjusted=req.use_adjusted
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    resp = {
        "symbol": req.symbol,
        "rows": int(len(df)),
        "start": str(df.index.min().date()) if len(df) else None,
        "end": str(df.index.max().date()) if len(df) else None,
    }
    if req.return_data:
        data = df.reset_index().tail(req.max_rows)
        resp["data"] = data.to_dict(orient="records")
    return resp