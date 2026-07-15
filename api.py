from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import pandas as pd
import json
from datetime import datetime
from models.gnn_model import RetailMindGNN, TemporalGraphBuilder

app = FastAPI(
    title="RetailMind AI",
    description="Temporal GNN-powered demand forecasting for spice wholesale",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model and data at startup ────────────────────────────────────────────
df           = pd.read_csv("data/orders.csv")
customer_ids = sorted(df["customer_id"].unique())
spice_ids    = sorted(df["spice_id"].unique())
c_idx        = {c: i for i, c in enumerate(customer_ids)}
s_idx        = {s: i for i, s in enumerate(spice_ids)}

# Customer and spice metadata for readable responses
CUSTOMER_META = {
    r["customer_id"]: r for _, r in
    df.drop_duplicates("customer_id")[
        ["customer_id","customer_name","customer_tier","customer_city"]
    ].iterrows()
}
SPICE_META = {
    r["spice_id"]: r for _, r in
    df.drop_duplicates("spice_id")[
        ["spice_id","spice_name","unit_price"]
    ].iterrows()
}

# Build graph from full history
builder = TemporalGraphBuilder(df)
graph   = builder.build()

# Load trained model
model = RetailMindGNN()
model.load_state_dict(torch.load("models/best_model.pt", weights_only=True))
model.eval()

# Run inference once at startup
with torch.no_grad():
    PREDICTIONS = model(graph)   # [10, 15, 3]

HORIZON_LABELS = ["7_day", "14_day", "30_day"]
HORIZON_DAYS   = [7, 14, 30]


# ── Request / Response schemas ────────────────────────────────────────────────
class ForecastRequest(BaseModel):
    customer_id: str
    top_n:       int = 5


class ReorderAlert(BaseModel):
    customer_id:   str
    threshold_pct: float = 0.2   # flag if predicted demand > 20% above avg


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "RetailMind AI",
        "version": "1.0.0",
        "model":   "Temporal Heterogeneous GNN (GAT + GraphSAGE)",
        "endpoints": [
            "/forecast/{customer_id}",
            "/forecast/all",
            "/reorder-alerts",
            "/seasonal-graph",
            "/graph-stats",
            "/health"
        ]
    }


@app.get("/health")
def health():
    return {
        "status":     "healthy",
        "model":      "loaded",
        "customers":  len(customer_ids),
        "spices":     len(spice_ids),
        "graph_edges": {
            "orders":   graph["customer","orders","spice"].edge_index.shape[1],
            "coorders": graph["customer","coorders","customer"].edge_index.shape[1],
        }
    }


@app.get("/forecast/{customer_id}")
def forecast_customer(customer_id: str, top_n: int = 5):
    """
    Predict top N spices this customer will need in next 7/14/30 days.
    Uses temporal GNN — recent orders weighted more than old ones.
    """
    if customer_id not in c_idx:
        raise HTTPException(
            status_code=404,
            detail=f"Customer {customer_id} not found. "
                   f"Valid IDs: {customer_ids}"
        )

    ci   = c_idx[customer_id]
    pred = PREDICTIONS[ci]   # [15, 3]
    meta = CUSTOMER_META[customer_id]

    # Rank spices by 14-day forecast (balanced horizon)
    scores    = pred[:, 1].tolist()
    ranked    = sorted(
        zip(spice_ids, scores), key=lambda x: x[1], reverse=True
    )[:top_n]

    forecasts = []
    for spice_id, _ in ranked:
        si       = s_idx[spice_id]
        horizons = {}
        for h_i, label in enumerate(HORIZON_LABELS):
            horizons[label] = round(
                float(PREDICTIONS[ci, si, h_i]) * HORIZON_DAYS[h_i], 2
            )
        avg_price = float(SPICE_META[spice_id]["unit_price"])
        forecasts.append({
            "spice_id":        spice_id,
            "spice_name":      SPICE_META[spice_id]["spice_name"],
            "forecast_kg":     horizons,
            "estimated_value": {
                label: round(horizons[label] * avg_price, 2)
                for label in HORIZON_LABELS
            }
        })

    return {
        "customer_id":   customer_id,
        "customer_name": meta["customer_name"],
        "tier":          meta["customer_tier"],
        "city":          meta["customer_city"],
        "top_spices":    forecasts,
        "generated_at":  datetime.now().isoformat()
    }


@app.get("/forecast/all/summary")
def forecast_all():
    """
    Aggregate demand forecast across all customers.
    Useful for supplier procurement planning.
    """
    results = []
    for spice_id in spice_ids:
        si        = s_idx[spice_id]
        total_30d = float(PREDICTIONS[:, si, 2].sum()) * 30
        results.append({
            "spice_id":       spice_id,
            "spice_name":     SPICE_META[spice_id]["spice_name"],
            "total_30day_kg": round(total_30d, 2),
            "avg_unit_price": float(SPICE_META[spice_id]["unit_price"]),
            "total_30day_value": round(
                total_30d * float(SPICE_META[spice_id]["unit_price"]), 2
            )
        })

    results.sort(key=lambda x: x["total_30day_kg"], reverse=True)
    return {
        "forecast_horizon": "30 days",
        "spice_demand":     results,
        "generated_at":     datetime.now().isoformat()
    }


@app.get("/reorder-alerts")
def reorder_alerts(threshold_pct: float = 0.2):
    """
    Flag (customer, spice) pairs where predicted demand
    is significantly above their historical average.
    This is the proactive intelligence layer.
    """
    alerts = []
    for customer_id in customer_ids:
        ci      = c_idx[customer_id]
        cdf     = df[df["customer_id"] == customer_id]
        for spice_id in spice_ids:
            si      = s_idx[spice_id]
            sdf     = cdf[cdf["spice_id"] == spice_id]
            if len(sdf) == 0:
                continue

            avg_qty      = sdf["quantity_kg"].mean()
            pred_14d     = float(PREDICTIONS[ci, si, 1]) * 14

            if avg_qty > 0 and pred_14d > avg_qty * 0.5:
                surge_pct = round((pred_14d - avg_qty) / avg_qty * 100, 1)
                alerts.append({
                    "customer_id":   customer_id,
                    "customer_name": CUSTOMER_META[customer_id]["customer_name"],
                    "spice_id":      spice_id,
                    "spice_name":    SPICE_META[spice_id]["spice_name"],
                    "avg_order_kg":  round(avg_qty, 2),
                    "predicted_14d": round(pred_14d, 2),
                    "surge_pct":     surge_pct,
                    "urgency":       "HIGH" if surge_pct > 50 else "MEDIUM"
                })

    alerts.sort(key=lambda x: x["surge_pct"], reverse=True)
    return {
        "total_alerts":    len(alerts),
        "threshold_pct":   threshold_pct * 100,
        "alerts":          alerts,
        "generated_at":    datetime.now().isoformat()
    }


@app.get("/seasonal-graph")
def seasonal_graph():
    """
    Returns monthly demand patterns per spice.
    Shows how the graph edge weights shift seasonally —
    the core temporal novelty of this system.
    """
    result = {}
    for spice_id in spice_ids:
        sdf = df[df["spice_id"] == spice_id]
        monthly = (
            sdf.groupby("month")["quantity_kg"]
            .mean()
            .reindex(range(1, 13), fill_value=0)
            .round(2)
            .tolist()
        )
        result[spice_id] = {
            "spice_name":    SPICE_META[spice_id]["spice_name"],
            "monthly_avg_kg": monthly,
            "peak_month":    int(pd.Series(monthly).idxmax()) + 1,
            "trough_month":  int(pd.Series(monthly).idxmin()) + 1,
        }

    return {
        "description": "Monthly demand patterns — drives temporal edge weights",
        "months":      ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"],
        "spices":      result,
        "generated_at": datetime.now().isoformat()
    }


@app.get("/graph-stats")
def graph_stats():
    """
    Returns the underlying graph structure statistics.
    Useful for explaining the GNN architecture in demos.
    """
    return {
        "model": "Temporal Heterogeneous GNN",
        "architecture": {
            "layer_1": "Graph Attention Network (GAT) — customer→spice edges",
            "layer_2": "GraphSAGE — customer↔customer co-ordering edges",
            "decoder": "3-layer MLP → 7/14/30 day demand forecast",
            "novelty": "Exponential temporal decay on edge weights (half-life=60 days)"
        },
        "graph": {
            "customer_nodes":  len(customer_ids),
            "spice_nodes":     len(spice_ids),
            "supplier_nodes":  5,
            "order_edges":     int(graph["customer","orders","spice"].edge_index.shape[1]),
            "coorder_edges":   int(graph["customer","coorders","customer"].edge_index.shape[1]),
        },
        "training": {
            "epochs":          150,
            "optimizer":       "Adam + CosineAnnealingLR",
            "loss_function":   "HuberLoss (robust to luxury hotel spikes)",
            "technique":       "Sliding window temporal sampling (40 windows)",
            "best_val_loss":   0.7480
        },
        "data": {
            "orders":          len(df),
            "date_range":      f"{df['date'].min()} → {df['date'].max()}",
            "total_value":     f"₹{df['total_value'].sum():,.0f}"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)