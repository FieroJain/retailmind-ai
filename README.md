# RetailMind AI

> Temporal Graph Neural Network for proactive demand forecasting in B2B spice wholesale.

Built as an open-source AI intelligence layer for ERPNext retail deployments.

---

# The Core Idea

Most demand forecasting treats history as a flat table. RetailMind treats it as a **dynamic graph**—hotels, spices, and suppliers are nodes; orders are edges weighted by recency. As seasons change, edge weights decay and rewire, and the Graph Neural Network learns to anticipate those shifts before they happen.

```text
Supplier ───► Spice ───► Customer
                 ▲
                 │
      Temporal Order History
 (edges decay with half-life = 60 days)
```

This makes the system **proactive**, not reactive—it predicts demand before stock shortages occur instead of responding after inventory has already fallen.

---

# Architecture

## Graph Structure

### Nodes

| Node Type | Count | Features |
|-----------|------:|----------|
| Customer (Hotels) | 10 | Tier, average order value, order frequency, peak season |
| Spice SKU | 15 | Price, volatility, demand seasonality |
| Supplier | 5 | SKU coverage, average supply volume |

### Edges

| Edge Type | Count | Weight |
|-----------|------:|--------|
| Customer → Spice (Orders) | 4,147 | Temporal decay × ordered quantity |
| Customer ↔ Customer (Co-ordering) | 7,176 | Shared spices × purchasing month |

---

## Model Pipeline

### Layer 1 — Graph Attention Network (GAT)

Learns which spices influence purchasing decisions for each customer.

### Layer 2 — GraphSAGE

Captures relationships between customers ordering similar spice combinations.

### Decoder

Three-layer MLP producing simultaneous:

- 7-day forecast
- 14-day forecast
- 30-day forecast

### Training Strategy

- Sliding temporal windows (40 windows)
- Huber Loss
- Adam Optimizer
- Cosine Annealing Learning Rate Scheduler

---

# Temporal Edge Decay

Instead of assigning equal importance to historical purchases, RetailMind applies exponential temporal decay:

```math
weight = e^(-days_ago × ln(2) / 60)
```

This means:

- An order placed yesterday contributes almost full weight.
- An order from two months ago contributes half as much.
- Older purchasing behavior gradually fades from the graph.

The graph therefore rewires itself over time, allowing the model to naturally learn seasonal purchasing behavior such as:

- Onam demand
- Christmas hospitality demand
- Wedding season purchasing
- Tourism-driven hotel consumption

without explicitly providing calendar-based features.

---

# Results

| Metric | Value |
|--------|------:|
| Training Windows | 40 |
| Best Validation Loss | 0.748 (Huber Loss) |
| Forecast Horizons | 7 / 14 / 30 Days |
| Reorder Alerts Generated | 12 |
| Customers | 10 Hotels |
| Products | 15 Spice SKUs |
| Synthetic Dataset | 4,147 Orders |
| Simulated GMV | ₹61 Crore |

---

# API Endpoints

Run:

```bash
python api.py
```

Swagger documentation becomes available at:

```
http://localhost:8000/docs
```

| Endpoint | Description |
|----------|-------------|
| `GET /forecast/{customer_id}` | Demand forecast for a specific customer |
| `GET /forecast/all/summary` | Aggregate demand across all customers |
| `GET /reorder-alerts` | Products predicted to require replenishment |
| `GET /seasonal-graph` | Monthly demand statistics |
| `GET /graph-stats` | Graph architecture and training metadata |
| `GET /health` | Service health and graph information |

---

## Example Response

### `GET /forecast/C001`

```json
{
  "customer_name": "Taj Malabar Resort",
  "tier": "luxury",
  "top_spices": [
    {
      "spice_name": "Cardamom",
      "forecast_kg": {
        "7_day": 2.94,
        "14_day": 15.01,
        "30_day": 33.36
      },
      "estimated_value": {
        "30_day": 89936.89
      }
    }
  ]
}
```

---

# Quick Start

Clone the repository:

```bash
git clone https://github.com/FieroJain/retailmind-ai.git
cd retailmind-ai
```

Create a virtual environment:

```bash
python -m venv venv
```

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate synthetic wholesale data:

```bash
python data/seed_data.py
```

Train the Graph Neural Network:

```bash
python train.py
```

Start the API:

```bash
python api.py
```

Open:

```
http://localhost:8000/docs
```

---

# Project Structure

```text
retailmind-ai/
│
├── data/
│   └── seed_data.py
│
├── models/
│   └── gnn_model.py
│
├── api.py
├── train.py
├── requirements.txt
├── README.md
│
└── outputs/
    ├── trained_model.pt
    └── generated_data.csv
```

---

# ERPNext Integration

RetailMind is designed as an independent AI microservice.

Workflow:

```text
ERPNext
    │
    │ REST API
    ▼
RetailMind AI
    │
    ├── Build temporal graph
    ├── Train GNN
    ├── Predict future demand
    └── Generate reorder alerts
    │
    ▼
ERPNext Purchase Suggestions
```

No modifications to ERPNext are required. The system simply consumes ERP data and returns intelligent purchasing recommendations.

---

# Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Deep Learning | PyTorch |
| Graph Learning | PyTorch Geometric |
| API | FastAPI |
| Data Processing | pandas |
| Machine Learning | scikit-learn |
| Numerical Computing | NumPy |

---

# Future Improvements

- Real ERPNext integration
- Live inventory synchronization
- Supplier lead-time prediction
- Dynamic pricing optimization
- Graph Transformer architecture
- Explainable AI using GNNExplainer
- Reinforcement learning for purchasing optimization
- Multi-warehouse forecasting
- LLM-powered procurement assistant

---

# Why Graph Neural Networks?

Traditional forecasting models (ARIMA, Prophet, LSTM) primarily learn from historical time-series data.

RetailMind instead models the wholesale ecosystem as a graph.

This enables the model to learn:

- Customer purchasing communities
- Product relationships
- Supplier influence
- Seasonal transitions
- Hidden purchasing patterns
- Cross-product dependencies

making demand prediction significantly more context-aware than standard forecasting approaches.

---

# License

This project is released for educational and research purposes.

---

## Author

**Fiero Jain**

AI Engineer | Graph Machine Learning | FastAPI | PyTorch | ERPNext Intelligence

GitHub: https://github.com/FieroJain

---

> *RetailMind AI explores how Temporal Graph Neural Networks can provide proactive demand forecasting and intelligent inventory recommendations for ERP-based retail and wholesale ecosystems.*