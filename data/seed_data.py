import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

random.seed(42)
np.random.seed(42)

# ── Kerala spice wholesale catalogue ──────────────────────────────────────────
SPICES = [
    {"id": "SP001", "name": "Cardamom",      "base_price": 2800, "unit": "kg"},
    {"id": "SP002", "name": "Black Pepper",  "base_price": 650,  "unit": "kg"},
    {"id": "SP003", "name": "Turmeric",      "base_price": 180,  "unit": "kg"},
    {"id": "SP004", "name": "Cinnamon",      "base_price": 420,  "unit": "kg"},
    {"id": "SP005", "name": "Cloves",        "base_price": 1100, "unit": "kg"},
    {"id": "SP006", "name": "Ginger",        "base_price": 220,  "unit": "kg"},
    {"id": "SP007", "name": "Coriander",     "base_price": 130,  "unit": "kg"},
    {"id": "SP008", "name": "Cumin",         "base_price": 310,  "unit": "kg"},
    {"id": "SP009", "name": "Fenugreek",     "base_price": 95,   "unit": "kg"},
    {"id": "SP010", "name": "Saffron",       "base_price": 45000,"unit": "kg"},
    {"id": "SP011", "name": "Star Anise",    "base_price": 890,  "unit": "kg"},
    {"id": "SP012", "name": "Mustard Seeds", "base_price": 85,   "unit": "kg"},
    {"id": "SP013", "name": "Red Chilli",    "base_price": 195,  "unit": "kg"},
    {"id": "SP014", "name": "Mace",          "base_price": 1800, "unit": "kg"},
    {"id": "SP015", "name": "Nutmeg",        "base_price": 950,  "unit": "kg"},
]

# ── Hotel customers ───────────────────────────────────────────────────────────
CUSTOMERS = [
    {"id": "C001", "name": "Taj Malabar Resort",      "tier": "luxury",   "city": "Kochi"},
    {"id": "C002", "name": "Leela Kovalam",            "tier": "luxury",   "city": "Kovalam"},
    {"id": "C003", "name": "Casino Hotel",             "tier": "premium",  "city": "Kochi"},
    {"id": "C004", "name": "Gokulam Park",             "tier": "premium",  "city": "Kozhikode"},
    {"id": "C005", "name": "Raviz Ashtamudi",          "tier": "luxury",   "city": "Kollam"},
    {"id": "C006", "name": "Hotel Malabar Palace",     "tier": "standard", "city": "Kozhikode"},
    {"id": "C007", "name": "Brunton Boatyard",         "tier": "luxury",   "city": "Kochi"},
    {"id": "C008", "name": "Uday Samudra",             "tier": "premium",  "city": "Kovalam"},
    {"id": "C009", "name": "Spree Shenbagam",          "tier": "standard", "city": "Thrissur"},
    {"id": "C010", "name": "Trident Cochin",           "tier": "luxury",   "city": "Kochi"},
]

# ── Suppliers ─────────────────────────────────────────────────────────────────
SUPPLIERS = [
    {"id": "SUP001", "name": "Wayanad Spice Growers Co-op", "region": "Wayanad"},
    {"id": "SUP002", "name": "Idukki Hill Spices",          "region": "Idukki"},
    {"id": "SUP003", "name": "Malabar Traders",             "region": "Kozhikode"},
    {"id": "SUP004", "name": "Kerala Spice Exchange",       "region": "Kochi"},
    {"id": "SUP005", "name": "Western Ghats Organics",      "region": "Munnar"},
]

# ── Seasonal multipliers (Kerala festival & tourism calendar) ─────────────────
# Month: 1=Jan ... 12=Dec
SEASONAL_DEMAND = {
    "Cardamom":      [0.9, 0.8, 0.8, 0.7, 0.7, 0.8, 0.9, 1.1, 1.4, 1.6, 1.5, 1.3],
    "Black Pepper":  [1.0, 0.9, 0.9, 0.8, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.5, 1.3],
    "Saffron":       [0.7, 0.7, 0.8, 0.9, 0.8, 0.7, 0.8, 1.0, 1.3, 1.2, 1.4, 1.5],
    "Turmeric":      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.3, 1.2, 1.1],
    "Ginger":        [1.1, 1.0, 0.9, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.3, 1.2, 1.2],
}

DEFAULT_SEASONAL = [1.0] * 12


def get_seasonal_multiplier(spice_name: str, month: int) -> float:
    pattern = SEASONAL_DEMAND.get(spice_name, DEFAULT_SEASONAL)
    return pattern[month - 1]


def get_tier_multiplier(tier: str) -> float:
    return {"luxury": 3.5, "premium": 2.0, "standard": 1.0}[tier]


def generate_orders(start_date: str = "2023-01-01",
                    end_date: str   = "2024-12-31") -> pd.DataFrame:
    """
    Generate synthetic B2B spice orders.
    Each hotel places orders roughly every 2-3 weeks per spice it regularly uses.
    Luxury hotels order more SKUs and larger quantities.
    Seasonal multipliers simulate Kerala festival/tourism demand cycles.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")
    records = []

    for customer in CUSTOMERS:
        tier_mult = get_tier_multiplier(customer["tier"])

        # Each hotel regularly uses a subset of spices
        n_spices = {"luxury": 12, "premium": 9, "standard": 6}[customer["tier"]]
        hotel_spices = random.sample(SPICES, n_spices)

        for spice in hotel_spices:
            supplier = random.choice(SUPPLIERS)
            current  = start + timedelta(days=random.randint(0, 14))

            while current <= end:
                month        = current.month
                season_mult  = get_seasonal_multiplier(spice["name"], month)

                base_qty     = random.uniform(5, 20) * tier_mult * season_mult
                quantity     = round(base_qty + np.random.normal(0, base_qty * 0.1), 2)
                quantity     = max(1.0, quantity)

                price_noise  = random.uniform(0.95, 1.05)
                unit_price   = round(spice["base_price"] * price_noise, 2)
                total_value  = round(quantity * unit_price, 2)

                records.append({
                    "order_id":      f"ORD-{len(records)+1:05d}",
                    "date":          current.strftime("%Y-%m-%d"),
                    "month":         month,
                    "week":          current.isocalendar()[1],
                    "year":          current.year,
                    "customer_id":   customer["id"],
                    "customer_name": customer["name"],
                    "customer_tier": customer["tier"],
                    "customer_city": customer["city"],
                    "spice_id":      spice["id"],
                    "spice_name":    spice["name"],
                    "supplier_id":   supplier["id"],
                    "supplier_name": supplier["name"],
                    "quantity_kg":   quantity,
                    "unit_price":    unit_price,
                    "total_value":   total_value,
                    "season_mult":   season_mult,
                })

                # Next order: 14-21 days later with slight randomness
                gap      = random.randint(14, 21)
                current += timedelta(days=gap)

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return df


def save_data(df: pd.DataFrame, path: str = "data/orders.csv") -> None:
    df.to_csv(path, index=False)
    print(f"✅ Saved {len(df)} orders → {path}")
    print(f"   Date range  : {df['date'].min()} → {df['date'].max()}")
    print(f"   Customers   : {df['customer_id'].nunique()}")
    print(f"   Spices      : {df['spice_id'].nunique()}")
    print(f"   Suppliers   : {df['supplier_id'].nunique()}")
    print(f"   Total value : ₹{df['total_value'].sum():,.0f}")


if __name__ == "__main__":
    df = generate_orders()
    save_data(df)
    print("\n── Sample orders ──")
    print(df.head(10).to_string(index=False))