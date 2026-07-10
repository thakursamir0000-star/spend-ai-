"""Generate 6 months of realistic synthetic transaction data for testing."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

start_date = datetime(2026, 1, 1)
end_date = datetime(2026, 6, 30)

MERCHANTS = {
    "Food & Dining": [
        ("Swiggy", 150, 600), ("Zomato", 200, 700), ("Dominos", 300, 800),
        ("McDonalds", 200, 500), ("Local Cafe", 100, 350), ("Dosa Plaza", 120, 300),
        ("Burger King", 250, 600), ("Pizza Hut", 400, 900), ("Subway", 200, 450)
    ],
    "Groceries": [
        ("Big Basket", 500, 2000), ("Blinkit", 200, 1500), ("Zepto", 300, 1200),
        ("Reliance Fresh", 400, 2500), ("DMart", 800, 3500), ("Local Grocer", 200, 1000)
    ],
    "Transport": [
        ("Uber", 100, 500), ("Ola", 120, 450), ("Indian Oil", 1000, 3000),
        ("BPCL", 800, 2500), ("Metro Card", 30, 100), ("Parking", 20, 100),
        ("Rapido", 30, 150), ("Uber Pool", 80, 250)
    ],
    "Shopping": [
        ("Amazon", 300, 5000), ("Flipkart", 400, 4000), ("Myntra", 500, 3000),
        ("Ajio", 400, 2500), ("Local Store", 200, 2000), ("IKEA", 1000, 8000),
        ("Decathlon", 500, 4000), ("Nykaa", 300, 2000)
    ],
    "Subscriptions": [
        ("Netflix", 499, 1500), ("Spotify", 119, 200), ("Amazon Prime", 299, 1500),
        ("Hotstar", 299, 1500), ("YouTube Premium", 129, 200),
        ("Gym Membership", 1500, 3000), ("iCloud", 75, 300),
        ("Google One", 130, 500)
    ],
    "Bills & Utilities": [
        ("Electricity Board", 500, 3000), ("Water Bill", 200, 1000),
        ("Jio Recharge", 299, 1000), ("Airtel Recharge", 299, 1000),
        ("Broadband Bill", 600, 1500), ("Rent Payment", 8000, 25000),
        ("Society Maintenance", 1500, 4000)
    ],
    "Entertainment": [
        ("BookMyShow", 200, 800), ("PVR", 400, 1500), ("Steam", 300, 3000),
        ("PlayStation Store", 500, 4000), ("Zomato Dining", 500, 2000)
    ],
    "Healthcare": [
        ("Apollo Pharmacy", 200, 1500), ("Practo", 300, 1000),
        ("Doctor Visit", 500, 2000), ("Health Checkup", 1000, 5000),
        ("Dental Clinic", 500, 3000)
    ],
    "Transfers": [
        ("UPI to Friend", 100, 3000), ("Credit Card Payment", 5000, 30000),
        ("FD Deposit", 5000, 25000), ("Mutual Fund", 1000, 10000),
        ("Rent Transfer", 8000, 25000)
    ],
    "Other": [
        ("Misc Expense", 50, 1000), ("Cash Withdrawal", 500, 5000),
        ("ATM Fee", 20, 30), ("Random", 100, 500)
    ]
}

CATEGORY_WEIGHTS = {
    "Food & Dining": 30, "Groceries": 15, "Transport": 12,
    "Shopping": 10, "Subscriptions": 5, "Bills & Utilities": 8,
    "Entertainment": 5, "Healthcare": 3, "Transfers": 7, "Other": 5
}

HOLIDAY_MONTHS = {1: "New Year", 3: "Holi", 4: "Summer Vacations", 10: "Diwali", 12: "Christmas"}

transactions = []
current = start_date
txn_id = 1

while current <= end_date:
    n_txns = random.randint(3, 10)

    if current.weekday() >= 5:
        n_txns += random.randint(2, 5)

    for _ in range(n_txns):
        cat = random.choices(list(CATEGORY_WEIGHTS.keys()), weights=list(CATEGORY_WEIGHTS.values()))[0]
        merchant_name, min_amt, max_amt = random.choice(MERCHANTS[cat])

        if cat == "Food & Dining" and current.weekday() >= 5:
            max_amt = int(max_amt * 1.5)

        amount = round(random.uniform(min_amt, max_amt), 0)

        if cat == "Transfers":
            amount = round(amount * random.choice([0.5, 1, 2]), 0)

        hour = random.randint(6, 23)
        minute = random.randint(0, 59)
        txn_datetime = current + timedelta(hours=hour, minutes=minute)

        description = f"{merchant_name} {current.strftime('%d %b')}"

        transactions.append({
            "Date": txn_datetime.strftime("%d-%m-%Y"),
            "Description": description,
            "Merchant": merchant_name,
            "Amount": amount,
            "Category": cat
        })
        txn_id += 1

    if current.day == 1:
        for _ in range(random.randint(1, 3)):
            cat = "Subscriptions"
            merchant_name, min_amt, max_amt = random.choice(MERCHANTS[cat])
            amount = round(random.uniform(min_amt, max_amt), 0)
            description = f"{merchant_name} Monthly"
            transactions.append({
                "Date": current.strftime("%d-%m-%Y"),
                "Description": description,
                "Merchant": merchant_name,
                "Amount": amount,
                "Category": cat
            })

    current += timedelta(days=1)

df = pd.DataFrame(transactions)
df = df.sort_values("Date").reset_index(drop=True)
df.to_csv("sample_transactions.csv", index=False)
print(f"Generated {len(df)} transactions from {df['Date'].min()} to {df['Date'].max()}")
print(f"Categories: {df['Category'].value_counts().to_dict()}")
