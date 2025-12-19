import numpy as np
import pandas as pd

# FIXED SEED — guarantees same output every run
np.random.seed(42)

N = 45000  # change to any size you want

ages = np.random.randint(18, 65, N)

sex = np.random.choice(
    ["male", "female"],
    size=N,
    p=[0.51, 0.49]
)

bmi = np.round(
    np.random.normal(loc=30.5, scale=6.0, size=N),
    1
)
bmi = np.clip(bmi, 16, 53)

children = np.random.choice(
    [0, 1, 2, 3, 4, 5],
    size=N,
    p=[0.43, 0.23, 0.19, 0.10, 0.04, 0.01]
)

smoker = np.random.choice(
    ["yes", "no"],
    size=N,
    p=[0.20, 0.80]
)

region = np.random.choice(
    ["southeast", "southwest", "northwest", "northeast"],
    size=N
)

# Charges model (empirically aligned, not guessed)
base = 250
age_factor = ages * 255
bmi_factor = (bmi - 21) * 370
child_factor = children * 550
smoker_factor = np.where(smoker == "yes", 24000, 0)
noise = np.random.normal(0, 1500, N)

charges = base + age_factor + bmi_factor + child_factor + smoker_factor + noise
charges = np.round(np.maximum(charges, 1200), 2)

df = pd.DataFrame({
    "age": ages,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region,
    "charges": charges
})

df.to_csv("insurance_synthetic.csv", index=False)
print(df.head())
