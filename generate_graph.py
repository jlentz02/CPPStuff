import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV
df = pd.read_csv("experiments-O3.csv", header=None)

# Extract x-values from the first row (skip first cell)
x = df.iloc[0, 1:].astype(float)

# Plot each algorithm
for i in range(1, len(df)):
    algorithm = df.iloc[i, 0]
    y = df.iloc[i, 1:].astype(float)
    y = (2*x**3/1e9)/(y/1e6)

    plt.plot(x, y, marker='o', label=algorithm)

# Labels (fill these in)
plt.xlabel("X Axis Label")
plt.ylabel("Y Axis Label")
plt.title("Benchmark Results")


plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

