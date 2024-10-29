import numpy as np
import matplotlib.pyplot as plt

# Data
Values = [0.500, 0.484, 0.474, -0.132, -0.182, -0.268, -0.414]
Name = ["GDP", "Life Expect", "Pop Growth", "Area", "Military", "Unemployment", "Inflation"]

# Determine colors based on values
colors = ['green' if value > 0 else 'red' for value in Values]

# Plotting the PC1 loadings
plt.figure(figsize=(12, 6))
bars = plt.bar(Name, Values, color=colors)  # Create bars with the specified colors
plt.title('PC1 Loadings')
plt.ylabel('Loadings')
plt.xlabel('Fields')
plt.xticks(rotation=45)
plt.grid(axis='y')

# Annotate bars with values
for bar in bars:
    plt.annotate(f'{bar.get_height():.3f}',  # Round to 3 decimal places
                 (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
