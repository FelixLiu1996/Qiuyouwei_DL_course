from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

digits = load_digits()
print(digits.data)

scaler = StandardScaler()
scaler.fit(digits.data)
