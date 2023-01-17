import matplotlib.pyplot as plt
import numpy as np
import json
import scipy

LEGEND = True
CURVE = lambda x,a,c,d: a*np.log(x+0.01)+c*x+d
# CURVE = None

color = None if LEGEND else "k"

with open("data/lambda_search_fisher_data.json", "r") as f:
    data = json.load(f)

plt.figure(figsize=(12,8))
if CURVE is not None:
    all_data = np.concatenate([np.array(r) for r in data], axis=0)
    all_data = all_data[all_data[:, 0].argsort()]
    X = all_data.T[0]
    y = all_data.T[1]
    optimal_params, _ = scipy.optimize.curve_fit(CURVE, X, y)
    print(optimal_params)
    X = np.linspace(np.min(X), np.max(X), 1000)
    y = CURVE(X, *optimal_params)
    plt.plot(X,y, "r--", label="Optimal Curve")

for i, repetition in enumerate(data):
    repetition = np.array(repetition)
    repetition = repetition[repetition[:, 0].argsort()]
    plt.scatter(repetition.T[0], repetition.T[1], color=color, label=f"Trial {i+1}")



if LEGEND:
    plt.legend(fancybox=True, shadow=True, bbox_to_anchor=(1.04, 1))
plt.title("Sequential Learning Measure over EWC Lambda Hyperparameter\nUsing Fisher Information Matrix")
plt.xlabel("Lambda")
plt.ylabel("Measure")
plt.tight_layout()
plt.show()