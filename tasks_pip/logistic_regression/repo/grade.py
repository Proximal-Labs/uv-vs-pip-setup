import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")

import logistic_regression as lr

lr.logistic_regression()
print("OK")