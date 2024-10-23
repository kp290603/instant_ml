import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

X = df.drop("{target}", axis=1, inplace = False)
y = df["{target}"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state = 101)

model = KNeighborsRegressor(n_neighbors = {n_neighbors}, weights = "{weights}")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)