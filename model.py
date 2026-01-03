from sklearn.linear_model import LinearRegression
import numpy as np

def train_risk_model():
    # Sample training data
    age = np.array([18,25,35,45,55,65]).reshape(-1,1)
    risk = np.array([1.6,1.3,1.1,1.2,1.4,1.6])

    model = LinearRegression()
    model.fit(age, risk)
    return model
