from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_dataset():

    iris = load_iris()

    X = iris['data']
    y = iris['target']
    names = iris['target_names']
    feature_names = iris['feature_names']
    
    print(f"Shape of X (data): {X.shape}")
    print(f"Shape of y (target): {y.shape} {y.dtype}")
    print(f"Example of x and y pair: {X[0]} {y[0]}")

    # Scale data to have mean 0 and variance 1 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data set into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=2)

    print("Shape of training set X", X_train.shape)
    print("Shape of test set X", X_test.shape)

    return X_train, X_test, y_train, y_test