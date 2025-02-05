# Import here whatever you may need
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class LinearRegressor:
    """
    Linear Regression model that can perform both simple and multiple linear regression.

    Attributes:
        coefficients (np.ndarray): Coefficients of the independent variables in the regression model.
        intercept (float): Intercept of the regression model.
    """

    def __init__(self):
        """Initializes the LinearRegressor model with default coefficient and intercept values."""
        self.coefficients = None
        self.intercept = None

    def fit_simple(self, X, y):
        """
        Fit the model using simple linear regression (one independent variable).

        This method calculates the coefficients for a linear relationship between
        a single predictor variable X and a response variable y.

        Args:
            X (np.ndarray): Independent variable data (1D array).
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if np.ndim(X) > 1:
            X = X.reshape(1, -1)

        # Train linear regression model with only one coefficient (DONE)
        n = len(X)
        w = (n * np.sum(X * y) - np.sum(X) * np.sum(y)) / (n * np.sum(X**2) - (np.sum(X))**2)
        b = (np.sum(y) - w * np.sum(X)) / n
        self.coefficients = w
        self.intercept = b

    # This part of the model you will only need for the last part of the notebook
    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array where each column is a variable).
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # Train linear regression model with multiple coefficients (DONE)
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # Convert to 2D if it's 1D
        y = y.ravel()
        X = np.c_[np.ones(X.shape[0]), X]
        XT = X.T
        beta = np.linalg.inv(XT @ X) @ XT @ y
        b = beta[0]
        w = beta[1:]
        self.intercept = b
        self.coefficients = w

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        if np.ndim(X) == 1:
            # Predict when X is only one variable (DONE)
            predictions = self.coefficients*X + self.intercept
        else:
            # Predict when X is more than one variable (DONE)
            predictions = X @ self.coefficients + self.intercept
        return predictions


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """
    # R^2 Score
    N = len(y_true)
    # Calculate R^2 (DONE)
    RSS = np.sum((y_true - y_pred)**2)
    data_mean = np.mean(y_true)  # pregunté esto en clase
    TSS = np.sum((y_true-data_mean)**2)
    r_squared = 1 - (RSS/TSS)

    # Root Mean Squared Error
    # Calculate RMSE (DONE)
    rmse = np.sqrt((1/N)*np.sum((y_true - y_pred)**2))  # alternativamente np.sqrt((1/N)*RSS)

    # Mean Absolute Error
    # Calculate MAE (DONE)
    mae = (1/N)*np.sum(np.abs(y_true-y_pred))

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


# ### Scikit-Learn comparison


def sklearn_comparison(x, y, linreg):
    ### Compare your model with sklearn linear regression model
    # Import Linear regression from sklearn (DONE)
    from sklearn.linear_model import LinearRegression

    # Assuming your data is stored in x and y
    # Reshape x to be a 2D array, as scikit-learn expects 2D inputs for the features (DONE)
    x_reshaped = x.reshape(-1,1)

    # Create and train the scikit-learn model
    # Train the LinearRegression model (DONE)
    sklearn_model = LinearRegression()
    sklearn_model.fit(x_reshaped, y)

    # Now, you can compare coefficients and intercepts between your model and scikit-learn's model
    print("Custom Model Coefficient:", linreg.coefficients)
    print("Custom Model Intercept:", linreg.intercept)
    print("Scikit-Learn Coefficient:", sklearn_model.coef_[0])
    print("Scikit-Learn Intercept:", sklearn_model.intercept_)
    return {
        "custom_coefficient": linreg.coefficients,
        "custom_intercept": linreg.intercept,
        "sklearn_coefficient": sklearn_model.coef_[0],
        "sklearn_intercept": sklearn_model.intercept_,
    }

def anscombe_quartet():
    # Load Anscombe's quartet
    # These four datasets are the same as in slide 19 of chapter 02-03: Linear and logistic regression
    anscombe = sns.load_dataset("anscombe")

    # Anscombe's quartet consists of four datasets
    # Construct an array that contains, for each entry, the identifier of each dataset (DONE)
    datasets = anscombe["dataset"].unique()

    models = {}
    results = {"R2": [], "RMSE": [], "MAE": []}
    for dataset in datasets:

        # Filter the data for the current dataset
        # (DONE)
        data = anscombe[anscombe["dataset"] == dataset]

        # Create a linear regression model
        # (DONE)
        model = LinearRegressor()

        # Fit the model
        # (DONE)
        X = data["x"]
        y = data["y"]  
        model.fit_simple(X, y)

        # Create predictions for dataset
        # (DONE)
        y_pred = model.predict(X)

        # Store the model for later use
        models[dataset] = model

        # Print coefficients for each dataset
        print(
            f"Dataset {dataset}: Coefficient: {model.coefficients}, Intercept: {model.intercept}"
        )

        evaluation_metrics = evaluate_regression(y, y_pred)

        # Print evaluation metrics for each dataset
        print(
            f"R2: {evaluation_metrics['R2']}, RMSE: {evaluation_metrics['RMSE']}, MAE: {evaluation_metrics['MAE']}"
        )
        results["R2"].append(evaluation_metrics["R2"])
        results["RMSE"].append(evaluation_metrics["RMSE"])
        results["MAE"].append(evaluation_metrics["MAE"])
    return anscombe, datasets, models,  results


# Go to the notebook to visualize the results
