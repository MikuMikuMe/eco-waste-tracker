# eco-waste-tracker

Creating a comprehensive Python program for an eco-waste-tracker involves integrating machine learning components, data processing, and more. This example will give you a basic structure for such a project. We'll simulate the data for this exercise.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EcoWasteTracker:
    def __init__(self):
        """
        Initialization of the EcoWasteTracker class.
        """
        self.model = None
        self.data = None

    def load_data(self):
        """
        This function simulates loading data.
        This is where you would normally load the data from a file or database.
        """
        # Simulated data with features: population, urbanization level, seasonal_index
        try:
            data = {
                'population': np.random.randint(1000, 10000, 100),
                'urbanization_level': np.random.rand(100) * 100,
                'seasonal_index': np.random.rand(100) * 10,
                'waste_generated': np.random.rand(100) * 1000
            }
            self.data = pd.DataFrame(data)
            logging.info("Data loading successful.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")

    def preprocess_data(self):
        """
        Handle data preprocessing, including handling missing values or scaling.
        """
        try:
            if self.data is not None:
                # Example of data normalization
                self.data['population'] = self.data['population'] / self.data['population'].max()
                self.data['urbanization_level'] = self.data['urbanization_level'] / 100
                logging.info("Data preprocessing completed.")
            else:
                raise ValueError("Data not loaded.")
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")

    def train_model(self):
        """
        Train a machine learning model to predict waste generation.
        """
        try:
            if self.data is not None:
                X = self.data.drop('waste_generated', axis=1)
                y = self.data['waste_generated']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                self.model = LinearRegression()
                self.model.fit(X_train, y_train)

                y_pred = self.model.predict(X_test)
                error = mean_squared_error(y_test, y_pred)

                logging.info(f"Model training completed. MSE: {error}")
            else:
                raise ValueError("Data not preprocessed.")
        except Exception as e:
            logging.error(f"Error during model training: {e}")

    def predict_waste(self, new_data):
        """
        Predict the waste generated using the trained model.

        :param new_data: A DataFrame with the same features used in training.
        :return: Predicted waste generation in the given new data.
        """
        try:
            if self.model is not None:
                prediction = self.model.predict(new_data)
                logging.info(f"Prediction success: {prediction}")
                return prediction
            else:
                raise ValueError("Model not trained.")
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return None

if __name__ == "__main__":
    tracker = EcoWasteTracker()
    tracker.load_data()
    tracker.preprocess_data()
    tracker.train_model()

    # Simulate new data for prediction
    new_data = pd.DataFrame({
        'population': [0.5, 0.6],
        'urbanization_level': [0.3, 0.8],
        'seasonal_index': [5, 2]
    })

    predictions = tracker.predict_waste(new_data)
    print(predictions)
```

### Explanation
- **Data Simulation**: In this example, we simulate data to demonstrate how the program works. In a real-world scenario, you would load actual data from a file or database.
- **Logging**: Proper logging is set up throughout the program to track its progress and any potential errors.
- **Machine Learning**: A basic linear regression model is used as a placeholder for a more complex model that you might have trained using historical waste generation data.
- **Predictive Functionality**: The `predict_waste` method allows the user to make predictions on new data, demonstrating the system's intelligent waste management capabilities.
- **Error Handling**: Proper try-except blocks are placed in major functions to catch and log any potential errors.

This program is a starting point and can be expanded with more sophisticated models and integration with data sources for real-time waste tracking.