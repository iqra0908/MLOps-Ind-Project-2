# MLOps-Ind-Project-2| Coffee Quality Data - Splitting and AutoML with H2O

This README provides instructions on how to split the Coffee Quality Data into training, validation, and testing sets using H2O and perform AutoML for predictive modeling.

## Dataset

The Coffee Quality Data is used for training and evaluating machine learning models. The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/fatihb/coffee-quality-data-cqi?resource=download).

### Prerequisites

Before running the code, ensure that you have the following:

- Python installed (version 3.6 or higher)
- H2O Python package installed (can be installed via `pip install h2o`)
- Azure account with access to Azure ML service

## H20 

### Steps

1. Download the dataset: Download the Coffee Quality Data from the Kaggle dataset page and save the file (`coffee_data.csv`) to your local machine.

2. Import the dataset into H2O: Use the H2O library to import the Coffee Quality Data file (`coffee_data.csv`) into H2O.

3. Split the data: Split the imported dataset into training, validation, and testing sets using the appropriate function from the H2O library.

4. Define AutoML configuration: Set up the AutoML configuration for H2O, specifying the desired runtime, maximum number of models, and other relevant parameters.

5. Train the models: Train the machine learning models using H2O's AutoML functionality. This process will automatically explore different models and hyperparameter combinations.

6. Evaluate the models: Analyze the leaderboard generated by H2O's AutoML to assess the performance of the trained models on the validation set.

7. Make predictions: Select the best-performing model and apply it to the test set to generate predictions.

8. Shutdown H2O: Once the analysis and predictions are complete, shut down the H2O cluster.

Make sure to replace `'coffee_data.csv'` with the appropriate path to the downloaded dataset file on your local machine.
## Azure ML AutoML

### Steps

1. Download the dataset: Download the Coffee Quality Data from the Kaggle dataset page and save the file (`coffee_data.csv`) to your local machine.

2. Split the data: Split the imported dataset into training, validation, and testing sets.

3. Configure AutoML settings: Set up the AutoML configuration for Azure ML, including the task type, primary metric, explainability, allowed and blocked models, training time, validation type, and maximum concurrent iterations.

4. Train the models: Use Azure ML's AutoML functionality to train machine learning models. This process will automatically explore different models and hyperparameter combinations.

5. Evaluate the models: Analyze the results and generated leaderboard to assess the performance of the trained models on the validation set. The primary metric used in this case is the Normalized Root Mean Squared Error (NRMSE).

6. Make predictions: Select the best-performing model from the leaderboard. In this case, the best model found was the Voting Ensemble with a Normalized Root Mean Squared Error (NRMSE) of 0.04804. Apply the selected model to the test set to generate predictions.

7. Deploy the model as an endpoint: Use Azure ML to deploy the trained model as an endpoint. This will allow you to make predictions on new data.

8. Test the endpoint: Verify that the deployed endpoint is working correctly by sending sample data and receiving predictions.

9. Cleanup: If needed, delete the deployed endpoint and any associated resources to avoid incurring unnecessary costs.



## Conclusion

Following the steps outlined in this README will allow you to split the Coffee Quality Data into training, validation, and testing sets using H2O and perform AutoML for predictive modeling. You can further modify the steps or incorporate additional processing or analysis steps as needed for your project.
