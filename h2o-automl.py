import h2o

# Initialize H2O cluster
h2o.init()

# Load data
data = h2o.import_file('df_arabica_clean.csv')

# Split data into train, validation, and test sets
train, valid, test = data.split_frame(ratios=[0.7, 0.15], seed=123)

# Define AutoML configuration
from h2o.automl import H2OAutoML
aml = H2OAutoML(max_runtime_secs=3600, max_models=10)

# Train the models
aml.train(x=data.columns[:-1], y=data.columns[-1], training_frame=train, validation_frame=valid)

# View leaderboard
leaderboard = aml.leaderboard
print(leaderboard)

# Use the best model for predictions
best_model = aml.leader
predictions = best_model.predict(test)

# Shutdown H2O cluster
h2o.shutdown()
