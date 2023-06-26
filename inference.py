import h2o

# Initialize H2O cluster
h2o.init()

data = h2o.import_file('df_arabica_clean.csv')

# Split data into train, validation, and test sets
train, valid, test = data.split_frame(ratios=[0.7, 0.15], seed=123)

loaded_model = h2o.load_model('model/DRF_1_AutoML_1_20230626_01238.zip')
predictions = loaded_model.predict(test)

test_with_predictions = test.cbind(predictions)
h2o.export_file(test_with_predictions, 'inference_file.csv')

# Shutdown H2O cluster
h2o.shutdown()