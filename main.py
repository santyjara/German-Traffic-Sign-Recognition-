from model import model
import csv


n_ch = 3
height = 32
width = height
input_shape = (height,width,n_ch)
n_outputs = 43
test_size = 0.15
learning_rate = 0.001
epochs = 4
batch_size = 100

# Instance our model
model = model(input_shape,n_outputs,test_size,learning_rate,epochs,batch_size)

# Load data
model.load_data('train')

# Shuffle the data
X,y = model.permute_feature_label_data()

# Split the data into X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = model.split(X,y)

# Train the model
model.train_model(X_train,y_train,X_test,y_test)

# Load the images contained in the test folder
images,name = model.load_data_to_predict(folder_name='test_files')

# Make predictions
predictions = model.predict(images)


def save_submission(predictions,name):
    """
    Write predictions to a csv file

    :param predictions: list containing the predictions (N)
    :param name: list cointaining the image names (N)
    :return: save a csv file with name and prediction
    """

    with open('submission.csv', 'w', newline='') as csvfile:
        fieldnames = ['file_id', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, j in zip(name, predictions):
            writer.writerow({'file_id': i, 'label': j})

#save_submission(predictions,name)