import pandas as pd
import numpy as np
import gdown
import zipfile
import os
from tensorflow import keras
from scipy.stats import mode
from PIL import Image

# !mkdir 'Data'
# !mkdir 'Model'

# Define necessary classes and functions

class DataGenerator(keras.utils.Sequence):
    """Custom data generator class to handle image loading and preprocessing."""
    def __init__(self, dataframe, batch_size, target_size=None, shuffle=True):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataframe))

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch = self.dataframe.iloc[indices]
        X = np.zeros((len(batch), *self.target_size, 3))
        y = np.zeros((len(batch), 22))

        for i, (image_path, label) in enumerate(zip(batch['image_path'], batch['label'])):
            image = Image.open(image_path)
            if self.target_size:
                image = image.resize(self.target_size)
            X[i] = np.array(image) / 255.0  # Normalizing the data
            y[i, label] = 1

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def evaluate_models(models, validation_generator):
    """Evaluates a list of models on a given data generator."""
    models_true_labels = []
    models_pred_labels = []

    for model in models:
        true_labels = []
        pred_labels = []

        for i in range(len(validation_generator)):
            data, labels = validation_generator[i]
            predictions = model.predict(data)
            pred_labels.extend(np.argmax(predictions, axis=1))
            true_labels.extend(np.argmax(labels, axis=1))

        models_true_labels.append(true_labels)
        models_pred_labels.append(pred_labels)

    return models_true_labels, models_pred_labels

def load_model(path, model_name):
    """Loads a model and its history from the specified path."""
    path_model = os.path.join(path, model_name + '.json')
    path_weights = os.path.join(path, model_name + '.h5')
    path_history = os.path.join(path, 'history_' + model_name + '.npy')

    with open(path_model, 'r') as file:
        loaded_model_json = file.read()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(path_weights)

    history = np.load(path_history, allow_pickle=True).item()
    return model, history

def main():
    # Parameters
    test_url = 'https://drive.google.com/uc?id=1mtigz-kMPtI_IJKUmsG_wqNIhuBFUGdu'
    dest = '.'
    saving_path = os.path.join(dest, 'Model')
    result_path = os.path.join(dest, 'q1_submission.csv')
    target_size = (224, 224)

    # Download and unzip test data
    output_path = os.path.join(dest, 'Data', 'test.zip')
    gdown.download(test_url, output_path, quiet=False)
    zip_file_saving_path = os.path.join(dest, 'Data', 'test')

    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(zip_file_saving_path)

    # Prepare test data
    data = []
    for folders, _, files in os.walk(zip_file_saving_path):
        for file in files:
            file_path = os.path.join(folders, file)
            data.append({'image_path': file_path, 'label': 0})  # Label is dummy
    df_test = pd.DataFrame(data)
    test_generator = DataGenerator(dataframe=df_test, batch_size=512, target_size=target_size)

    # Load models
    DenseNet201_imagenet, _ = load_model(saving_path, 'DenseNet201_imagenet')
    Xception_augmented_imagenet, _ = load_model(saving_path, 'Xception_augmented_imagenet')
    DenseNet201_augmented_imagenet, _ = load_model(saving_path, 'DenseNet201_augmented_imagenet')
    selected_models = [Xception_augmented_imagenet, DenseNet201_augmented_imagenet, DenseNet201_imagenet]

    # Evaluate models and prepare submission
    true_labels_test, pred_labels_test = evaluate_models(selected_models, test_generator)
    majority_votes = mode(pred_labels_test, axis=0).mode
    categories = ['baked_potato', 'baklava', 'caesar_salad', 'cheesecake', 'cheese_sandwich', 
                    'chicken', 'chicken_curry', 'chocolate_cake', 'donuts', 'eggs', 'falafel', 'fish', 
                    'french_fries', 'hamburger', 'hot_dog', 'ice_cream', 'lasagna', 'omelette', 'pizza', 
                    'spaghetti', 'steak', 'sushi']

    y_predicted = [sorted(categories)[label] for label in np.array(majority_votes.flatten())]
    final = pd.concat([df_test['image_path'].str.extract(r'/([^/]+)$')[0], pd.Series(y_predicted)], axis=1)

    final.columns = ['name', 'predicted']
    final.to_csv(result_path, index=False)

if __name__ == "__main__":
    main()
