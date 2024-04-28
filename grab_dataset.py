from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt

def load_dataset():
    path = "datasets"
    # Fetch dataset
    # We tried resizing the images to be bigger but we had insufficient computer memory to run on larger images
    # lfw_dataset = fetch_lfw_people(resize=2.0, data_home = path, min_faces_per_person=20, color=True, download_if_missing = False)
    lfw_dataset = fetch_lfw_people(data_home = path, min_faces_per_person=20, color=True, download_if_missing = False)
    
    n_samples, h, w, c = lfw_dataset.images.shape

    # Grab images
    X = lfw_dataset.images
    n_features = X.shape[1]

    # Grab labels
    y = lfw_dataset.target
    target_names = lfw_dataset.target_names
    n_classes = target_names.shape[0]

    return X, n_features, y, target_names, n_classes

load_dataset()