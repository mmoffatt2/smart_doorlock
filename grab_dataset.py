from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt

def load_dataset():
    path = "datasets"
    #lfw_dataset = fetch_lfw_people(resize=1.1, data_home = path, min_faces_per_person=20, color=True, download_if_missing = False)
    lfw_dataset = fetch_lfw_people(data_home = path, min_faces_per_person=20, color=True, download_if_missing = False)
    n_samples, h, w, c = lfw_dataset.images.shape
    # for machine learning we use the 2 data directly (as relative pixel
    # positions info is ignored by this model)
    X = lfw_dataset.images
    n_features = X.shape[1]

    # the label to predict is the id of the person
    y = lfw_dataset.target
    target_names = lfw_dataset.target_names
    n_classes = target_names.shape[0]
    # print("X: ", X.shape)
    # print("y: ", y.shape)
    # print("target_names: ", target_names)
    # plt.imshow(X[13])
    # plt.show()
    # print(y)
    return X, n_features, y, target_names, n_classes

load_dataset()