import numpy as np
import matplotlib.pyplot as plt
from neuron import h
from model import Nexc, encoding_dim
from decoding import predict_logistic, fit_logistic_decoder
from sklearn.metrics import accuracy_score
from mnist_data import generate_inputs


seed = 999

train_size = 600
test_size = 10

train_image_array, train_labels = generate_inputs(
    plot=False,
    train_size=train_size,
    dataset="train",
    data_prefix="./datasets/mnist",
    seed=seed,
)
test_image_array, test_labels = generate_inputs(
    plot=False,
    test_size=test_size,
    dataset="test",
    data_prefix="./datasets/mnist",
    seed=seed,
)

cell_spikes_exc_train = {}
cell_spikes_exc_test = {}

cell_spikes_train_dict = np.load("cell_spikes_train_mnist.npz")
for i in range(0, Nexc):
    cell_spikes_exc_train[i] = cell_spikes_train_dict[f"{i}"]

cell_spikes_test_dict = np.load("cell_spikes_test_mnist.npz")
for i in range(0, Nexc):
    cell_spikes_exc_test[i] = cell_spikes_test_dict[f"{i}"]


presentation_time = 0.1


logistic_decoder = fit_logistic_decoder(
    cell_spikes_exc_train,
    n_units=Nexc,
    unit_offset=0,
    label_dur=presentation_time * 1000.0,
    labels=train_labels,
)
output_predictions_train = predict_logistic(
    cell_spikes_exc_train,
    logistic_decoder,
    label_count=len(train_labels),
    label_dur=presentation_time * 1000.0,
    n_units=Nexc,
    unit_offset=0,
)
output_train_score = accuracy_score(train_labels, output_predictions_train)

output_predictions_test = predict_logistic(
    cell_spikes_exc_test,
    logistic_decoder,
    label_count=len(test_labels),
    label_dur=presentation_time * 1000.0,
    n_units=Nexc,
    unit_offset=0,
)
output_test_score = accuracy_score(test_labels, output_predictions_test)


print(
    f"output score (train): {output_train_score}\n"
    f"output score (test): {output_test_score}\n"
)
