import numpy as np
from itertools import product
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from scipy.spatial import cKDTree


class SpikeTransform:
    def __init__(self, **kwargs):
        pass

    def __call__(self, spike_times, start_time=0.0):
        return spike_times - start_time


class FirstSpikeTransform(SpikeTransform):
    def __init__(self, **kwargs):
        pass

    def __call__(self, spike_times, start_time=0.0):
        return spike_times[0] - start_time


class LastSpikeTransform(SpikeTransform):
    def __init__(self, **kwargs):
        pass

    def __call__(self, spike_times, start_time=0.0):
        return spike_times[-1] - start_time


class FirstLastSpikeTransform(SpikeTransform):
    def __init__(self, **kwargs):
        pass

    def __call__(self, spike_times, start_time=0.0):
        return spike_times[-1] - spike_times[0]


class MeanISISpikeTransform(SpikeTransform):
    def __init__(self, **kwargs):
        pass

    def __call__(self, spike_times, start_time=0.0):
        if len(spike_times) > 1:
            sp_intervals = np.diff(spike_times)
        else:
            sp_intervals = [spike_times[0] - start_time]
        return np.mean(sp_intervals)


def bin_spikes(spike_times_dict, input_count, input_dur):
    activity_bin_offsets = np.arange(0, input_count * input_dur, input_dur)
    activity_bins = activity_bin_offsets[1:]
    spike_bin_inds_dict = {}
    for gid, spike_times in spike_times_dict.items():
        inds = np.digitize(spike_times, activity_bins).flatten()
        spike_bin_inds_dict[gid] = inds

    return activity_bin_offsets, spike_bin_inds_dict


def make_activity_matrix(
    spike_times_dict, input_count, input_dur, n_units, unit_offset, spike_transform
):
    activity_bin_offsets, spike_bin_inds_dict = bin_spikes(
        spike_times_dict, input_count, input_dur
    )
    X = np.zeros((input_count, n_units))

    for i in range(input_count):
        for gid in sorted(spike_times_dict):
            unit_no = gid - unit_offset
            spike_times = spike_times_dict[gid].flatten()
            spike_bin_inds = spike_bin_inds_dict[gid]
            inds_i = np.argwhere(spike_bin_inds == i).flatten()
            if len(inds_i) > 0:
                X[i, unit_no] = spike_transform(spike_times[inds_i])

    return X


def fit_logistic_decoder(
    spike_times_dict,
    labels,
    label_dur,
    n_units,
    spike_transform=FirstLastSpikeTransform(),
    unit_offset=0,
    max_n_pca_components=30,
):
    X = make_activity_matrix(
        spike_times_dict,
        len(labels),
        label_dur,
        n_units,
        unit_offset,
        spike_transform=spike_transform,
    )

    param_grid = {
        "pca__n_components": range(10, max_n_pca_components, 10),
        "logisticregression__C": np.logspace(-4, 4, 4),
    }

    pca = PCA()
    scaler = StandardScaler()
    reg_model = LogisticRegression(tol=0.01, penalty="l1", solver="saga")
    ppl = make_pipeline(scaler, pca, reg_model)
    clf = GridSearchCV(ppl, param_grid, n_jobs=1)

    clf.fit(X, labels)

    return clf


def predict_logistic(
    spike_times_dict,
    decoder,
    label_count,
    label_dur,
    n_units,
    unit_offset=0,
    spike_transform=FirstLastSpikeTransform(),
):
    X = make_activity_matrix(
        spike_times_dict,
        label_count,
        label_dur,
        n_units,
        unit_offset,
        spike_transform=spike_transform,
    )

    return decoder.predict(X)


def fit_rate_decoder(activity, labels, n_labels, rate_decoder, ncap=20):
    n_steps, n_units = activity[0].shape
    for i, this_example_activity in enumerate(activity):
        rate_sum = np.sum(this_example_activity, axis=0)
        sequence = tuple(sorted(np.argsort(rate_sum)[::-1][:ncap]))
        if sequence not in rate_decoder:
            rate_decoder[sequence] = np.zeros(n_labels)

        rate_decoder[sequence][int(labels[i])] += 1

    kdt_matrix = np.vstack(tuple(rate_decoder.keys()))
    print(kdt_matrix.shape)
    kdt = cKDTree(kdt_matrix)

    return rate_decoder, kdt, kdt_matrix


def predict_rate(activity, rate_decoder, kdt, kdt_matrix, n_labels, ncap=20):
    n_examples = len(activity)
    n_steps, n_units = activity[0].shape

    predictions = []
    for i, this_example_activity in enumerate(activity):
        score = np.zeros(n_labels)
        rate_sum = np.sum(this_example_activity, axis=0)
        sequence = tuple(sorted(np.argsort(rate_sum)[::-1][:ncap]))

        nn = kdt.query(sequence, k=1)[1]
        key = tuple(kdt_matrix[nn])
        score += rate_decoder[key]

        predictions.append(np.argmax(score))

    return predictions


def predict_ngram(activity, ngram_decoder, n_labels, n):
    """
    Predicts between ``n_labels`` using ``ngram_decoder``.
    :param activity: Spike activity of shape ``(n_examples, time, n_neurons)``.
    :param ngram_decoder: Previously recorded ngram score model.
    :param n_labels: The number of target labels in the data or
    :param n: The max size of n-gram to use.
    :return: Predictions per example.
    """
    n_examples = len(activity)
    n_steps, n_units = activity[0].shape

    predictions = []
    for i, this_example_activity in enumerate(activity):
        score = np.zeros(n_labels)

        # Aggregate all of the firing neurons' indices
        this_example_orders_per_step = []
        for step in range(n_steps):
            step_nz = np.nonzero(this_example_activity[step])[0]
            if len(step_nz) > 0:
                step_order = step_nz[np.argsort(-this_example_activity[step][step_nz])]
                this_example_orders_per_step.append(step_order)

        for order in zip(*(this_example_orders_per_step[k:] for k in range(n))):
            for sequence in product(*order):
                if sequence in ngram_decoder:
                    score += ngram_decoder[sequence]

        predictions.append(np.argmax(score))

    return predictions


def fit_ngram_decoder(activity, labels, n_labels, n, ngram_decoder, dropout=None):
    """
    Fits ngram scores model by adding the count of each firing sequence of length n from the past ``n_examples``.

    :param activity: Firing activity of shape ``(n_examples, time, n_neurons)``.
    :param labels: The ground truth labels of shape ``(n_examples)``.
    :param n_labels: The number of target labels in the data.
    :param n: The max size of n-gram to use.
    :param ngram_decoder: Previously recorded scores to update.
    :return: Dictionary mapping n-grams to vectors of per-class unit activity.
    """

    n_steps, n_units = activity[0].shape
    act_units = {}

    for i, this_example_activity in enumerate(activity):
        inv_labels = np.ones(n_labels)
        inv_labels[int(labels[i])] = 0.0
        inv_labels_inds = np.nonzero(inv_labels)[0]

        this_example_orders_per_step = []
        for step in range(n_steps):
            step_nz = np.nonzero(this_example_activity[step])[0]
            if len(step_nz) > 0:
                step_order = step_nz[np.argsort(-this_example_activity[step][step_nz])]
                n_order = len(step_order)
                for u in step_order:
                    n_act = act_units.get(u, 0)
                    act_units[u] = n_act + 1
                if (n_order > 1) and (dropout is not None) and (dropout > 0.0):
                    n_choice = int(round(n_order * dropout))
                    n_acts = np.asarray(
                        [act_units.get(u, 0) for u in step_order], dtype=np.float32
                    )
                    sum_acts = np.sum(n_acts)
                    prob_acts = None
                    if sum_acts > 0.0:
                        prob_acts = n_acts / np.sum(n_acts)
                    dropout_selection = np.random.choice(
                        range(n_order), size=n_choice, p=prob_acts, replace=False
                    )
                    step_order = np.delete(step_order, dropout_selection)
                this_example_orders_per_step.append(step_order)

        for order in zip(*(this_example_orders_per_step[k:] for k in range(n))):
            for sequence in product(*order):
                if sequence not in ngram_decoder:
                    ngram_decoder[sequence] = np.zeros(n_labels)

                ngram_decoder[sequence][int(labels[i])] += 1
                ngram_decoder[sequence][inv_labels_inds] -= 2.0

    return ngram_decoder
