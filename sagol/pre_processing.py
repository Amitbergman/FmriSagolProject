import numpy as np
from sklearn.preprocessing import OneHotEncoder


def get_one_hot_from_index(index, size):
    res = np.zeros(size)
    res[index] = 1.0
    return res


def one_hot_encode_contrasts(X: np.ndarray) -> np.ndarray:
    # Reshaping to get a matrix like numpy array is necessary for the one hot encoding ahead
    X = X.reshape(len(X), len(X[0]))
    # Assuming the contrast feature is the last feature of X.
    contrast_data = X[:, -1:]

    one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
    one_hot_encoded = one_hot_encoder.fit_transform(contrast_data)

    # Deleting old contrast int-column and replace it by one-hot-encoded column
    X = np.delete(X, -1, axis=1)
    X = np.concatenate((X, one_hot_encoded), axis=1)

    return X


def group_by_contrast(X, y, first_index_of_contrast: int):
    if len(X) != len(y):
        raise RuntimeError("X and y must be of the same length!")
    X = np.array(X)
    y = np.array(y)
    T = [np.where(x[first_index_of_contrast:] == 1)[0][0] for x in X]  # Extracting contrasts as numbers out of X data
    contrasts = sorted(set(T))  # Getting the unique contrasts
    num_of_contrasts = len(contrasts)

    # Needed for the case that the contrasts' numbers are sparsed in the range they are in
    contrast_to_index = {contrasts[i]: i for i in range(num_of_contrasts)}

    # Bucket sorting
    counters = [0] * num_of_contrasts
    for t in T:
        counters[contrast_to_index[t]] += 1
    X_grouped, y_grouped = np.ndarray(X.shape), np.ndarray(y.shape)
    insertion_indexes = [0]
    for i in range(1, num_of_contrasts):
        insertion_indexes.append(insertion_indexes[i - 1] + counters[i - 1])
    for i in range(len(T)):
        insertion_index = insertion_indexes[contrast_to_index[T[i]]]
        X_grouped[insertion_index] = X[i]
        y_grouped[insertion_index] = y[i]
        insertion_indexes[contrast_to_index[T[i]]] += 1

    # Returning the number of instances in each group too
    return X_grouped, y_grouped, counters


def centeralize_1d_data(X, y, axis: str, first_index_of_contrast=None):
    if len(X) != len(y):
        raise RuntimeError("X and y must be of the same length!")
    if axis == 'contrast':
        if first_index_of_contrast is None:
            raise RuntimeError(
                "You must pass as parameter the first index of contrast if you want to normalize by this axis!")
        X_centeralized, y_centeralized, counters = group_by_contrast(X, y, first_index_of_contrast)
        group = 0
        first_index_of_group = 0
        mean_vector = np.zeros(len(X_centeralized[0][:first_index_of_contrast]))

        # This loop calculates the mean vector of each group and reduces it from each vector in the group
        for i in range(len(X_centeralized) + 1):
            if i == first_index_of_group + counters[group]:
                mean_vector /= counters[group]
                mean_vector = np.concatenate(
                    (mean_vector, np.zeros(len(X_centeralized[0][first_index_of_contrast:]), dtype=float)))
                for j in range(first_index_of_group, i):
                    X_centeralized[j] -= mean_vector
                group += 1
                first_index_of_group = i
                mean_vector = np.zeros(len(X_centeralized[0][:first_index_of_contrast]))
            if i == len(X_centeralized):
                break
            mean_vector += X_centeralized[i][:first_index_of_contrast]
    else:
        raise RuntimeError('No such axis: ' + axis + '!')

    # Notcie that the new X and y might not be shuffled
    return X_centeralized, y_centeralized
