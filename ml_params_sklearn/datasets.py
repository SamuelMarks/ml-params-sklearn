from ml_params.datasets import load_data_from_ml_prepare
from ml_prepare.datasets import datasets2classes
from sklearn.datasets import fetch_openml


def load_data_from_openml_or_ml_prepare(dataset_name, datasets_dir=None,
                                        K=None, as_numpy=True, **data_loader_kwargs):
    """
    Acquire from the official openml zoo, or the ophthalmology focussed ml-prepare library

    :param dataset_name: name of dataset
    :type dataset_name: ```str```

    :param datasets_dir: directory to look for models in. Default is ~/tensorflow_datasets.
    :type datasets_dir: ```None or str```

    :param K: backend engine, e.g., `np` or `tf`
    :type K: ```None or np or tf or Any```

    :param as_numpy: Convert to numpy ndarrays
    :type as_numpy: ```bool```

    :param data_loader_kwargs: pass this as arguments to data_loader function
    :type data_loader_kwargs: ```**data_loader_kwargs```

    :return: Train and tests dataset splits
    :rtype: ```Tuple[np.ndarray, np.ndarray]```
    """
    if dataset_name in datasets2classes:
        return load_data_from_ml_prepare(dataset_name=dataset_name,
                                         tfds_dir=datasets_dir,
                                         as_numpy=as_numpy,
                                         **data_loader_kwargs)

    data_loader_kwargs.update({
        'dataset_name': dataset_name,
        'datasets_dir': datasets_dir,

    })
    if 'scale' not in data_loader_kwargs:
        data_loader_kwargs['scale'] = 255.

    X, y = fetch_openml(dataset_name, version=1, return_X_y=True, data_home=datasets_dir)
    X = X / data_loader_kwargs['scale']

    if dataset_name == 'mnist_784':
        # rescale the data, use the traditional train/test split
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]
    else:
        X_half_len = len(X) // 2
        y_half_len = len(y) // 2
        X_train, X_test = X[:X_half_len], X[X_half_len:]
        y_train, y_test = y[:y_half_len], y[y_half_len:]

    return (X_train, y_train), (X_test, y_test)
