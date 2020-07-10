""" Implementation of ml_params API """

# Mostly https://github.com/scikit-learn/scikit-learn/blob/269afa3/examples/neural_networks/plot_mnist_filters.py
import warnings
from os import path
from sys import stdout
from typing import Tuple

import numpy as np
from ml_params.base import BaseTrainer
from sklearn.exceptions import ConvergenceWarning

from ml_params_sklearn import get_logger
from ml_params_sklearn.datasets import load_data_from_openml_or_ml_prepare

logger = get_logger('.'.join((path.basename(path.dirname(__file__)),
                              path.basename(__file__).rpartition('.')[0])))


class SkLearnTrainer(BaseTrainer):
    """ Implementation of ml_params BaseTrainer for SkLearnTrainer """

    data = None  # type: (None or Tuple[np.ndarry, np.ndarray] )
    model = None  # contains the model, e.g., a `tl.Serial`

    def load_data(self, dataset_name, data_loader=load_data_from_openml_or_ml_prepare,
                  data_type='infer', output_type='numpy', K=None,
                  **data_loader_kwargs):
        """
        Load the data for your ML pipeline. Will be fed into `train`.

        :param dataset_name: name of dataset
        :type dataset_name: ```str```

        :param data_loader: function that returns the expected data type.
         Defaults to TensorFlow Datasets and ml_prepare combined one.
        :type data_loader: ```None or (*args, **kwargs) -> tf.data.Datasets or Any```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```**data_loader_kwargs```

        :param data_type: incoming data type, defaults to 'infer'
        :type data_type: ```str```

        :param output_type: outgoing data_type, defaults to no conversion
        :type output_type: ```None or 'numpy'```

        :param K: backend engine, e.g., `np` or `tf`
        :type K: ```None or np or tf or Any```

        :return: Dataset splits (by default, your train and test)
        :rtype: ```Tuple[np.ndarray, np.ndarray]```
        """
        self.data = super(SkLearnTrainer, self).load_data(dataset_name=dataset_name,
                                                          data_loader=data_loader,
                                                          data_type=data_type,
                                                          output_type=output_type,
                                                          K=K,
                                                          **data_loader_kwargs)

    def train(self, callbacks, epochs, loss, metrics,
              metric_emit_freq, optimizer, save_directory,
              output_type='infer', writer=stdout, alpha=1e-4,
              learning_rate=.1, *args, **kwargs):
        """
        Run the training loop for your ML pipeline.

        :param callbacks: Collection of callables that are run inside the training loop
        :type callbacks: ```None or List[Callable] or Tuple[Callable]```

        :param epochs: number of epochs (must be greater than 0)
        :type epochs: ```int```

        :param loss: Loss function, can be a string (depending on the framework) or an instance of a class
        :type loss: ```str or Callable or Any```

        :param metrics: Collection of metrics to monitor, e.g., accuracy, f1
        :type metrics: ```None or List[Callable or str] or Tuple[Callable or str]```

        :param metric_emit_freq: Frequency of metric emission, e.g., `lambda: epochs % 10 == 0`, defaults to every epoch
        :type metric_emit_freq: ```None or (*args, **kwargs) -> bool```

        :param optimizer: Optimizer, can be a string (depending on the framework) or an instance of a class
        :type callbacks: ```str or Callable or Any```

        :param save_directory: Directory to save output in, e.g., weights in h5 files. If None, don't save.
        :type save_directory: ```None or str```

        :param output_type: `if save_directory is not None` then save in this format, e.g., 'h5'.
        :type output_type: ```str```

        :param writer: Writer for all output, could be a TensorBoard instance, a file handler like stdout or stderr
        :type writer: ```stdout or Any```

        :param learning_rate: learning rate
        :type learning_rate: ```float```

        :param alpha:
        :type alpha: ```float```

        :param args:
        :param kwargs:
        :return:
        """
        super(SkLearnTrainer, self).train(callbacks=callbacks,
                                          epochs=epochs,
                                          loss=loss,
                                          metrics=metrics,
                                          metric_emit_freq=metric_emit_freq,
                                          optimizer=optimizer,
                                          save_directory=save_directory,
                                          output_type=output_type,
                                          writer=writer,
                                          *args, **kwargs)
        assert self.data is not None
        assert self.model is not None

        self.model.max_iter = epochs
        self.model.alpha = alpha
        self.model.learning_rate_init = learning_rate
        (X_train, y_train), (X_test, y_test) = self.data

        if hasattr(self.model, 'epochs'):
            self.model.epochs = epochs
        elif hasattr(self.model, 'max_depth'):
            self.model.max_depth = epochs

        if hasattr(self.model, 'learning_rate'):
            self.model.learning_rate = learning_rate
        elif hasattr(self.model, 'lr'):
            self.model.lr = learning_rate

        # this example won't converge because of CI's time constraints, so we catch the
        # warning and are ignore it here
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning,
                                    module='sklearn')
            self.model.fit(X_train, y_train)

        print('Training set score:\t%f' % self.model.score(X_train, y_train))
        print('Test set score:\t\t%f' % self.model.score(X_test, y_test))


del Tuple, get_logger

__all__ = ['SkLearnTrainer']
