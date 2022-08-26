import os
import warnings
import sys
import pytest
import logging
from pathlib import Path

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from dtaidistance import util_numpy, dtw
from pqdtw import quantizer_cc as q

logger = logging.getLogger("be.kuleuven.dtai.distance")
directory = None
numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")


def __take_subset(X, Y, nb_samples):
    if nb_samples > len(X):
        warnings.warn(f"The number of requested datapoints ({nb_samples}) is larger than the amount of data available ({len(X)}). Returning {len(X)} samples.")
        return X, Y
    indice_list = np.random.choice(len(X), nb_samples)
    return X[indice_list], Y[indice_list]


def __load_dataset(train_file, test_file, max_train_samples = None, max_test_samples = None):
    scaler = StandardScaler()
    
    # train data
    X = np.loadtxt(train_file)
    X_scaled = scaler.fit_transform(X[:,1:len(X[0])])
    Y = X[:,0]
    if max_train_samples is not None:
        X_scaled, Y = __take_subset(X_scaled, Y, max_train_samples)

    # test data
    X2 = np.loadtxt(test_file)
    X2_scaled = scaler.transform(X2[:,1:len(X2[0])])
    Y2 = X2[:,0]
    if max_test_samples is not None:
        X2_scaled, Y2 = __take_subset(X2_scaled, Y2, max_test_samples)

    return X_scaled, Y, X2_scaled, Y2 


@pytest.fixture(scope="module")
def data():
    trainSize = 500
    valSize = 1500
    testSize = 3000
    XTrain, YTrain, XTest, YTest = __load_dataset(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), 'rsrc', 'ECG5000', 'ECG5000_TRAIN.txt'),
        os.path.join(os.path.abspath(os.path.dirname(__file__)), 'rsrc', 'ECG5000', 'ECG5000_TEST.txt'),
        trainSize, testSize+valSize)
    XVal, XTest, YVal, YTest = train_test_split(XTest, YTest, train_size=valSize, test_size=testSize)
    return {'train': (XTrain, YTrain), 'val': (XVal, YVal), 'test': (XTest, YTest)}


@numpyonly
@pytest.mark.benchmark(group="pq")
def test_distance_approx(benchmark, data):
    with util_numpy.test_uses_numpy() as np:
        distParams = {
            'window': 2, 
            'psi': 0
        }
        kmeansDistParams = {
            'window': 2, 
        }
        quantizationDistParams = {
            'window': 1, 
            'psi': 0
        }
        pq = q.PQ(70,
                subspace_type=q.SubspaceType.NO_OVERLAP,
                codebook_size=100,
                dist_strategy=q.SymmetricPQDistanceStrategy(),
                dist_params=distParams,
                quantization_dist_params=quantizationDistParams,
                kmeans_dist_params=kmeansDistParams)
        pq.fit(data['train'][0], max_iter=20)

        def d():
            return pq.approx_distance_matrix(data['test'][0], parallel=True)

        benchmark(d)

@numpyonly
@pytest.mark.benchmark(group="pq")
def test_distance_approx_double_overlap(benchmark, data):
    with util_numpy.test_uses_numpy() as np:
        distParams = {
            'window': 2, 
            'psi': 0
        }
        kmeansDistParams = {
            'window': 2, 
        }
        quantizationDistParams = {
            'window': 1, 
            'psi': 0
        }
        pq = q.PQ(70,
                subspace_type=q.SubspaceType.DOUBLE_OVERLAP,
                codebook_size=100,
                dist_strategy=q.SymmetricPQDistanceStrategy(),
                dist_params=distParams,
                quantization_dist_params=quantizationDistParams,
                kmeans_dist_params=kmeansDistParams)
        pq.fit(data['train'][0], max_iter=20)

        def d():
            return pq.approx_distance_matrix(data['test'][0], parallel=True)

        benchmark(d)

@numpyonly
@pytest.mark.benchmark(group="pq")
def test_distance_approx_prealign(benchmark, data):
    with util_numpy.test_uses_numpy() as np:
        distParams = {
            'window': 2, 
            'psi': 0
        }
        kmeansDistParams = {
            'window': 2, 
        }
        quantizationDistParams = {
            'window': 1, 
            'psi': 0
        }
        pq = q.PQ(70,
                subspace_type=q.SubspaceType.PRE_ALIGN,
                codebook_size=100,
                dist_strategy=q.SymmetricPQDistanceStrategy(),
                dist_params=distParams,
                quantization_dist_params=quantizationDistParams,
                kmeans_dist_params=kmeansDistParams)
        pq.fit(data['train'][0], max_iter=20)

        def d():
            return pq.approx_distance_matrix(data['test'][0], parallel=True)

        benchmark(d)


@numpyonly
@pytest.mark.benchmark(group="pq")
def test_distance_approx_assymetric(benchmark, data):
    with util_numpy.test_uses_numpy() as np:
        distParams = {
            'window': 2, 
            'psi': 0
        }
        kmeansDistParams = {
            'window': 2, 
        }
        quantizationDistParams = {
            'window': 1, 
            'psi': 0
        }
        pq = q.PQ(70,
                subspace_type=q.SubspaceType.DOUBLE_OVERLAP,
                codebook_size=100,
                dist_strategy=q.AsymmetricPQDistanceStrategy(),
                dist_params=distParams,
                quantization_dist_params=quantizationDistParams,
                kmeans_dist_params=kmeansDistParams)
        pq.fit(data['train'][0], max_iter=20)

        def d():
            return pq.approx_distance_matrix(data['test'][0], parallel=True)

        benchmark(d)


@numpyonly
@pytest.mark.benchmark(group="pq")
def test_distance_approx_replace_0_with_exact(benchmark, data):
    with util_numpy.test_uses_numpy() as np:
        distParams = {
            'window': 2, 
            'psi': 0
        }
        kmeansDistParams = {
            'window': 2, 
        }
        quantizationDistParams = {
            'window': 1, 
            'psi': 0
        }
        pq = q.PQ(70,
                subspace_type=q.SubspaceType.DOUBLE_OVERLAP,
                codebook_size=100,
                dist_strategy=q.ExactWhen0PQDistanceStrategy(),
                dist_params=distParams,
                quantization_dist_params=quantizationDistParams,
                kmeans_dist_params=kmeansDistParams)
        pq.fit(data['train'][0], max_iter=20)

        def d():
            return pq.approx_distance_matrix(data['test'][0], parallel=True)

        benchmark(d)


@numpyonly
@pytest.mark.benchmark(group="pq")
def test_distance_approx_replace_0_with_keogh(benchmark, data):
    with util_numpy.test_uses_numpy() as np:
        distParams = {
            'window': 2, 
            'psi': 0
        }
        kmeansDistParams = {
            'window': 2, 
        }
        quantizationDistParams = {
            'window': 1, 
            'psi': 0
        }
        pq = q.PQ(70,
                subspace_type=q.SubspaceType.DOUBLE_OVERLAP,
                codebook_size=100,
                dist_strategy=q.AsymmetricKeoghWhen0PQDistanceStrategy(),
                dist_params=distParams,
                quantization_dist_params=quantizationDistParams,
                kmeans_dist_params=kmeansDistParams)
        pq.fit(data['train'][0], max_iter=20)

        def d():
            return pq.approx_distance_matrix(data['test'][0], parallel=True)

        benchmark(d)


@numpyonly
@pytest.mark.benchmark(group="pq")
def test_distance_exact(benchmark, data):
    distParams = {
        'window': 2, 
        'psi': 0
    }
 
    def d():
        return dtw.distance_matrix(data['test'][0], use_c=True, parallel=True, **distParams)

    benchmark(d)

if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")

