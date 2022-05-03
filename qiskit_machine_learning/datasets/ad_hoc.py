# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
ad hoc dataset
"""

from typing import Tuple
import itertools as it
from functools import reduce
import numpy as np
import scipy
from qiskit.utils import algorithm_globals, optionals

from qiskit_machine_learning.datasets.dataset_helper import (
    features_and_labels_transform,
)


def ad_hoc_data(
    training_size,
    test_size,
    n,
    gap,
    one_hot=True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Generates a toy dataset that can be fully separated with
    ``qiskit.circuit.library.ZZ_Feature_Map`` according to the procedure
    outlined in [1]. To construct the dataset, we first sample uniformly
    distributed vectors :math:`\vec{x} \in (0, 2\pi]^{n}` and apply the
    feature map

    .. math::
        |\Phi(\vec{x})\rangle = U_{{\Phi} (\vec{x})} H^{\otimes n} U_{{\Phi} (\vec{x})}
        H^{\otimes n} |0^{\otimes n} \rangle

    where

    .. math::
        U_{{\Phi} (\vec{x})} = \exp \left( i \sum_{S \subseteq [n] } \phi_S(\vec{x})
        \prod_{i \in S} Z_i \right)

    and

    .. math::
        \begin{cases}
        \phi_{\{i, j\}} = (\pi - x_i)(\pi - x_j) \\
        \phi_{\{i\}} = x_i
        \end{cases}

    We then attribute labels to the vectors according to the rule

    .. math::
        m(\vec{x}) = \begin{cases}
        1 & \langle \Phi(\vec{x}) | V^\dagger \prod_i Z_i V | \Phi(\vec{x}) \rangle > \Delta \\
        -1 & \langle \Phi(\vec{x}) | V^\dagger \prod_i Z_i V | \Phi(\vec{x}) \rangle < -\Delta
        \end{cases}

    where :math:`\Delta` is the separation gap, and
    :math:`V\in \mathrm{SU}(4)` is a random unitary.

    The current implementation only works with n = 2 or 3.

    **References:**

    [1] Havlíček V, Córcoles AD, Temme K, Harrow AW, Kandala A, Chow JM,
    Gambetta JM. Supervised learning with quantum-enhanced feature
    spaces. Nature. 2019 Mar;567(7747):209-12.
    `arXiv:1804.11326 <https://arxiv.org/abs/1804.11326>`_

    Args:
        training_size: the number of training samples.
        test_size: the number of testing samples.
        n: number of qubits (dimension of the feature space). Must be 2 or 3.
        gap: separation gap (:math:`\Delta`).
        plot_data: whether to plot the data. Requires matplotlib.
        one_hot: if True, return the data in one-hot format.
        include_sample_total: if True, return all points in the uniform
            grid in addition to training and testing samples.

    Returns:
        Training and testing samples.

    Raises:
        ValueError: if n is not 2 or 3.
    """
    class_labels = [r"A", r"B"]

    # Define auxiliary matrices and initial state
    z = np.diag([1, -1])
    i_2 = np.eye(2)
    h_2 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    h_n = reduce(np.kron, [h_2] * n)
    psi_0 = np.ones(2**n) / np.sqrt(2**n)

    # Generate Z matrices acting on each qubits
    z_i = np.array([reduce(np.kron, [i_2] * i + [z] + [i_2] * (n - i - 1)) for i in range(n)])

    # Construct the parity operator
    bitstrings = ["".join(bstring) for bstring in it.product(*[["0", "1"]] * n)]
    bitstring_parity = [bstr.count("1") % 2 for bstr in bitstrings]
    d_m = np.diag((-1) ** np.array(bitstring_parity))

    # Construct random unitary by exponentiating a random hermitian matrix
    basis = algorithm_globals.random.random((2**n, 2**n))
    basis += basis.T.conj()
    basis = scipy.linalg.expm(1j * basis)
    m_m = basis.conj().T @ d_m @ basis

    # Generate random points in the feature space and compute the expectation value of the parity
    # Keep the points if the absolute value of the expectation value exceeds the gap provided by
    # the user
    ind_pairs = [[i, i + 1] for i in range(n - 1)]
    sample_total = []
    x_sample, y_sample = [], []
    while (
        y_sample.count(0) < training_size + test_size
        or y_sample.count(1) < training_size + test_size
    ):
        x = 2 * np.pi * algorithm_globals.random.random(n)
        phi = np.sum(x[:, None, None] * z_i, axis=0)
        phi += sum([(np.pi - x[i1]) * (np.pi - x[i2]) * z_i[i1] @ z_i[i2] for i1, i2 in ind_pairs])
        u_u = scipy.linalg.expm(1j * phi)  # pylint: disable=no-member
        psi = u_u @ h_n @ u_u @ psi_0
        exp_val = np.real(psi.conj().T @ m_m @ psi)
        if exp_val < -gap and y_sample.count(0) < training_size + test_size:
            x_sample.append(x)
            y_sample.append(0)
        if exp_val > gap and y_sample.count(1) < training_size + test_size:
            x_sample.append(x)
            y_sample.append(1)
        if np.random.uniform() > 0.99:
            print(y_sample.count(0), y_sample.count(1))
    x_sample, y_sample = np.array(x_sample), np.array(y_sample)

    training_input = {
        key: (x_sample[y_sample == k, :])[:training_size] for k, key in enumerate(class_labels)
    }
    test_input = {
        key: (x_sample[y_sample == k, :])[training_size : (training_size + test_size)]
        for k, key in enumerate(class_labels)
    }

    training_feature_array, training_label_array = features_and_labels_transform(
        training_input, class_labels, one_hot
    )
    test_feature_array, test_label_array = features_and_labels_transform(
        test_input, class_labels, one_hot
    )

    return (
        training_feature_array,
        training_label_array,
        test_feature_array,
        test_label_array,
    )
