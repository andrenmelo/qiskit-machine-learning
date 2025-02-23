# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Torch Connector 2."""

import unittest
from test import QiskitMachineLearningTestCase
import numpy as np
from ddt import ddt

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals, optionals
import qiskit_machine_learning.optionals as _optionals
from qiskit_machine_learning.neural_networks import CircuitQNN, TwoLayerQNN
from qiskit_machine_learning.connectors import TorchConnector


@ddt
class TestTorchConnector(QiskitMachineLearningTestCase):
    """Torch Connector Tests 2."""

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    @unittest.skipIf(not _optionals.HAS_TORCH, "PyTorch not available.")
    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 12345
        # specify quantum instances
        self.sv_quantum_instance = QuantumInstance(
            qiskit.Aer.get_backend("aer_simulator_statevector"),
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
        # pylint: disable=no-member
        self.qasm_quantum_instance = QuantumInstance(
            qiskit.providers.aer.AerSimulator(),
            shots=100,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
        import torch

        torch.manual_seed(algorithm_globals.random_seed)

    def test_opflow_qnn_hybrid_batch_gradients(self):
        """Test gradient back-prop for batch input in hybrid opflow qnn."""
        from torch import cat, Tensor
        from torch.nn import Linear, Module, MSELoss
        import torch.nn.functional as F
        from torch.optim import SGD

        num_inputs = 2

        # set up QNN
        qnn = TwoLayerQNN(
            num_qubits=num_inputs,
            quantum_instance=self.sv_quantum_instance,
            input_gradients=True,  # for hybrid qnn
        )

        # set up dummy hybrid PyTorch module
        class Net(Module):
            """PyTorch nn module."""

            def __init__(self):
                super().__init__()
                self.fc1 = Linear(4, 2)
                self.qnn = TorchConnector(
                    qnn, np.array([0.1] * qnn.num_weights)
                )  # Apply torch connector
                self.fc2 = Linear(1, 1)  # 1-dimensional output from QNN

            def forward(self, x):
                """Forward pass."""
                x = F.relu(self.fc1(x))
                x = self.qnn(x)  # apply QNN
                x = self.fc2(x)
                return cat((x, 1 - x), -1)

        model = Net()

        # random data set
        x = Tensor(np.random.rand(5, 4))
        y = Tensor(np.random.rand(5, 2))

        # define optimizer and loss
        optimizer = SGD(model.parameters(), lr=0.1)
        f_loss = MSELoss(reduction="sum")

        # loss and gradients without batch
        optimizer.zero_grad(set_to_none=True)
        sum_of_individual_losses = 0.0
        for x_i, y_i in zip(x, y):
            output = model(x_i)
            sum_of_individual_losses += f_loss(output, y_i)
        sum_of_individual_losses.backward()
        sum_of_individual_gradients = 0.0
        for n, param in model.named_parameters():
            # make sure gradient is not None
            self.assertFalse(param.grad is None)
            if n.endswith(".weight"):
                sum_of_individual_gradients += np.sum(param.grad.numpy())

        # loss and gradients with batch
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        batch_loss = f_loss(output, y)
        batch_loss.backward()
        batch_gradients = 0.0
        for n, param in model.named_parameters():
            # make sure gradient is not None
            self.assertFalse(param.grad is None)
            if n.endswith(".weight"):
                batch_gradients += np.sum(param.grad.numpy())

        # making sure they are equivalent
        self.assertAlmostEqual(
            np.linalg.norm(sum_of_individual_gradients - batch_gradients), 0.0, places=4
        )

        self.assertAlmostEqual(
            sum_of_individual_losses.detach().numpy(),
            batch_loss.detach().numpy(),
            places=4,
        )

    def test_circuit_qnn_hybrid_batch_gradients(self):
        """Test gradient back-prop for batch input in hybrid circuit qnn."""
        from torch import cat, Tensor
        from torch.nn import Linear, Module, MSELoss
        import torch.nn.functional as F
        from torch.optim import SGD

        output_shape, interpret = 2, lambda x: f"{x:b}".count("1") % 2
        num_inputs = 2

        feature_map = ZZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs, entanglement="linear", reps=1)

        qc = QuantumCircuit(num_inputs)
        qc.append(feature_map, range(num_inputs))
        qc.append(ansatz, range(num_inputs))

        qnn = CircuitQNN(
            qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            input_gradients=True,  # for hybrid qnn
            interpret=interpret,
            output_shape=output_shape,
            quantum_instance=self.sv_quantum_instance,
        )

        # set up dummy hybrid PyTorch module
        class Net(Module):
            """PyTorch nn module."""

            def __init__(self):
                super().__init__()
                self.fc1 = Linear(4, 2)
                self.qnn = TorchConnector(
                    qnn, np.array([0.1] * qnn.num_weights)
                )  # Apply torch connector
                self.fc2 = Linear(2, 1)  # 2-dimensional output from QNN

            def forward(self, x):
                """Forward pass."""
                x = F.relu(self.fc1(x))
                x = self.qnn(x)  # apply QNN
                x = self.fc2(x)
                return cat((x, 1 - x), -1)

        model = Net()

        # random data set
        x = Tensor(np.random.rand(5, 4))
        y = Tensor(np.random.rand(5, output_shape))

        # define optimizer and loss
        optimizer = SGD(model.parameters(), lr=0.1)
        f_loss = MSELoss(reduction="sum")

        # loss and gradients without batch
        optimizer.zero_grad(set_to_none=True)
        sum_of_individual_losses = 0.0
        for x_i, y_i in zip(x, y):
            output = model(x_i)
            sum_of_individual_losses += f_loss(output, y_i)
        sum_of_individual_losses.backward()
        sum_of_individual_gradients = 0.0
        for n, param in model.named_parameters():
            # make sure gradient is not None
            self.assertFalse(param.grad is None)
            if n.endswith(".weight"):
                sum_of_individual_gradients += np.sum(param.grad.numpy())

        # loss and gradients with batch
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        batch_loss = f_loss(output, y)
        batch_loss.backward()
        batch_gradients = 0.0
        for n, param in model.named_parameters():
            # make sure gradient is not None
            self.assertFalse(param.grad is None)
            if n.endswith(".weight"):
                batch_gradients += np.sum(param.grad.numpy())

        # making sure they are equivalent
        self.assertAlmostEqual(
            np.linalg.norm(sum_of_individual_gradients - batch_gradients), 0.0, places=4
        )

        self.assertAlmostEqual(
            sum_of_individual_losses.detach().numpy(),
            batch_loss.detach().numpy(),
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
