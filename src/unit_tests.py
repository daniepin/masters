import unittest
import torch

# Import the functions to be tested
from your_module import (
    initialize_data_dict,
    initialize_number_dict,
    initialize_eye_matrix,
    compute_energy_regularization,
)


class TestYourFunctions(unittest.TestCase):
    def test_initialize_data_dict(self):
        num_classes = 10
        sample_number = 5
        output_channels = 3
        device = torch.device("cpu")

        data_dict = initialize_data_dict(
            num_classes, sample_number, output_channels, device
        )

        self.assertEqual(
            data_dict.size(), (num_classes, sample_number, output_channels)
        )
        self.assertEqual(data_dict.device, device)

    def test_initialize_number_dict(self):
        num_classes = 5
        number_dict = initialize_number_dict(num_classes)

        self.assertEqual(len(number_dict), num_classes)
        for i in range(num_classes):
            self.assertEqual(number_dict[i], 0)

    def test_initialize_eye_matrix(self):
        num_classes = 4
        device = torch.device("cpu")

        eye_matrix = initialize_eye_matrix(num_classes, device)

        self.assertEqual(eye_matrix.size(), (num_classes, num_classes))
        self.assertEqual(eye_matrix.device, device)

    def test_compute_energy_regularization(self):
        # Mock the necessary inputs
        data_dict = torch.zeros(5, 3, 10)
        number_dict = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
        state = {"num_classes": 5, "sample_number": 3, "weigth_energy": 0.1}
        output = torch.randn(3, 10)
        target = torch.LongTensor([0, 1, 2])
        epoch = 5

        lr_reg_loss = compute_energy_regularization(
            data_dict, number_dict, state, output, target, epoch
        )

        # Perform assertions based on expected outcomes
        self.assertIsInstance(lr_reg_loss, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
