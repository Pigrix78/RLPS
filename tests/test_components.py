import unittest

import numpy as np
import torch
import torch.nn.functional as F

from rlps.data import stratified_split
from rlps.psgcn import PSGCN


class ComponentTests(unittest.TestCase):
    def test_stratified_split_keeps_every_class(self):
        labels = np.repeat(np.arange(3), 20)
        train, test = stratified_split(labels, 0.1, 3, seed=7)
        self.assertEqual(set(labels[train]), {0, 1, 2})
        self.assertEqual(set(labels[test]), {0, 1, 2})
        self.assertEqual(len(set(train).intersection(test)), 0)

    def test_psgcn_forward_and_joint_loss(self):
        torch.manual_seed(3)
        model = PSGCN(input_channels=22, classes=9)
        points = torch.randn(4, 49, 22)
        adjacency = torch.eye(9)
        target = torch.tensor([0, 1, 2, 3])
        probabilities, graph_distribution = model(points, adjacency)
        self.assertEqual(tuple(probabilities.shape), (4, 9))
        self.assertEqual(tuple(graph_distribution.shape), (9, 9))
        self.assertTrue(torch.allclose(probabilities.sum(1), torch.ones(4), atol=1.0e-5))
        hard_target = F.one_hot(target, 9).float()
        loss = F.mse_loss(probabilities, hard_target) + F.mse_loss(probabilities, graph_distribution[target])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
