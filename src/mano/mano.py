""" 
Implementation of the estimation score provided by MaNo.

[1] R. Xie, A. Odonnat, V. Feofanov et al. MaNo: Exploiting Matrix Norm for Unsupervised
    Accuracy Estimation under Distribution Shifts. NeurIPS 2024.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class MaNo:
    def __init__(self, norm_order=4, threshold=5.0, taylor_order=2, batch_size=None, device='cpu'):
        """
        MaNo provides an unsupervised logit-based estimation of the test accuracy 
        in a training-free fashion. It consists of three simple steps:

        1) Criterion: Determine the appropriate logit normalization,
        2) Normalization: normalize the logits such that they have the same range,
        2) Aggregation: aggregate the normalized logits using an entry-wise matrix norm.

        Parameters
        ----------
        norm_order: int, default=4
            The order of the norm at the aggregation step.
        threshold: float, default=5.0
            The threshold that decides the normalization strategy. If uncertainty is larger than threshold,
            then the softmax is applied, otherwise, a Taylor approximation of the softmax is applied.
        taylor_order: int, default=2
            The Taylor approximation order.
        batch_size: int, default=None
            If batch_size is not None, then the score is evaluated in a batch fashion.
        device: {'cpu', 'cuda'}, default=None
            On which device calculations are performed
        """

        self.norm_order = norm_order
        self.threshold = threshold
        self.taylor_order = taylor_order
        self.batch_size = batch_size
        self.device = device
        self.dataloader = None
        self.n_samples = None
        self.n_classes = None
        self.criterion = None

        # Taylor order must be a positive integer
        assert self.taylor_order < 1, "Invalid value: taylor_order must be >= 1."

    def evaluate(self, x):
        """Recover MaNo estimation score. """

        self.n_samples, self.n_classes = x.shape
        batch_size = self.n_samples if self.batch_size is None else self.batch_size
        dataset = TensorDataset(x.type(torch.float))
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Compute uncertainty criterion to select the proper normalization
        if self.criterion is None:
            self.get_uncertainty_(x)
            
        # Compute MaNo estimation score
        scores = []
        for batch_idx, logits in enumerate(self.dataloader):
            logits = logits.to(self.device)
            with torch.no_grad():

                # Normalization
                normalized_logits = self.normalize_(logits)

                # Aggregation
                score = self.aggregate_(normalized_logits)
            scores.append(score)
        
        return torch.Tensor(scores).mean()

    def get_criterion_(self):
        """Compute the uncertainty criterion at the dataset level.
        The criterion is equal to the average KL divergence between
        the uniform and the softmax probabilities. A low value means that the softmax 
        probabilities are close to the uniform and hence that the model is uncertain.
        """

        divergences = []
        for batch_idx, logits in enumerate(self.dataloader):
            logits = logits.to(self.device)
            with torch.no_grad():

                # Compute uniform targets
                targets = (1 / self.n_classes) *torch.ones((logits.shape[0], self.n_classes))
                targets = targets.to(self.device)

                # Recover KL divergence
                divergence = F.cross_entropy(logits, targets)
                divergences.append(divergence)

        self.criterion = torch.Tensor(divergences).mean()
        return

    def normalize_(self, logits):
        """Normalize the logits. """

        # Model is uncertain
        if self.criterion <= self.threshold:
            outputs = self.taylor_softmax_(logits) 
        
        # Model is confident
        else:
            outputs = torch.softmax(logits, dim=1)
        return outputs

    def taylor_softmax_(self, logits):
        """Compute Taylor approximation. """

        outputs = 1 + logits
        for i in range(2, self.taylor_order + 1):
            outputs += (logits ** i) / i

        # Ensure that all entries are positive (needed when taylor_order > 2)
        min_value = torch.min(outputs, 1, keepdim=True)[0].expand_as(outputs)
        outputs = F.normalize(outputs - min_value, dim=1, p=1)
        return outputs

    def aggregate_(self, logits):
        """Compute the normalized matrix p-norm of logits. """

        # Compute the p-norm 
        score = torch.norm(logits, p=self.norm_order) 

        # Normalization to obtain a score in [0, 1]
        score /= ((self.n_samples * self.n_classes) ** (1 / self.norm_order))

        return score