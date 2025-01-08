"""
Implementation of the MaNo estimation score provided in [1].

[1] R. Xie, A. Odonnat, V. Feofanov et al. MaNo: Exploiting Matrix Norm for Unsupervised
    Accuracy Estimation under Distribution Shifts. NeurIPS 2024.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class MaNo:
    """Implementation of the MaNo estimation score provided in [1]_.

    MaNo provides an unsupervised logit-based estimation of the test accuracy
    in a training-free fashion. It consists in three simple steps:

    1) Criterion: Determine the appropriate logit normalization,
    2) Normalization: normalize the logits such that they have the same range,
    2) Aggregation: aggregate the normalized logits using an entry-wise matrix norm.

    Parameters
    ----------
    norm_order: int, default=4
        The order of the norm at the aggregation step.
    threshold: float, default=5.0
        The threshold that decides the normalization strategy.
        If self.criterion is larger than threshold, softmax normalization is applied.
        Else, a Taylor approximation of the softmax is applied.
    taylor_order: int, default=2
        The Taylor approximation order.
    batch_size: int, default=None
        If batch_size is not None, then the score is evaluated in a batch fashion.
    device: torch.device, torch.device("cpu")
        Determine which device calculations are performed.

    References
    ----------

    .. [1] R. Xie, A. Odonnat, V. Feofanov et al. MaNo: Exploiting Matrix Norm for Unsupervised
        Accuracy Estimation under Distribution Shifts. NeurIPS 2024.
    """

    def __init__(
        self,
        norm_order=4,
        threshold=5.0,
        taylor_order=2,
        batch_size=None,
        device=torch.device("cpu"),
    ):
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
        """Recover MaNo estimation score."""

        # Initialize dataloader
        self.n_samples, self.n_classes = x.shape
        batch_size = self.n_samples if self.batch_size is None else self.batch_size
        dataset = TensorDataset(x.type(torch.float))
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Compute uncertainty criterion to select the proper normalization
        if self.criterion is None:
            self._get_criterion()

        # Compute MaNo estimation score
        scores = []
        for _, logits in enumerate(self.dataloader):
            logits = logits.to(self.device)
            with torch.no_grad():

                # Normalization
                normalized_logits = self._softrun(logits)

                # Aggregation
                score = self._aggregate(normalized_logits)
            scores.append(score)

        return torch.Tensor(scores).mean()

    def _get_criterion(self):
        """Compute the uncertainty criterion at the dataset level.
        The criterion is equal to the average KL divergence between
        the uniform and the softmax probabilities. A low value means that the softmax
        probabilities are close to the uniform and hence that the model is uncertain.
        """

        divergences = []
        for _, logits in enumerate(self.dataloader):
            logits = logits.to(self.device)
            with torch.no_grad():

                # Compute uniform targets
                targets = (1 / self.n_classes) * torch.ones((logits.shape[0], self.n_classes))
                targets = targets.to(self.device)

                # Compute KL divergence
                divergence = F.cross_entropy(logits, targets)
                divergences.append(divergence)

        self.criterion = torch.Tensor(divergences).mean()
        return

    def _softrun(self, logits):
        """Normalize the logits following Eq.(6) of [1].
        If self.criterion is larger than threshold, softmax normalization is applied.
        Else, a Taylor approximation of the softmax is applied.

        References
        ----------

        [1] R. Xie, A. Odonnat, V. Feofanov et al. MaNo: Exploiting Matrix Norm for Unsupervised
            Accuracy Estimation under Distribution Shifts. NeurIPS 2024.
        """

        # Apply softmax normalization
        if self.criterion > self.threshold:
            outputs = torch.softmax(logits, dim=1)

        # Apply Taylor approximation
        else:
            outputs = self._taylor_softmax(logits)

        return outputs

    def _taylor_softmax(self, logits):
        """Compute Taylor approximation."""

        outputs = 1 + logits
        for i in range(2, self.taylor_order + 1):
            outputs += (logits**i) / i

        # This is done to ensure that outputs is a probability distribution
        min_value = torch.min(outputs, 1, keepdim=True)[0].expand_as(outputs)
        outputs = F.normalize(outputs - min_value, dim=1, p=1)
        return outputs

    def _aggregate(self, logits):
        """Compute the normalized matrix p-norm of logits."""

        # Compute the p-norm
        score = torch.norm(logits, p=self.norm_order)

        # Normalization to obtain a score in [0, 1]
        score /= (self.n_samples * self.n_classes) ** (1 / self.norm_order)

        return score
