import torch
from algs.base_alg import Base_alg
import torch.nn as nn


# class MaNo(Base_alg):
#     def evaluate(self):
#         self.base_model.train()
#         score_list = []

#         for batch_idx, batch_data in enumerate(self.val_loader):
#             inputs, labels = batch_data[0], batch_data[1]
#             inputs, labels = inputs.to(self.device), labels.to(self.device)

#             with torch.no_grad():
#                 outputs = self.base_model(inputs)
#                 outputs = self.scaling_method(outputs)
#                 score = torch.norm(outputs, p=self.args["norm_type"]) / (
#                     (outputs.shape[0] * outputs.shape[1])
#                     ** (1 / self.args["norm_type"])
#                 )
#             # final score
#             score_list.append(score)

#         scores = torch.Tensor(score_list).numpy()
#         return scores.mean()

#     def scaling_method(self, logits):
#         loss = self.args["delta"]
#         if loss > 5:
#             outputs = torch.softmax(logits, dim=1)
#         else:
#             outputs = logits + 1 + logits**2 / 2
#             min_value = torch.min(outputs, 1, keepdim=True)[0].expand_as(outputs)

#             # Remove min values to ensure all entries are positive. This is especially
#             # needed when the approximation order is higher than 2.
#             outputs = nn.functional.normalize(outputs - min_value, dim=1, p=1)
#         return outputs

#     def uniform_cross_entropy(self):
#         losses = []
#         for batch_idx, batch_data in enumerate(self.val_loader):
#             if batch_idx < 5:
#                 inputs, labels = batch_data[0], batch_data[1]
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 with torch.no_grad():
#                     logits = self.base_model(inputs)
#                     targets = torch.ones(
#                         (logits.shape[0], self.args["num_classes"])
#                     ).to(self.device) * (1 / self.args["num_classes"])
#                     loss = nn.functional.cross_entropy(logits, targets)
#                     losses.append(loss)
#             else:
#                 break
#         losses = torch.Tensor(loss)
#         return torch.mean(losses)




# NEW MANO
class MaNo:
    def __init__(self, norm_order=4, threshold=5.0, taylor_order=2, batch_size=None, device='cpu'):
        """
        Logit-based accuracy estimation method that consists of two steps:
        1) Logit normalization: choose appopriate normalization based on an uncertainty criterion
        2) Aggregation: aggregate the normalized logits using an entry-wise matrix norm

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
        self.uncertainty = None
        if self.taylor_order < 1:
            raise KeyError('taylor_order must be >= 1.')

    def evaluate(self, logits):
        batch_size = logits.shape[0] if self.batch_size is None else self.batch_size

        # Compute criterion $\phi$ to select the proper normalization
        if self.uncertainty is None:
            self.calculate_uncertainty(logits)
        
        n_samples, n_classes = logits.shape

        dataset = TensorDataset(logits.type(torch.float))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Compute the estimation score
        scores = []
        for batch_idx, logits_batch in enumerate(dataloader):
            logits_batch = logits_batch.to(self.device)
            with torch.no_grad():
                # normalization
                normalized_logits_batch = self._normalize(logits_batch)
                # aggregation
                score_batch = torch.norm(normalized_logits_batch, p=self.norm_order) / ((n_samples * n_classes) ** (1 / self.norm_order))
            scores.append(score_batch)
        
        return torch.Tensor(scores).mean()

    def _taylor_softmax(self, logits):
        outputs = 1 + logits
        for i in range(2, self.taylor_degree + 1):
            outputs += (logits ** i) / i
        # Remove min values to ensure all entries are positive. This is especially
        # needed when the approximation order is higher than 2.
        min_value = torch.min(outputs, 1, keepdim=True)[0].expand_as(outputs)
        outputs = nn.functional.normalize(outputs - min_value, dim=1, p=1)
        return outputs

    def _normalize(self, logits):
        if self.uncertainty > self.threshold:
            outputs = torch.softmax(logits, dim=1)
        else:
            outputs = self._taylor_softmax(logits)
        return outputs

    def calculate_uncertainty(self, logits):
        batch_size = logits.shape[0] if self.batch_size is None else self.batch_size
        n_samples, n_classes = logits.shape

        dataset = TensorDataset(logits.type(torch.float))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        uncertainties = []
        for batch_idx, logits_batch in enumerate(dataloader):
            # if batch_idx < 5:
            logits_batch = logits_batch.to(self.device)
            with torch.no_grad():
                targets_batch = torch.ones(
                    (logits_batch.shape[0], n_classes)
                ).to(self.device) * (1 / n_classes)
                uncertainty_batch = nn.functional.cross_entropy(logits_batch, targets)
                uncertainties.append(uncertainty_batch)
            # else:
                # break
        self.uncertainty = torch.Tensor(uncertainties).mean()
        return
