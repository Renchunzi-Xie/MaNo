import torch
from algs.base_alg import Base_alg
import torch.nn as nn


class MaNo(Base_alg):
    def evaluate(self):
        self.base_model.train()
        score_list = []

        for batch_idx, batch_data in enumerate(self.val_loader):
            inputs, labels = batch_data[0], batch_data[1]
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            with torch.no_grad():
                outputs = self.base_model(inputs)
                outputs = self.scaling_method(outputs)
                score = torch.norm(outputs, p=self.args['norm_type']) / (
                            (outputs.shape[0] * outputs.shape[1]) ** (1 / self.args['norm_type']))
            # final score
            score_list.append(score)

        scores = torch.Tensor(score_list).numpy()
        return scores.mean()

    def scaling_method(self, logits):
        loss = self.args['delta']
        if loss > 5:
            outputs = nn.functional.normalize(logits, dim=1, p=1)
        else:
            outputs = logits + 1 + logits ** 2 / 2
            min_value = torch.min(outputs, 1, keepdim=True)[0].expand_as(outputs)
            outputs = nn.functional.normalize(outputs - min_value, dim=1, p=1)
        return outputs

    def uniform_cross_entropy(self):
        losses = []
        for batch_idx, batch_data in enumerate(self.val_loader):
            if batch_idx < 5:
                inputs, labels = batch_data[0], batch_data[1]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    logits = self.base_model(inputs)
                    targets = torch.ones((logits.shape[0], self.args["num_classes"])).to(self.device) * (
                                1 / self.args["num_classes"])
                    loss = nn.functional.cross_entropy(logits, targets)
                    losses.append(loss)
            else:
                break
        losses = torch.Tensor(loss)
        return torch.mean(losses)