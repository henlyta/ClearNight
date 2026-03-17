import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import os
import matplotlib.pyplot as plt

class Unit(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, padding=kernel_size//2)

        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, inputs):
        inputs = inputs.to(self.conv.weight.device)
        return self.conv(inputs)


class TopKRouter(nn.Module):
    def __init__(self, inchannel, num_units, top_k, num_labels, hidden_size=128):
        super().__init__()
        self.num_units = num_units
        self.top_k = top_k
        self.num_labels = num_labels
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp_shared = nn.Sequential(
            nn.Linear(inchannel, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 2)
        )
        self.unit_head = nn.Linear(hidden_size // 2, self.num_units)
        self.classification_head = nn.Linear(hidden_size // 2, self.num_labels)
        self.l2_lambda = 0.01

        # init
        for layer in self.mlp_shared:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        nn.init.kaiming_normal_(self.unit_head.weight)
        nn.init.zeros_(self.unit_head.bias)
        nn.init.kaiming_normal_(self.classification_head.weight)
        nn.init.zeros_(self.classification_head.bias)

    def forward(self, inputs):
        batch_size, channels, height, width = inputs.size()
        pooled_inputs = self.adaptive_pool(inputs)
        flat_inputs = pooled_inputs.view(batch_size, -1)
        shared_features = self.mlp_shared(flat_inputs)
        unit_scores = self.unit_head(shared_features)
        classification_logits = self.classification_head(shared_features)

        noise = torch.normal(0, 0.01, size=unit_scores.size()).to(unit_scores.device)
        unit_scores += noise

        top_k_scores, top_k_indices = torch.topk(unit_scores, self.top_k, dim=1)
        probabilities = F.softmax(top_k_scores, dim=1)

        unit_usage = torch.zeros(self.num_units, device=inputs.device)
        for idx in top_k_indices.view(-1):
            unit_usage[idx] += 1
        unit_usage = unit_usage / (batch_size * self.top_k)

        # L2
        l2_regularization = 0
        for param in self.mlp_shared.parameters():
            l2_regularization += torch.norm(param, p=2)
        for param in self.unit_head.parameters():
            l2_regularization += torch.norm(param, p=2)
        for param in self.classification_head.parameters():
            l2_regularization += torch.norm(param, p=2)
        l2_regularization = self.l2_lambda * l2_regularization

        return probabilities, top_k_indices, classification_logits, l2_regularization, unit_usage

def compute_load_balancing_loss(unit_usage):
    mean_usage = unit_usage.mean()
    variance = ((unit_usage - mean_usage) ** 2).mean()
    return variance

class ScatterFunction(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, indices):
        ctx.save_for_backward(inputs, indices)
        batch_size, num_units, channels, height, width = inputs.size()
        unit_outputs = torch.zeros(batch_size, num_units, channels, height, width, device=inputs.device)
        indices_expanded = indices.unsqueeze(2).expand(-1, -1, channels, -1, -1)
        unit_outputs.scatter_(1, indices_expanded, inputs)
        return unit_outputs

    @staticmethod
    def backward(ctx, grad_output):
        inputs, indices = ctx.saved_tensors
        batch_size, num_units, channels, height, width = grad_output.size()
        grad_inputs = torch.zeros_like(inputs)
        grad_inputs = grad_inputs.scatter_add_(1, indices.unsqueeze(2).expand(-1, -1, channels, -1, -1), grad_output)

        return grad_inputs, None

class MixtureOfUnits(nn.Module):
    def __init__(self, input_channels, output_channels, num_units, top_k, num_labels):
        super().__init__()
        self.num_units = num_units
        self.top_k = top_k
        self.router = TopKRouter(input_channels, num_units, top_k, num_labels)
        self.units = nn.ModuleList([
            Unit(input_channels, output_channels) for _ in range(num_units)
        ])
        self.gate = nn.Softmax(dim=-1)

    def forward(self, inputs, file_name=None):
        device = inputs.device
        batch_size, channels, height, width = inputs.size()

        probabilities, unit_indices, classification_logits, l2_regularization, unit_usage = self.router(inputs)
        load_balancing_loss = compute_load_balancing_loss(unit_usage)

        expanded_inputs = inputs.unsqueeze(1).expand(-1, self.num_units, -1, -1, -1)
        indices = unit_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
        unit_outputs = ScatterFunction.apply(expanded_inputs, indices)
        unit_results = [unit(unit_outputs[:, i]) for i, unit in enumerate(self.units)]
        combined_results = torch.stack(unit_results, dim=1)
        gate_weights = self.gate(combined_results.mean(dim=[2, 3, 4]))
        gate_weights = gate_weights.view(batch_size, self.num_units, 1, 1, 1)
        final_output = torch.sum(combined_results * gate_weights, dim=1)

        return final_output, classification_logits, l2_regularization, unit_indices, load_balancing_loss
