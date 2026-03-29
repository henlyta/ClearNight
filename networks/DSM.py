import torch
import torch.nn as nn
import torch.nn.functional as F

class Unit(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, padding=kernel_size // 2)

        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, inputs):
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
        batch_size = inputs.size(0)

        pooled_inputs = self.adaptive_pool(inputs)
        flat_inputs = pooled_inputs.view(batch_size, -1)
        shared_features = self.mlp_shared(flat_inputs)

        unit_scores = self.unit_head(shared_features)

        classification_logits = self.classification_head(shared_features)

        noise = torch.normal(0, 0.01, size=unit_scores.size(), device=inputs.device)
        unit_scores_noisy = unit_scores + noise

        top_k_scores, top_k_indices = torch.topk(unit_scores_noisy, self.top_k, dim=1)

        mask = torch.full_like(unit_scores, float('-inf'))
        mask.scatter_(1, top_k_indices, 0.0)
        routed_logits = unit_scores_noisy + mask

        routing_weights = F.softmax(routed_logits, dim=-1)

        unit_usage = routing_weights.mean(dim=0)  # [num_units]

        l2_regularization = sum(p.norm(2) for p in self.parameters()) * self.l2_lambda

        return routing_weights, top_k_indices, classification_logits, l2_regularization, unit_usage


def compute_load_balancing_loss(unit_usage):
    mean_usage = unit_usage.mean()
    variance = ((unit_usage - mean_usage) ** 2).mean()
    return variance

class MixtureOfUnits(nn.Module):
    def __init__(self, input_channels, output_channels, num_units, top_k, num_labels):
        super().__init__()
        self.num_units = num_units
        self.top_k = top_k
        self.router = TopKRouter(input_channels, num_units, top_k, num_labels)
        self.units = nn.ModuleList([
            Unit(input_channels, output_channels) for _ in range(num_units)
        ])

    def forward(self, inputs, file_name=None):
        routing_weights, unit_indices, classification_logits, l2_regularization, unit_usage = self.router(inputs)
        load_balancing_loss = compute_load_balancing_loss(unit_usage)

        final_output = torch.zeros_like(inputs)

        for i, unit in enumerate(self.units):
            expert_weight = routing_weights[:, i].view(-1, 1, 1, 1)
            if expert_weight.sum() > 0:
                final_output += expert_weight * unit(inputs)

        return final_output, classification_logits, l2_regularization, unit_indices, load_balancing_loss
