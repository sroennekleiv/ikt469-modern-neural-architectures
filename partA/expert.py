import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    def __init__(self, in_channels, num_experts):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),   
            nn.Flatten(),
            nn.Linear(16, num_experts)
        )

    def forward(self, x):
        return self.net(x)

class MixtureOfExperts(nn.Module):
    def __init__(self, experts, router):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.router = router
        self.num_experts = len(experts)

    def forward(self, x):
        # Router output: [B, K]
        gate_logits = self.router(x)
        gate_weights = F.softmax(gate_logits, dim=1)  # g_i(x)

        # Expert outputs: list of [B, C]
        expert_outputs = [expert(x) for expert in self.experts]

        # Stack → [B, K, C]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        # Weighted sum: Σ g_i(x) z_i(x)
        gate_weights = gate_weights.unsqueeze(-1)  # [B, K, 1]
        mixed_output = (gate_weights * expert_outputs).sum(dim=1)

        return mixed_output

