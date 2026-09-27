import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the Surrogate Gradient for Spiking Functions
class SurrogateSpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()  # Spike if greater than 0

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_grad = grad_input / (
            torch.abs(input) + 1e-10
        )  # Surrogate gradient approximation
        return spike_grad


# Leaky Integrate-and-Fire (LIF) Neuron Layer
class LIFNeuronLayer(nn.Module):
    def __init__(self, in_features, out_features, tau_mem=20.0, dt=1.0):
        super(LIFNeuronLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.tau_mem = tau_mem
        self.dt = dt
        self.threshold = 1.0
        self.reset()

    def reset(self):
        self.membrane_potential = torch.zeros(self.linear.out_features)

    def forward(self, x):
        self.membrane_potential = self.membrane_potential * (
            1 - self.dt / self.tau_mem
        ) + self.linear(x)
        spikes = SurrogateSpikeFunction.apply(self.membrane_potential - self.threshold)
        self.membrane_potential = self.membrane_potential * (1 - spikes)  # Resetting
        return spikes


# Simple SNN with Temporal Dynamics
class TemporalSNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, time_steps=10):
        super(TemporalSNN, self).__init__()
        self.time_steps = time_steps
        self.lif1 = LIFNeuronLayer(input_size, hidden_size)
        self.lif2 = LIFNeuronLayer(hidden_size, output_size)

    def forward(self, x):
        # Assuming input x is of shape (batch_size, time_steps, input_size)
        batch_size, time_steps, _ = x.size()
        spike_record = []

        for step in range(time_steps):
            cur_x = x[:, step, :]
            spikes1 = self.lif1(cur_x)
            spikes2 = self.lif2(spikes1)
            spike_record.append(spikes2)

        # Reset states
        self.lif1.reset()
        self.lif2.reset()

        # Example way to aggregate spikes over time (e.g., sum or mean)
        output = torch.stack(spike_record, dim=1).sum(dim=1)  # Summing spikes over time
        return output


# Example Usage
input_size = 10
hidden_size = 20
output_size = 2
time_steps = 10

model = TemporalSNN(input_size, hidden_size, output_size, time_steps)
input_data = torch.rand(5, time_steps, input_size)  # Example batch of size 5

output = model(input_data)
print("Output shape:", output.shape)
