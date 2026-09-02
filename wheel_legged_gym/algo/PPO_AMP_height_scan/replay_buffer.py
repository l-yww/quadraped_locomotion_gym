import torch
import numpy as np


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, obs_dim, buffer_size, device, store_labels=False):
        """Initialize a ReplayBuffer object.
        Arguments:
            buffer_size (int): maximum size of buffer
        """
        self.states = torch.zeros(buffer_size, obs_dim).to(device)
        self.next_states = torch.zeros(buffer_size, obs_dim).to(device)
        self.store_labels = store_labels
        if self.store_labels:
            self.labels = torch.zeros(buffer_size, dtype=torch.long, device=device)
        self.buffer_size = buffer_size
        self.device = device

        self.step = 0
        self.num_samples = 0

    def insert(self, states, next_states, labels=None):
        """Add new states to memory."""

        num_states = states.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states
        if self.store_labels:
            if labels is None:
                labels = torch.zeros(num_states, dtype=torch.long, device=self.device)
            else:
                labels = labels.to(device=self.device, dtype=torch.long)
        # ensure to not exceed the buffer size
        if end_idx > self.buffer_size:
            first_len = self.buffer_size - self.step
            self.states[self.step:self.buffer_size] = states[:first_len]
            self.next_states[self.step:self.buffer_size] = next_states[:first_len]
            self.states[:end_idx - self.buffer_size] = states[first_len:]
            self.next_states[:end_idx - self.buffer_size] = next_states[first_len:]
            if self.store_labels:
                self.labels[self.step:self.buffer_size] = labels[:first_len]
                self.labels[:end_idx - self.buffer_size] = labels[first_len:]
        else:
            self.states[start_idx:end_idx] = states
            self.next_states[start_idx:end_idx] = next_states
            if self.store_labels:
                self.labels[start_idx:end_idx] = labels

        # end_idx <= self.num_samples <= self.buffer_size
        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        # loop back to the beginning if the end of the buffer is reached
        self.step = (self.step + num_states) % self.buffer_size

    def feed_forward_generator(self,
                               num_mini_batch,
                               mini_batch_size):
        for _ in range(num_mini_batch):
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            if self.store_labels:
                yield (self.states[sample_idxs].to(self.device),
                       self.next_states[sample_idxs].to(self.device),
                       self.labels[sample_idxs].to(self.device))
            else:
                yield (self.states[sample_idxs].to(self.device),
                       self.next_states[sample_idxs].to(self.device))
