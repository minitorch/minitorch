import torch

import minitorch


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class Network(torch.nn.Module):
    def __init__(self, hidden_layers):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        h = self.layer1.forward(x).relu()
        h = self.layer2.forward(h).relu()
        return self.layer3.forward(h).sigmoid()


class Linear(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weight = torch.nn.Parameter(2 * (torch.rand((in_size, out_size)) - 0.5))
        self.bias = torch.nn.Parameter(2 * (torch.rand((out_size,)) - 0.5))

    def forward(self, x):
        return x @ self.weight + self.bias


class TorchTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(torch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(torch.tensor(X)).detach()

    def train(
        self,
        data,
        learning_rate,
        max_epochs=500,
        log_fn=default_log_fn,
    ):
        self.model = Network(self.hidden_layers)
        self.max_epochs = max_epochs
        model = self.model

        losses = []
        for epoch in range(1, max_epochs + 1):

            # Forward
            out = model.forward(torch.tensor(data.X, requires_grad=True)).view(data.N)
            y = torch.tensor(data.y)
            probs = (out * y) + (out - 1.0) * (y - 1.0)
            loss = -probs.log().sum()

            # Update
            loss.view(1).backward()

            for p in model.parameters():
                if p.grad is not None:
                    p.data = p.data - learning_rate * (p.grad / float(data.N))
                    p.grad.zero_()

            # Logging
            pred = out > 0.5
            correct = ((y == 1) * (pred)).sum() + ((y == 0) * (~pred)).sum()
            loss_num = loss.reshape(-1).item()
            losses.append(loss_num)

            if epoch % 10 == 0 or epoch == max_epochs:
                log_fn(epoch, loss_num, correct.item(), losses)


if __name__ == "__main__":
    PTS = 250
    HIDDEN = 10
    RATE = 0.5
    TorchTrain(HIDDEN).train(minitorch.datasets["Xor"](PTS), RATE)
