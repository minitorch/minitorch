from mnist import MNIST
import minitorch
import visdom
import numpy

vis = visdom.Visdom()
mndata = MNIST("data/")
images, labels = mndata.load_training()


BACKEND = minitorch.make_tensor_functions(minitorch.FastOps)
RATE = 0.01
HIDDEN = 20
BATCH = 16


class Network(minitorch.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = MMLinear(784, HIDDEN)
        self.layer2 = MMLinear(HIDDEN, HIDDEN)
        self.layer3 = MMLinear(HIDDEN, 1)

    def forward(self, x):
        # ASSIGN1.5
        h = self.layer1.forward(x).relu()
        h = self.layer2.forward(h).relu()
        return self.layer3.forward(h).sigmoid()
        # END ASSIGN2.5


class MMLinear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        r = minitorch.rand((in_size, out_size))
        r.type_(BACKEND)
        self.weights = minitorch.Parameter(0.1 * (r - 0.5))
        r = minitorch.rand((out_size,))
        r.type_(BACKEND)
        self.bias = minitorch.Parameter(0.1 * (r - 0.5))
        self.out_size = out_size

    def forward(self, x):
        # ASSIGN3.5
        batch, in_size = x.shape
        return minitorch.matmul(
            x.view(batch, 1, in_size),
            self.weights.value.view(1, in_size, self.out_size),
        ).view(batch, self.out_size) + self.bias.value.view(1, self.out_size)
        # END ASSIGN3.5


class Conv2d(minitorch.Module):
    def __init__(self, in_channels, out_channels, kh, kw):
        super().__init__()
        r = minitorch.rand((out_channels, in_channels, kh, kw))
        r.type_(BACKEND)
        self.weights = minitorch.Parameter(0.1 * (r - 0.5))
        r = minitorch.rand((out_channels, 1, 1))
        r.type_(BACKEND)
        self.bias = minitorch.Parameter(0.1 * (r - 0.5))

    def forward(self, input):
        out = minitorch.Conv2dFun.apply(input, self.weights.value) + self.bias.value
        return out


class Network2(minitorch.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 4, 3, 3)
        self.conv2 = Conv2d(4, 8, 3, 3)
        self.linear1 = MMLinear(392, 64)
        self.linear2 = MMLinear(64, 1)

        # For vis
        self.mid = None
        self.out = None

    def forward(self, x):
        x = self.conv1(x).relu()
        self.mid = x
        x = self.conv2(x).relu()
        self.out = x
        x = minitorch.avgpool2d(x, (4, 4))
        x = self.linear1(x.view(BATCH, 392)).relu()
        x = self.linear2(x).sigmoid()
        return x


ys = []
X = []
for i in range(10000):
    y = labels[i]
    if y == 3 or y == 5:
        ys.append(1.0 if y == 3 else 0.0)
        X += images[i]

val_ys = []
val_x = []
for i in range(10000, 10500):
    y = labels[i]
    if y == 3 or y == 5:
        val_ys.append(1.0 if y == 3 else 0.0)
        val_x += images[i]
vis.images(
    numpy.array(val_x).reshape((len(val_ys), 1, 28, 28))[:BATCH], win="val_images"
)


model = Network2()
losses = []
for epoch in range(250):
    total_loss = 0.0
    cur = 0
    for i, j in enumerate(range(0, len(ys), BATCH)):
        if len(ys) - j <= BATCH:
            continue
        y = minitorch.tensor(ys[j : j + BATCH], (BATCH,))
        x = minitorch.tensor(X[cur : cur + 28 * 28 * BATCH], (BATCH, 28 * 28))
        x.requires_grad_(True)
        y.requires_grad_(True)
        y.type_(BACKEND)
        x.type_(BACKEND)
        # Forward
        out = model.forward(x.view(BATCH, 1, 28, 28)).view(BATCH)
        prob = (out * y) + (out - 1.0) * (y - 1.0)
        loss = -prob.log()
        (loss.sum().view(1)).backward()
        total_loss += loss[0]
        losses.append(total_loss)

        # Update
        for p in model.parameters():
            if p.value.grad is not None:
                p.update(p.value - RATE * (p.value.grad / float(BATCH)))
        if i % 10 == 0:
            correct = 0
            y = minitorch.tensor(val_ys[:BATCH], (BATCH,))
            x = minitorch.tensor(val_x[: (BATCH * 28 * 28)], (BATCH, 28 * 28))
            out = model.forward(x.view(BATCH, 1, 28, 28)).view(BATCH)
            for i in range(BATCH):
                if y[i] == 1 and out[i] > 0.5:
                    correct += 1
                if y[i] == 0 and out[i] < 0.5:
                    correct += 1
            for channel in range(4):
                vis.images(
                    -1 * model.mid.to_numpy()[:, channel : channel + 1],
                    win=f"mid_images_{channel}",
                    opts=dict(nrow=4, caption=f"mid_images_{channel}"),
                )
            for channel in range(8):
                vis.images(
                    -1 * model.out.to_numpy()[:, channel : channel + 1],
                    win=f"out_images_{channel}",
                    opts=dict(nrow=4, caption=f"out_images_{channel}"),
                )

            print("Epoch ", epoch, " loss ", total_loss, "correct", correct)
            # im = f"Epoch: {epoch}"
            # data.graph(im, lambda x: model.forward(minitorch.tensor(x, (1, 2)))[0, 0])
            # plt.plot(losses, c="blue")
            # vis.matplot(plt, win="loss")
            total_loss = 0.0
        cur += 28 * 28 * BATCH
