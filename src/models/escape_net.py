import math

import torch
import torch.nn as nn


class ESCAPE_NET(nn.Module):
    def __init__(
        self, model_name="IG1", labels=3, dataset="RAT4", kernel_size=8, dropout=0.2
    ):
        super(ESCAPE_NET, self).__init__()
        cfg = {
            "ESCAPE_NET": [64, "A", 64, "A", 64],
            "IG1": [64, "A"],
            "IG2": [64, "A", 64, "A"],
        }

        self.dataset = dataset
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.model_name = model_name

        self.features, self.classifier = self._make_layers(cfg[self.model_name])
        # self._initialize_weights2()

    def _initialize_weights2(self):
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        # Defining the feature extraction layers
        layers = []
        in_channels = 1
        padding = "same"

        if self.model_name == "IG1":
            pool_size = [4]
        elif self.model_name == "IG2":
            pool_size = [2, 4]
        elif self.model_name == "ESCAPE_NET":
            pool_size = [2, 2, 2]
        count = 0
        for i, x in enumerate(cfg):
            stride = 1

            if x == "A":
                layers += [nn.AvgPool2d(kernel_size=pool_size[count])]
                count += 1
            else:
                layers += [
                    nn.Conv2d(
                        in_channels,
                        x,
                        kernel_size=self.kernel_size,
                        padding="same",
                        stride=stride,
                        bias=False,
                    )
                ]
                layers += [nn.ReLU(inplace=True)]
                # layers += [nn.Dropout(self.dropout)]
                in_channels = x
                self.kernel_size = self.kernel_size // 2

        features = nn.Sequential(*layers)
        feature_layers = len(layers)

        # Defining the classification layers
        if self.model_name == "ESCAPE_NET":
            classifier = nn.Sequential(
                nn.Linear(64 * 14 * 25, 256, bias=False),
                nn.ReLU(inplace=True),
                # nn.Dropout(0.5),
                nn.Linear(256, 3, bias=False),
                nn.Softmax(dim=1),
            )
        if self.model_name == "IG1":
            classifier = nn.Sequential(
                nn.Linear(64 * 14 * 25, 256, bias=False),
                nn.ReLU(inplace=True),
                # nn.Dropout(0.5),
                nn.Linear(256, 3, bias=False),
                nn.Softmax(dim=1),
            )
        if self.model_name == "IG2":
            classifier = nn.Sequential(
                nn.Linear(64 * 7 * 12, 256, bias=False),
                nn.ReLU(inplace=True),
                # nn.Dropout(0.5),
                nn.Linear(256, 3, bias=False),
                nn.Softmax(dim=1),
            )
        return (features, classifier)


def test():
    for a in ["IG1", "IG2", "ESCAPE_NET"]:
        net = ESCAPE_NET(a)
        x = torch.randn(64, 1, 56, 100)
        net = nn.DataParallel(net)
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            net.cuda()
            x = x.cuda()
        y = net(x)
        print(net)


if __name__ == "__main__":
    test()
