import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class TSM(nn.Module):
    def __init__(self, n_segment=8, fold_div=8):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div
    
    def temporal_shift(self, x, n_segment=8, fold_div=8):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]           # shift left
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]  # shift right
        out[:, :, 2*fold:] = x[:, :, 2*fold:]          # no shift

        return out.view(nt, c, h, w)

    def forward(self, x):
        return self.temporal_shift(x, self.n_segment, self.fold_div)
    
class SimAM(nn.Module):
    def __init__(self, channels, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        var = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n
        attention = x_minus_mu_square / (4 * (var + self.e_lambda)) + 0.5
        return x * torch.sigmoid(attention)

class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)

class Identity(torch.nn.Module):
    def forward(self, input):
        return input

class SegmentConsensus(torch.nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(SegmentConsensus, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output
    

class MobileNetV4TSMInflated(nn.Module):
    def __init__(self, num_segments=8, new_length=5, num_classes=2):
        super().__init__()
        # self.in_channels = T * C  # 要 inflate 的 channel 數
        self.new_length = new_length
        self.num_segments = num_segments

        # 讀取官方 MobileNetV3 模型
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # 拆解主幹
        name = ["features", "avgpool", "classifier"]
        blocks = list(model.children())
        blocks = dict(zip(name, blocks))

        # SimAM + TSM 插入
        backbone = list(blocks["features"].children())
        inverted_residuals = backbone[1:-1]  # 排除 stem & head conv
        v4_blocks = [backbone[0]]  # 保留 stem conv

        for block in inverted_residuals:
            sub_block = block.block
            modules = list(sub_block.children())
            new_modules = []

            # 插入 TSM
            new_modules.append(TSM(n_segment=num_segments))

            for m in modules:
                new_modules.append(m)
                if m.__class__.__name__ == "SqueezeExcitation":
                    out_ch = m.fc2.out_channels
                    new_modules.append(SimAM(out_ch))
            
            new_block = nn.Sequential(*new_modules)
            v4_blocks.append(nn.Sequential(new_block))

        v4_blocks.append(backbone[-1])  # head conv

        # 重建 backbone
        self.features = nn.Sequential(*v4_blocks)
        self.avgpool = blocks["avgpool"]
        self.classifier = blocks["classifier"]
        self.classifier[3] = nn.Linear(self.classifier[3].in_features, num_classes)

        self.consensus = ConsensusModule(consensus_type="avg", dim=1)

        # inflate input conv channels
        conv1_temp = self.features[0][0]
        params = [x.clone() for x in conv1_temp.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (3 * new_length,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        new_conv1 = nn.Conv2d(3*new_length, conv1_temp.out_channels, 3, stride=2, padding=1, bias=False)
        new_conv1.weight.data = new_kernels
        self.features[0][0] = new_conv1


    def forward(self, x):  # x: [B, T, C, H, W]
        x = self.features(x) # B * num_segments * new_length, C, H, W
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = x.view(x.shape[0]//self.num_segments, self.num_segments, -1)
        x = self.consensus(x).squeeze(1)
        x = self.classifier(x)
        return x