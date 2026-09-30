import torch

from networks.CBDNet import CBDNet, CBDNet_EfficientNet

device = torch.device('cpu')


def check_model(name, model):
    model = model.to(device).eval()
    x = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 256, 256), f'{name}: unexpected output shape {tuple(out.shape)}'
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'{name}: output shape {tuple(out.shape)}, params {num_params:.2f}M')


check_model('CBDNet', CBDNet())
check_model('CBDNet_EfficientNet(b0)', CBDNet_EfficientNet(variant='b0', pretrained=False))

print('Smoke test passed.')
