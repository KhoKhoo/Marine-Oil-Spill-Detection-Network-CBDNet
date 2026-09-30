import functools

from networks.CBDNet import CBDNet, CBDNet_EfficientNet


def build_model(name, pretrained=True):
    """Return a zero-argument constructor for the requested model."""
    if name == 'resnet34':
        return CBDNet
    if name not in ('effb0', 'effb3'):
        raise ValueError('Unknown model: {}'.format(name))
    variant = 'b0' if name == 'effb0' else 'b3'
    return functools.partial(CBDNet_EfficientNet, variant=variant, pretrained=pretrained)
