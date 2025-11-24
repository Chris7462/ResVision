from torchvision.models import resnet101
from torchvision.models import ResNet101_Weights
from torchvision.models._utils import IntermediateLayerGetter


weights = ResNet101_Weights.DEFAULT
backbone = resnet101(weights=weights)

# Freeze backbone layers
for param in backbone.parameters():
    param.requires_grad = False

# Extract features from layer1, layer2, layer3, and layer4
# These correspond to conv2_x, conv3_x, conv4_x, and conv5_x
# The output channels for these layers are 256, 512, 1024, and 2048 respectively
return_layers = {
    'layer1': 'x1',  # conv2_x: (N, 256, H/4, W/4)
    'layer2': 'x2',  # conv3_x: (N, 512, H/8, W/8)
    'layer3': 'x3',  # conv4_x: (N, 1024, H/16, W/16)
    'layer4': 'x4',  # conv5_x: (N, 2048, H/32, W/32)
}
backbone_features = IntermediateLayerGetter(resnet101, return_layers=return_layers)
