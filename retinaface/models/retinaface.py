from typing import List, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision


from models.backbones import (
    mobilenet_v1_025,
    mobilenet_v1_050,
    mobilenet_v1,
    mobilenet_v2,
    resnet18,
    resnet34,
    resnet50
)
from torchvision import models
from models.common import SSH, FPN, IntermediateLayerGetterByIndex


def get_layer_extractor(cfg, backbone):
    """
    Selects the appropriate layers from the backbone based on the configuration.

    Args:
        cfg (dict): Configuration dictionary containing the model name and return layers.
        backbone (nn.Module): The backbone network from which to extract layers.

    Returns:
        IntermediateLayerGetter or IntermediateLayerGetterByIndex: The appropriate layer getter.
    """
    if cfg['name'] == "mobilenet_v2":
        return IntermediateLayerGetterByIndex(backbone, [6, 13, 18])
    else:
        return torchvision.models._utils.IntermediateLayerGetter(backbone, cfg['return_layers'])


def build_backbone(name, pretrained=False):
    """
    Builds the backbone of the RetinaFace model based on configuration.

    Args:
        name (str): Backbone name (e.g., 'mobilenet0.25', 'Resnet50').
        pretrained (bool): If True, load pretrained weights.

    Returns:
        nn.Module: The chosen backbone network.
    """
    backbone_map = {
        'mobilenet0.25': lambda: mobilenet_v1_025(pretrained=pretrained),
        'mobilenet0.50': mobilenet_v1_050,
        'mobilenet_v1': mobilenet_v1,
        'mobilenet_v2': lambda: mobilenet_v2(pretrained=pretrained),
        'resnet50': lambda: resnet50(pretrained=pretrained),
        'resnet34': lambda: resnet34(pretrained=pretrained),
        'resnet18': lambda: resnet18(pretrained=pretrained)
    }

    if name not in backbone_map:
        raise ValueError(f"Unsupported backbone name: {name}")

    return backbone_map[name]()


class ClassHead(nn.Module):
    def __init__(self, in_channels: int = 512, num_anchors: int = 2, fpn_num: int = 3) -> None:
        super().__init__()
        self.class_head = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_anchors * 2,
                kernel_size=(1, 1),
                stride=1,
                padding=0
            )
            for _ in range(fpn_num)
        ])

    def forward(self, x: List[Tensor]) -> Tensor:
        outputs = []
        for feature, layer in zip(x, self.class_head):
            outputs.append(layer(feature).permute(0, 2, 3, 1).contiguous())

        outputs = torch.cat([out.view(out.shape[0], -1, 2) for out in outputs], dim=1)
        return outputs


class BboxHead(nn.Module):
    def __init__(self, in_channels: int = 512, num_anchors: int = 2, fpn_num: int = 3) -> None:
        super().__init__()
        self.bbox_head = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_anchors * 4,
                kernel_size=(1, 1),
                stride=1,
                padding=0
            )
            for _ in range(fpn_num)
        ])

    def forward(self, x: List[Tensor]) -> Tensor:
        outputs = []
        for feature, layer in zip(x, self.bbox_head):
            outputs.append(layer(feature).permute(0, 2, 3, 1).contiguous())

        outputs = torch.cat([out.view(out.shape[0], -1, 4) for out in outputs], dim=1)
        return outputs


class LandmarkHead(nn.Module):
    def __init__(self, in_channels: int = 512, num_anchors: int = 2, fpn_num: int = 3) -> None:
        super().__init__()
        self.landmark_head = nn.ModuleList([
            nn.Conv2d(
                in_channels,
                num_anchors * 10,
                kernel_size=(1, 1),
                stride=1,
                padding=0
            )
            for _ in range(fpn_num)
        ])

    def forward(self, x: List[Tensor]) -> Tensor:
        outputs = []
        for feature, layer in zip(x, self.landmark_head):
            outputs.append(layer(feature).permute(0, 2, 3, 1).contiguous())

        outputs = torch.cat([out.view(out.shape[0], -1, 10) for out in outputs], dim=1)
        return outputs


class RetinaFace(nn.Module):
    def __init__(self, cfg: dict = None) -> None:
        """
        RetinaFace model constructor.

        Args:
            cfg (dict): A configuration dictionary containing model parameters.
        """
        super().__init__()
        print(cfg['name'])
        backbone = build_backbone(cfg['name'], cfg['pretrain'])
        self.fx = get_layer_extractor(cfg, backbone)  # feature extraction

        num_anchors = 2
        base_in_channels = cfg['in_channel']
        out_channels = cfg['out_channel']

        if cfg['name'] == "mobilenet_v2":
            fpn_in_channels = [32, 96, 1280]  # mobilenet v2
        else:
            fpn_in_channels = [
                base_in_channels * 2,
                base_in_channels * 4,
                base_in_channels * 8,
            ]

        self.fpn = FPN(fpn_in_channels, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.class_head = ClassHead(in_channels=cfg['out_channel'], num_anchors=num_anchors, fpn_num=3)
        self.bbox_head = BboxHead(in_channels=cfg['out_channel'], num_anchors=num_anchors, fpn_num=3)
        self.landmark_head = LandmarkHead(in_channels=cfg['out_channel'], num_anchors=num_anchors, fpn_num=3)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        out = self.fx(x)
        fpn = self.fpn(out)

        # single-stage headless module
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])

        features = [feature1, feature2, feature3]

        classifications = self.class_head(features)
        bbox_regressions = self.bbox_head(features)
        landmark_regressions = self.landmark_head(features)

        if self.training:
            output = (bbox_regressions, classifications, landmark_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), landmark_regressions)
        return output
