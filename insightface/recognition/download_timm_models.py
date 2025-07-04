import timm


model = timm.create_model("convnext_nano.in12k", pretrained=True)

model = timm.create_model("mobilenetv4_conv_small.e1200_r224_in1k", pretrained=True)
model = timm.create_model("mobilenetv4_conv_medium.e500_r256_in1k", pretrained=True)

model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True)
model = timm.create_model("vit_medium_patch16_gap_256.sw_in12k_ft_in1k", pretrained=True)
    