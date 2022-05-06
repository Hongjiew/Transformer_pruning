import torch
import timm
import sys
sys.path.append('../')
from tokenrank_vit import TokenRankVisionTransformer
from vit import VisionTransformerTeacher, checkpoint_filter_fn, _cfg
from PIL import Image
from transformers import ViTFeatureExtractor
import requests

patch_size = 16
layers = 12
prune_list = [2,5,8]
heads = 3
mlp_ratio = 4.
dims = 192
qkv_bias = False
tau_imp = 0.5
tau_prune = 0.1
model = TokenRankVisionTransformer(
                prune_list=prune_list, patch_size=patch_size,  embed_dim=dims, depth=layers,
                num_heads=heads, mlp_ratio=mlp_ratio,qkv_bias=qkv_bias, tau_prune = tau_prune,
                tau_imp=tau_imp)
model_path = '../model_weights/deit_tiny_patch16_224-a1311bcf.pth'
checkpoint = torch.load(model_path,map_location="cpu")
ckpt = checkpoint_filter_fn(checkpoint, model)
model.default_cfg = _cfg()
missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
print('# missing keys=', missing_keys)
print('# unexpected keys=', unexpected_keys)
print('sucessfully loaded from pre-trained weights:', model_path)
teacher_model = VisionTransformerTeacher(
                patch_size=patch_size,  embed_dim=dims, depth=layers,
                num_heads=heads, mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,
                )
missing, unexpected = teacher_model.load_state_dict(ckpt,strict=False)
print('Teacher model loaded')
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
inputs = feature_extractor(images=image, return_tensors="pt")['pixel_values']
model = model.train()
outputs = model(inputs)
print("Train mode working")
model = model.eval()
outputs = model(inputs)
print("Eval mode working")
