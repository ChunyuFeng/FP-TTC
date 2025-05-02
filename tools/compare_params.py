import torch
from fpttc.fp_ttc import FpTTC

# 1. 手动 load checkpoint
ckpt = torch.load('pretrained/fpttc_mix.pth.tar', map_location='cpu')
raw = ckpt.get('net', ckpt.get('state_dict', ckpt))

# 2. 提取 cnet 的键—值对
enc_ckpt = {}
for k,v in raw.items():
    if k.startswith('module.cnet.') or k.startswith('cnet.'):
        new_k = k.split('.', 2)[-1]
        enc_ckpt[new_k] = v

# 3. 实例化模型并加载
model = FpTTC(pretrained_cnet_path=None, freeze_cnet=False)
# 直接使用 load_state_dict 确保和你内部逻辑一致
model.cnet.load_state_dict(enc_ckpt, strict=False)

# 4. 再和原来加载后的模型比
model2 = FpTTC(pretrained_cnet_path='pretrained/fpttc_mix.pth.tar', freeze_cnet=False)

for name, param in model.cnet.state_dict().items():
    p1 = param.detach().cpu()
    p2 = model2.cnet.state_dict()[name].detach().cpu()
    try:
        torch.testing.assert_allclose(p1, p2)
    except AssertionError as e:
        print(f"Mismatch at {name}: {e}")
        break
else:
    print("✅ cnet parameters match exactly!")

for name, p in model2.cnet.named_parameters():
    assert not p.requires_grad, f"{name} 没有冻结！"
print("✅ cnet 参数已全部冻结")
