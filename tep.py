import torch
import torchvision.models as models


pthfile = r'F:/BUAA/10_Ckpt/03_Guided Diffusion/256x256_diffusion.pt'  # .pth文件的路径
model = torch.load(pthfile)  # 设置在cpu环境下查询
print('type:')
print(type(model))  # 查看模型字典长度
print('length:')
print(len(model))
print('key:')
for k in model.keys():  # 查看模型字典里面的key
    print(k)
# for m in model.parameters():
#     print(m)
#     print(m.requires_grad)
# print('value:')
# for k in model:  # 查看模型字典里面的value
#     print(k, model[k])

