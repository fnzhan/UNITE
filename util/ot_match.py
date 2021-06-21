import numpy as np
import torch
from geomloss import SamplesLoss
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sinkhorn import multihead_attn


num = 32

a = 1
b = 0.75

x1 = np.random.random(num) * a
y1 = np.random.random(num) * a
z1 = np.random.random(num) * a

x2 = np.random.random(num) * a + b
y2 = np.random.random(num) * a + b
z2 = np.random.random(num) * a + b

d1 = np.concatenate((x1[..., np.newaxis], y1[..., np.newaxis], z1[..., np.newaxis]), axis=1)
d2 = np.concatenate((x2[..., np.newaxis], y2[..., np.newaxis], z2[..., np.newaxis]), axis=1)





fig = plt.figure()
ax = Axes3D(fig)

sampleloss = SamplesLoss("sinkhorn", p=2, blur=1.0, debias=False, potentials=True)
d1_tensor = torch.from_numpy(d1).view(1, num, 3)  #.cuda()
d2_tensor = torch.from_numpy(d2).view(1, num, 3)  #.cuda()



# F_, G_ = sampleloss(d1_tensor, d2_tensor)
# _, N, D = d1_tensor.shape
# p, blur = 2, 0.05
# eps = blur ** p
# x_i, y_j = d1_tensor.view(-1, N, 1, D), d2_tensor.view(-1, 1, N, D)
# F_i, G_j = F_.view(-1, N, 1), G_.view(-1, 1, N)
# C_ij = (1 / p) * ((x_i - y_j) ** p).sum(-1)
# f = ((F_i + G_j - C_ij) / eps) #.exp()
# f = f[0].cpu().detach().numpy()



f = multihead_attn(d1_tensor, d2_tensor.contiguous(), eps=0.05,
                                 max_iter=100, log_domain=False)
f = f.permute(0, 2, 1)
# f_div_C = F.softmax(f_div_C*1000, dim=-1)
f = f[0].cpu().detach().numpy()




for n1 in range(num):
    idx = np.argmax(f[n1])
    # print (idx)
    plt.plot([x1[n1], x2[idx]], [y1[n1], y2[idx]], [z1[n1], z2[idx]], color='#808080', linewidth=1.5)

# ax.scatter(x1, y1, z1, c='#FF4500', label='Exemplar', s=35)
# ax.scatter(x2, y2, z2, c='b', label='Conditional Input', s=35)
ax.scatter(x1, y1, z1, c='#1E90FF', s=35)  #87CEEB blue  #FF4500  orange
ax.scatter(x2, y2, z2, c='#FF4500', s=35)
# ax.legend(loc='best')

# x_major_locator=MultipleLocator(0.5)
# y_major_locator=MultipleLocator(0.5)
# z_major_locator=MultipleLocator(0.5)
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)
# ax.zaxis.set_major_locator(z_major_locator)
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_zticklabels([])
# ax.set_axis_off()
# plt.axis('off')
# ax.set_zlabel('Z', fontdict={'size': 15, 'color': '#000000'})
# ax.set_ylabel('Y', fontdict={'size': 15, 'color': '#000000'})
# ax.set_xlabel('X', fontdict={'size': 15, 'color': '#000000'})
ax.axis("off")
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
plt.savefig('ot.png')