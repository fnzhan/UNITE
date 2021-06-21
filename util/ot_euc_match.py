import numpy as np
import torch
from geomloss import SamplesLoss
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sinkhorn import multihead_attn
from skimage.measure import compare_ssim


num = 32


a1 = 1.0
b1 = 0.0
a2 = 1.2
b2 = 1.2

x1 = np.random.random(num) * a1 + b1
y1 = np.random.random(num) * a1 + b1
z1 = np.random.random(num) * a1 + b1

x2 = np.random.random(num) * a2 + b2
y2 = np.random.random(num) * a2 + b2
z2 = np.random.random(num) * a2 + b2

# x2[:27] = x2[:27] * a2 + b2
# y2[:27] = y2[:27] * a2 + b2
# z2[:27] = z2[:27] * a2 + b2

x2[:-5] = x2[:-5] - 0.5
y2[:-5] = y2[:-5] - 0.5
z2[:-5] = z2[:-5] - 1.25

x2[:5] = x2[:5] + 0.5
y2[:5] = y2[:5] + 0.5
z2[:5] = z2[:5] - 7.5


d1 = np.concatenate((x1[..., np.newaxis], y1[..., np.newaxis], z1[..., np.newaxis]), axis=1)
d2 = np.concatenate((x2[..., np.newaxis], y2[..., np.newaxis], z2[..., np.newaxis]), axis=1)


def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

# euclidean.png
fig = plt.figure()
ax = Axes3D(fig)
for n1 in range(num):
    p1 = d1[n1]
    r = -100
    idx = 0
    for n2 in range(num):
        p2 = d2[n2]
        # tmp = np.linalg.norm(p1-p2)
        tmp = cos_sim(p1, p2)
        if tmp > r:
            r = tmp
            idx = n2
    plt.plot([x1[n1], x2[idx]], [y1[n1], y2[idx]], [z1[n1], z2[idx]], color='#808080', linewidth=1.0)
ax.scatter(x1, y1, z1, c='#1E90FF', s=35)  #87CEEB blue  #FF4500  orange
ax.scatter(x2, y2, z2, c='#FF4500', s=35)
# ax.legend(loc='best')
ax.axis("off")
plt.savefig('cosine.png')





# OT
fig = plt.figure()
ax = Axes3D(fig)
d1_tensor = torch.from_numpy(d1).view(1, num, 3)  #.cuda()
d2_tensor = torch.from_numpy(d2).view(1, num, 3)  #.cuda()
f = multihead_attn(d1_tensor, d2_tensor.contiguous(), eps=0.05,
                                 max_iter=100, log_domain=False)
f = f.permute(0, 2, 1)
# f_div_C = F.softmax(f_div_C*1000, dim=-1)
f = f[0].cpu().detach().numpy()
for n1 in range(num):
    idx = np.argmax(f[n1])
    # print (idx)
    plt.plot([x1[n1], x2[idx]], [y1[n1], y2[idx]], [z1[n1], z2[idx]], color='#808080', linewidth=1.5)
ax.scatter(x1, y1, z1, c='#1E90FF', s=35)  #87CEEB blue  #FF4500  orange
ax.scatter(x2, y2, z2, c='#FF4500', s=35)
ax.axis("off")
plt.savefig('ot.png')





# UOT
fig = plt.figure()
ax = Axes3D(fig)
d1_tensor = torch.from_numpy(d1).view(1, num, 3)  #.cuda()
d2_tensor = torch.from_numpy(d2).view(1, num, 3)  #.cuda()

d1_weight = torch.ones(1, num).double()
d2_weight = torch.ones(1, num).double()
# d1_weight[:5] = d1_weight * 0.1
# print (torch.norm(d1_weight, 2, 1, keepdim=True))
# print (torch.norm(d2_weight, 2, 1, keepdim=True))
d1_weight = d1_weight / torch.norm(d1_weight, 2, 1, keepdim=True)
d2_weight = d2_weight / torch.norm(d2_weight, 2, 1, keepdim=True)
# print (d2_weight)
d2_weight[0, :5] = d2_weight[0, :5] * 0.01
# d2_weight[:5] = d2_weight + 0.3
# torch.norm(phi_w_feat, 2, 1, keepdim=True)

sampleloss = SamplesLoss("sinkhorn", p=2, blur=0.05,
                         debias=False, potentials=True, reach=6.5)

F_, G_ = sampleloss(d1_weight, d1_tensor, d2_weight, d2_tensor)
_, N, D = d1_tensor.shape
p, blur = 2, 0.05
eps = blur ** p
x_i, y_j = d1_tensor.view(-1, N, 1, D), d2_tensor.view(-1, 1, N, D)
F_i, G_j = F_.view(-1, N, 1), G_.view(-1, 1, N)
C_ij = (1 / p) * ((x_i - y_j) ** p).sum(-1)
f = ((F_i + G_j - C_ij) / eps) #.exp()
f = f[0].cpu().detach().numpy()

for n1 in range(num):
    idx = np.argmax(f[n1])
    plt.plot([x1[n1], x2[idx]], [y1[n1], y2[idx]], [z1[n1], z2[idx]], color='#808080', linewidth=1.5)
ax.scatter(x1, y1, z1, c='#1E90FF', s=35)  #87CEEB blue  #FF4500  orange
ax.scatter(x2, y2, z2, c='#FF4500', s=35)
ax.axis("off")
# fig.set_size_inches(width/100.0/3.0, height/100.0/3.0)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
# plt.margins(0,0)
plt.savefig('uot.png')
