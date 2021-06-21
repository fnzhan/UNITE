import numpy as np
import torch
from geomloss import SamplesLoss
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


num = 32

x1 = np.random.random(num)
y1 = np.random.random(num)
z1 = np.random.random(num)

x2 = np.random.random(num) + 0.5
y2 = np.random.random(num) + 0.5
z2 = np.random.random(num) + 0.5

d1 = np.concatenate((x1[..., np.newaxis], y1[..., np.newaxis], z1[..., np.newaxis]), axis=1)
d2 = np.concatenate((x2[..., np.newaxis], y2[..., np.newaxis], z2[..., np.newaxis]), axis=1)


fig = plt.figure()
ax = Axes3D(fig)

for n1 in range(num):
    p1 = d1[n1]
    r = 100
    idx = 0
    for n2 in range(num):
        p2 = d2[n2]
        tmp = np.linalg.norm(p1-p2)
        if tmp < r:
            r = tmp
            idx = n2
    plt.plot([x1[n1], x2[idx]], [y1[n1], y2[idx]], [z1[n1], z2[idx]], color='#808080', linewidth=1.0)

ax.scatter(x1, y1, z1, c='r', label='Input Image')
ax.scatter(x2, y2, z2, c='b', label='Reference Image')

ax.legend(loc='best')

ax.set_zlabel('Z', fontdict={'size': 15, 'color': '#000000'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': '#000000'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': '#000000'})

plt.savefig('euclidean.png')





fig = plt.figure()
ax = Axes3D(fig)

sampleloss = SamplesLoss("sinkhorn", p=2, blur=1.0, debias=False, potentials=True)
d1_tensor = torch.from_numpy(d1).view(1, num, 3)  #.cuda()
d2_tensor = torch.from_numpy(d2).view(1, num, 3)  #.cuda()
F_, G_ = sampleloss(d1_tensor, d2_tensor)

_, N, D = d1_tensor.shape
p, blur = 2, 0.05
eps = blur ** p
x_i, y_j = d1_tensor.view(-1, N, 1, D), d2_tensor.view(-1, 1, N, D)
F_i, G_j = F_.view(-1, N, 1), G_.view(-1, 1, N)
C_ij = (1 / p) * ((x_i - y_j) ** p).sum(-1)
f = ((F_i + G_j - C_ij) / eps).exp()

f = f[0].cpu().detach().numpy()

for n1 in range(num):
    idx = np.argmax(f[n1])
    # print (idx)
    plt.plot([x1[n1], x2[idx]], [y1[n1], y2[idx]], [z1[n1], z2[idx]], color='#808080', linewidth=1.0)

ax.scatter(x1, y1, z1, c='r', label='Input Image')
ax.scatter(x2, y2, z2, c='b', label='Reference Image')

ax.legend(loc='best')

ax.set_zlabel('Z', fontdict={'size': 15, 'color': '#000000'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': '#000000'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': '#000000'})
plt.savefig('ot.png')