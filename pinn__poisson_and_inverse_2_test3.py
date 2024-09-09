import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
torch.set_default_dtype(torch.float64)

def tru(x, y):
    tru = np.exp(x*y)+np.sinh(x)
    return tru
# Select associated x derivatives .........
def trudx(x, y):
    trudx=y*np.exp(x*y)+np.cosh(x)
    return trudx

# Select y derviatives .....
def trudy(x, y):
    trudy=x*np.exp(x*y)
    return trudy

k0=1
k1=1
def f(x,y,mu):
  tru = mu[0]*torch.exp(x*y)*(x*x+y*y)+mu[1]*torch.sinh(x)
  return tru

# 定義一個全連接神經網絡
class FCN(nn.Module):
  "Defines a standard fully-connected network in PyTorch"

  def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
    super().__init__()
    activation = nn.Tanh        # 激活函數
    self.fcs = nn.Sequential(*[nn.Linear(N_INPUT, N_HIDDEN),activation()])
    # 重複 N_LAYERS - 1 次，以創建多個隱藏層
    self.fch = nn.Sequential(*[nn.Sequential(*[nn.Linear(N_HIDDEN, N_HIDDEN),activation()]) for _ in range(N_LAYERS-1)])
    self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

  def forward(self, x):
    x = self.fcs(x)
    x = self.fch(x)
    x = self.fce(x)
    return x

### data generation .......
n_bc = 4
n_data_per_bc = 30

# Define the datapoints ............;
engine = qmc.LatinHypercube(d=1)
data = np.zeros([n_bc ,n_data_per_bc , 5])
#data[i][j][0] 第i條邊中的第j個點的x座標
#data[i][j][1] 第i條邊中的第j個點的y座標
#data[i][j][2] 第i條邊中的第j個點的邊界值  tru(x,y)
#data[i][j][3] 第i條邊中的第j個點的對x導數  trudx(x,y)
#data[i][j][4] 第i條邊中的第j個點的對y導數  trudy(x,y)

for i, j in zip(range(n_bc), [0., 1., 0, 1.]):
    points = (engine.random(n=n_data_per_bc)[:, 0] - 0.) * 1
    #points = np.linspace(0, +1, n_data_per_bc)
    if i < 2:
        data[i, :, 0] = j
        data[i, :, 1] = points
    else:
        data[i, :, 0] = points
        data[i, :, 1] = j



# Values of data , derivative/x , and derivative/y ...
for j in range(0,n_data_per_bc):
    # bord x = 0
    # 計算Dirichlet problem 邊界問題
    data[0, j, 2] = tru(data[0, j, 0] ,data[0, j, 1] )
    # 計算Neumann problems 邊界問題 (導數)
    data[0, j, 3] = trudx(data[0, j, 0] ,data[0, j, 1] )
    data[0, j, 4] = trudy(data[0, j, 0] ,data[0, j, 1] )

    # bord x = 1
    data[1, j, 2] = tru(data[1, j, 0] ,data[1, j, 1] )
    data[1, j, 3] = trudx(data[1, j, 0] ,data[1, j, 1] )
    data[1, j, 4] = trudy(data[1, j, 0] ,data[1, j, 1] )

    # bord y = 0
    data[2,j,2] = tru(data[2, j, 0] ,data[2, j, 1] )
    data[2,j,3] = trudx(data[2, j, 0] ,data[2, j, 1] )
    data[2,j,4] = trudy(data[2, j, 0] ,data[2, j, 1] )

    # bord y = 1
    data[3,j,2] = tru(data[3, j, 0] ,data[3, j, 1] )
    data[3,j,3] = trudx(data[3, j, 0] ,data[3, j, 1] )
    data[3,j,4] = trudy(data[3, j, 0] ,data[3, j, 1] )


data = data.reshape(n_data_per_bc * n_bc, 5)

# t_d are data values, t_dx and td_y derivatives wrt x and y respectively .....
x_d, y_d, t_d, t_dx, t_dy = map(lambda x: np.expand_dims(x, axis=1),
                    [data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4] ])

Nc = 500
engine = qmc.LatinHypercube(d=2)
colloc = engine.random(n=Nc)
colloc = 1 * (colloc -0)

#
x_c, y_c = map(lambda x: np.expand_dims(x, axis=1),
               [colloc[:, 0], colloc[:, 1]])
# t_c = tru(x_c, y_c)
# t_cdx = trudx(x_c, y_c)
# t_cdy = trudy(x_c, y_c)

#
plt.figure(figsize=(7, 7))
plt.title("Boundary Data points and Collocation points",fontsize=16)
plt.scatter(data[:,0], data[:,1], marker="x", c="k", label="BDP")
plt.scatter(colloc[:,0], colloc[:,1], s=2, marker=".", c="r", label="CP")
plt.xlabel("x",fontsize=16)
plt.ylabel("y",fontsize=16)
plt.axis("square")
# plt.show()

#

x_d, y_d, t_d, t_dx, t_dy, x_c, y_c  = map(lambda x: torch.tensor(x,requires_grad=True).view(-1,1),
                             [x_d, y_d, t_d, t_dx, t_dy, x_c, y_c])

# pinn = FCN(2,1,20,5)
# u=pinn(torch.cat([x_c, y_c], dim=1))
# uderx = torch.autograd.grad(u, x_c, torch.ones_like(u), create_graph=True)[0]
# print(uderx)
# print(x_c)
# print(y_c.shape)
# print(x_d.shape)
# print(y_d.shape)
# print(t_d.shape)
# print(t_dx.shape)
# print(t_dy.shape)

torch.manual_seed(7777)
pinn = FCN(2,1,25,6)


def u_nn(x, y):
    u = pinn(torch.cat([x, y], dim=1))  #排成[[x0,y0],
                         #   [x1,y1]]
    return u

def uderx(x, y):
    u = pinn(torch.cat([x, y], dim=1))
    uderx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    return uderx

def udery(x, y):
    u = pinn(torch.cat([x, y], dim=1))
    udery = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    return udery

def F_nn(x, y,mu):
    u0 = u_nn(x,y)
    u_x = torch.autograd.grad(u0, x, torch.ones_like(u0), create_graph=True)[0]
    u_y = torch.autograd.grad(u0, y, torch.ones_like(u0), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    # Select the residual equation ....
    F = u_xx + u_yy - f(x,y,mu)
    retour = torch.mean(F ** 2)
    return retour

def mse(y, y_):
    return torch.mean((y-y_)**2)

# training .....
epochs = 80000
#定義了兩個可學習的參數mu(標量),用於估計微分方程中的未知常數
mu = torch.nn.Parameter(torch.zeros(2, requires_grad=True))
# optimiser = torch.optim.Adam(list(pinn.parameters())+[mu],lr=1e-3) #初始化優化器(使用Adam優化算法)
#初始化優化器(使用Adam優化算法)
optimiser1 = torch.optim.Adam(pinn.parameters(),lr=1e-4)
optimiser2 = torch.optim.Adam([mu],lr=1e-1)

mus = np.empty((0, 2)) # 创建一个空的二维数组，形状为 (0, 2)

loss_values = np.array([])
L_values = np.array([])
l_values = np.array([])
lambda1, lambda2 = 1, 1 #控制不同loss的權重
start = time.time()
for epoch in range(epochs):
    optimiser1.zero_grad() #将梯度清零 防止梯度值累加
    optimiser2.zero_grad()
    #loss on PDE
    L = F_nn(x_c, y_c,mu)
    #loss on data
    T_ = u_nn(x_d, y_d)
    l = mse(t_d, T_)
    T_ = uderx(x_d, y_d)
    l += mse(t_dx, T_)
    T_ = udery(x_d, y_d)
    l += mse(t_dy, T_)

    loss = lambda1*L + lambda2*l
    #計算加權總損失loss,對其進行反向傳播來獲得所有參數的梯度
    loss.backward()
    #使用優化器的step()方法更新模型參數
    optimiser1.step()
    optimiser2.step()
    mus = np.append(mus, np.array([[mu[0].item(),mu[1].item()]]),axis=0)
    if epoch % 100 == 0 or epoch == epochs-1:
      print(f"{epoch:5}, {loss.detach().numpy():.9f}")
      loss_values = np.append(loss_values, loss.detach().numpy())
      L_values = np.append(L_values, L.detach().numpy())
      l_values = np.append(l_values, l.detach().numpy())

#
end = time.time()
computation_time = {}
computation_time["pinn"] = end - start
print(f"\ncomputation time: {end-start:.3f}\n")
#
plt.figure()
#plt.semilogy(loss_values, label=model.name)
plt.semilogy(loss_values, label="Total loss")
plt.xlabel("Epochs" r'($\times 10^2$)',fontsize=16)
plt.legend()

plt.figure()
plt.title("Inverse problem - discover $\mu$0 value")
plt.plot(mus[:,0], label="PINN estimate", color='red')
plt.hlines(k0, 0, len(mus[:,0]), label="True value", color="tab:blue")
plt.legend()
plt.xlabel("Training step")
print(f"The final predicted value of mu is =")
print(mus[-1][0])

plt.figure()
plt.title("Inverse problem - discover $\mu$1 value")
plt.plot(mus[:,1], label="PINN estimate", color='red')
plt.hlines(k1, 0, len(mus[:,1]), label="True value", color="tab:blue")
plt.legend()
plt.xlabel("Training step")
print(f"The final predicted value of mu is =")
print(mus[-1][1])

plt.figure()
plt.semilogy(l_values, label='Loss_data')
plt.legend()
plt.semilogy(L_values, label='Loss_PDE')
plt.xlabel("Epochs" r'($\times 10^2$)',fontsize=16)
plt.legend()

n = 100
l = 1.
r = 2*l/(n+1)
T = np.zeros([n*n, n*n])
### plotting
#plt.figure("", figsize=(16, 8))
plt.figure(figsize=(14, 7))
#


X = np.linspace(0, 1, n)
Y = np.linspace(0, 1, n)
X0, Y0 = np.meshgrid(X, Y)


X = X0.reshape([n*n, 1])
Y = Y0.reshape([n*n, 1])
X_T = torch.tensor(X,requires_grad=True)
Y_T = torch.tensor(Y,requires_grad=True)

# Predicted solution by the network .....
S = u_nn(X_T, Y_T)
S = S.detach().numpy().reshape(n, n)
#

S2=S
plt.subplot(221)
plt.pcolormesh(X0, Y0, S2, cmap="turbo")
#plt.contour(X0, Y0, S2,18,linestyles='dashed',linewidths=1.5)
plt.colorbar(pad=-0.3)
#plt.scatter(data[:, 0], data[:, 1], marker=".", c="r", label="BDP")
#plt.scatter(colloc[:,0], colloc[:,1], marker=".", c="b")
plt.xlabel("X",fontsize=16)
plt.ylabel("Y",fontsize=16)
plt.title("PINN solution",fontsize=16)
plt.tight_layout()
plt.axis("square")

#plt.show()
#

#plt.figure("", figsize=(14, 7))
# True/exact solution to evaluate the error ...
TT = tru(X0,Y0)

TT2 = (TT - S2)
#TT2 = (TT - S2)/TT

plt.subplot(222)
plt.pcolormesh(X0, Y0, TT2, cmap="turbo")
#plt.contour(X0, Y0, TT2,21)
plt.colorbar(pad=-0.3)
plt.xlabel("X",fontsize=16)
plt.ylabel("Y",fontsize=16)
plt.title("different", fontsize=16)
#plt.title("Relative error", fontsize=16)
plt.tight_layout()
plt.axis("square")

fig = plt.figure(figsize=(18, 6))

# 左侧子图
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X0, Y0, S2, cmap='rainbow')
ax1.set_xlabel("x", fontsize=16)
ax1.set_ylabel("y", fontsize=16)
ax1.set_zlabel("u(x,y)", fontsize=16)
ax1.set_title("PINN Solution", fontsize=16)

# 右侧子图
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(X0, Y0, TT2, cmap='rainbow')
ax2.set_xlabel("x", fontsize=16)
ax2.set_ylabel("y", fontsize=16)
ax2.set_zlabel("u(x,y)", fontsize=16)
ax2.set_title("different", fontsize=16)

ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(X0, Y0, TT, cmap='rainbow')
ax3.set_xlabel("x", fontsize=16)
ax3.set_ylabel("y", fontsize=16)
ax3.set_zlabel("u(x,y)", fontsize=16)
ax3.set_title("True Solution", fontsize=16)

# plot f(x,y)
TT = f(X_T,Y_T,[k0,k1])
TT2 = f(X_T,Y_T,mus[-1])
TT = TT.detach().numpy().reshape(n, n)
TT2 = TT2.detach().numpy().reshape(n, n)
TT3 = (TT2-TT)

fig = plt.figure(figsize=(21, 7))

# 左侧子图
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X0, Y0, TT, cmap='rainbow')
ax1.set_xlabel("x", fontsize=16)
ax1.set_ylabel("y", fontsize=16)
ax1.set_zlabel("f(x,y)", fontsize=16)
ax1.set_title("true_f", fontsize=16)

# 右侧子图
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(X0, Y0, TT2, cmap='rainbow')
ax2.set_xlabel("x", fontsize=16)
ax2.set_ylabel("y", fontsize=16)
ax2.set_zlabel("f_nn(x,y)", fontsize=16)
ax2.set_title("PINN_f", fontsize=16)

ax2 = fig.add_subplot(133, projection='3d')
surf2 = ax2.plot_surface(X0, Y0, TT3, cmap='rainbow')
ax2.set_xlabel("x", fontsize=16)
ax2.set_ylabel("y", fontsize=16)
ax2.set_zlabel("f_nn-f", fontsize=16)
ax2.set_title("different", fontsize=16)

u0 = u_nn(X_T, Y_T)
u_x = torch.autograd.grad(u0, X_T, torch.ones_like(u0), create_graph=True)[0]
u_y = torch.autograd.grad(u0, Y_T, torch.ones_like(u0), create_graph=True)[0]
u_xx = torch.autograd.grad(u_x, X_T, torch.ones_like(u_x), create_graph=True)[0]
u_yy = torch.autograd.grad(u_y, Y_T, torch.ones_like(u_y), create_graph=True)[0]

# Select the residual equation ....
F = u_xx + u_yy - f(X_T,Y_T,[k0,k1])
S = F.detach().numpy().reshape(n, n)
F1 = u_xx + u_yy - f(X_T,Y_T,mus[-1])
S1 = F1.detach().numpy().reshape(n, n)

fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(121, projection='3d')
surf = ax.plot_surface(X0, Y0, S, cmap='rainbow')
ax.set_xlabel("x", fontsize=16)
ax.set_ylabel("y", fontsize=16)
# ax.set_zlabel("u(x,y)", fontsize=16)
ax.set_title("PDE loss k", fontsize=16)

ax = fig.add_subplot(122, projection='3d')
surf = ax.plot_surface(X0, Y0, S1, cmap='rainbow')
ax.set_xlabel("x", fontsize=16)
ax.set_ylabel("y", fontsize=16)
# ax.set_zlabel("u(x,y)", fontsize=16)
ax.set_title("PDE loss mu", fontsize=16)

# plt.show()

# 獲取所有圖像編號
fig_nums = plt.get_fignums()

# 使用迴圈保存所有圖像
for i in fig_nums:
    plt.figure(i)
    plt.savefig(f'PINN poisson and inverse_2 test3 運行結果/figure_{i}.png')