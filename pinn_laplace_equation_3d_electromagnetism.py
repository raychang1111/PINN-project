import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import qmc
tf.keras.backend.set_floatx("float64")

def coordinates_transform(r,theta,phi):
  return r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(theta)


v0 = 5
a = 1
max_r = 100
def true_solution(r,theta,phi):
  return v0*a/r
# r
# θ = Theta
# φ = Phi

n_bc = 2
n_data_per_bc = 300
engine = qmc.LatinHypercube(d=2)

#data[i][j][0] 第i邊界面中的第j個點的r座標
#data[i][j][1] 第i邊界面中的第j個點的Theta座標
#data[i][j][2] 第i邊界面中的第j個點的Phi座標
#data[i][j][3] 第i邊界面中的第j個點的邊界值
data = np.zeros([n_bc, n_data_per_bc, 4])

for i, j in zip(range(n_bc), [a,max_r]):
  points = engine.random(n=n_data_per_bc)
  points[:,0] = points[:,0]*np.pi
  points[:,1] = points[:,1]*2*np.pi

  data[i, :, 0] = j
  data[i, :, 1] = points[:,0]
  data[i, :, 2] = points[:,1]



for i in range(n_bc):
  for j in range(n_data_per_bc):
    data[i,j,3]=true_solution(data[i,j,0],data[i,j,1],data[i,j,2])

data = data.reshape(n_data_per_bc * n_bc, 4)

# print(data)

r_d, theta_d, phi_d ,t_d  = map(lambda x: np.expand_dims(x, axis=1),
                    [data[:, 0], data[:, 1], data[:, 2],data[:, 3] ])
# print(r_d)
Nc = 600
engine = qmc.LatinHypercube(d=3)

colloc = engine.random(n=Nc)
colloc[:,0] = colloc[:,0]*(max_r-a)+a
colloc[:,1] = colloc[:,1]*np.pi
colloc[:,2] = colloc[:,2]*2*np.pi

# print(colloc)

r_c, theta_c ,phi_c = map(lambda x: np.expand_dims(x, axis=1),
               [colloc[:, 0], colloc[:, 1], colloc[:, 2]])

#--------------------畫圖------------------------------------------------------------------------------------
X1 ,Y1 ,Z1 = coordinates_transform(r_d, theta_d, phi_d)

X2 ,Y2 ,Z2 = coordinates_transform(r_c, theta_c, phi_c)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制点点图
ax.scatter(X1, Y1, Z1, c='b', s=10, marker='x')
ax.scatter(X2, Y2, Z2, c='r', s=10, marker='.')

# # 设置坐标轴范围
# ax.set_xlim(-max_r, max_r)
# ax.set_ylim(-max_r, max_r)
# ax.set_zlim(-max_r, max_r)
# # 设置坐标轴刻度为原始数据值
# ax.set_xticks(np.linspace(-max_r, max_r, 11))
# ax.set_yticks(np.linspace(-max_r, max_r, 6))
# ax.set_zticks(np.linspace(-max_r, max_r, 6))

# print(X1)
# print("-------------------------------")
# print(X2)

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#視圖角度
# ax.view_init(elev=0, azim=0)

# 设置图形标题
ax.set_title('Boundary Data points and Collocation points')
# plt.show()
#--------------------------------------------------------------------------------------------------------
r_c, theta_c ,phi_c, r_d, theta_d, phi_d, t_d =map(lambda x: tf.convert_to_tensor(x,dtype=tf.float64),[r_c, theta_c ,phi_c, r_d, theta_d, phi_d, t_d])

# print(t_d)

def DNN_builder(in_shape=3, out_shape=1, n_hidden_layers=5, neuron_per_layer=20, actfn="tanh"):
  #使用 Keras 函數式 API 建立一個具有指定架構的全連接神經網路模型
  # input layer 輸入層
  input_layer = tf.keras.layers.Input(shape=(in_shape,))
  # hidden layers 隱藏層
  hidden = [tf.keras.layers.Dense(neuron_per_layer, activation=actfn)(input_layer)]
  for i in range(n_hidden_layers-1):
    new_layer = tf.keras.layers.Dense(neuron_per_layer,activation=actfn,activity_regularizer=None)(hidden[-1])
    hidden.append(new_layer)
  # output layer
  output_layer = tf.keras.layers.Dense(1, activation=None)(hidden[-1])
  # building the model
  name = f"DNN-{n_hidden_layers}"
  model = tf.keras.Model(input_layer, output_layer, name=name)
  return model

# 清除先前的 Tensorflow session (Session 是 TensorFlow 中執行計算的一個環境，它負責分配資源、執行操作並返回結果。)
tf.keras.backend.clear_session()
# 調用 DNN_builder 建立一個具有 5 個隱藏層、每層 20 個神經元 使用 tanh(雙曲正切函式) 激活函數的模型
model = DNN_builder(3, 1, 6, 25, "tanh")##ReLU  tanh
model.summary()


@tf.function
def u(r,theta,phi):
  u = model(tf.concat([r, theta, phi], axis=1))
  return u

# ...Residual equation .. is the Laplacian here for heat equation -> Delta u = 0
# 定義了一個函數 f(x, y),用於計算 PDE(偏微分方程)的殘差。它首先使用 u(x, y) 計算解的近似值,
# 然後對 u 兩次求偏導以得到其二階偏導數之和,也就是 Laplace 算子 Δu。
# 最後返回 Δu 的平方均值,作為 PDE 殘差的 loss
@tf.function
def f(r,theta,phi):
  u0 = u(r,theta,phi)
  #偏微分方程(極座標形式)
  u_r = tf.gradients(u0, r)[0]
  u_theta = tf.gradients(u0, theta)[0]
  u_phi = tf.gradients(u0, phi)[0]

  u_rr = tf.gradients(u_r*r*r, r)[0]
  u_tt = tf.gradients(u_theta*tf.sin(theta), theta)[0]
  u_pp = tf.gradients(u_phi, phi)[0]

  # F = (1/(r*r))*u_rr + (1/(r*r*tf.sin(theta)))*u_tt + (1/(r*r*tf.sin(theta)*tf.sin(theta)))*u_pp
  # F = u_rr + (1/(tf.sin(theta)))*u_tt + (1/(tf.sin(theta)*tf.sin(theta)))*u_pp
  F = tf.sin(theta)*tf.sin(theta)*u_rr + tf.sin(theta)*u_tt + u_pp
  # F = u_rr + u_tt + u_pp + u_theta + u_phi
  return tf.reduce_mean(F**2)


# 定義均方誤差 (MSE) 函數,用於計算神經網路在邊界資料點處的誤差
@tf.function
def mse(val, val_):
  return tf.reduce_mean((val-val_)**2)

# Training .... and print the evolution of the losses during
# a number of epochs defined below ...
# We use the vanilla-Pinns with soft constraints on the boundaries for Dirichlet conditions
# ........................................................................................
# 是訓練神經網路的主要部分。首先初始化一些變數,
# 包括 loss 的初始值、訓練的 epoch 數、優化器和學習率設定、訓練過程中的 loss 記錄等。
loss = 0
epochs = 100000  #迭代次數
# opt = tf.keras.optimizers.legacy.Adam(learning_rate=2e-4)


# ------------------------------------------------------------------------------------
# 定义初始学习率和衰减步骤数
initial_learning_rate = 1e-3
decay_steps = epochs
alpha = 0

# 使用余弦衰减学习率调度器
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_learning_rate, decay_steps=decay_steps,alpha=alpha)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# ------------------------------------------------------------------------------------
# opt = tf.keras.optimizers.Adam(learning_rate=2e-4)

epoch = 0

# 權重係數
lambda1, lambda2 = 1e5, 1

loss_values = np.array([])
l_values = np.array([])
L_values = np.array([])
#
start = time.time() #計時間
# 訓練迴圈
for epoch in range(epochs):
  # print(lr_schedule(epoch).numpy())
  # 使用 GradientTape 計算損失函數的梯度
  with tf.GradientTape() as tape:
    T_ = u(r_d, theta_d, phi_d)  #將邊界資訊(x,y)丟給神經網路 得到預測輸出值
    # loss on physics - PDE
    #將邊界內的點代入神經網路得到輸出u 在代入計算偏微分方程
    #因為Laplace方程Δu=0(輸出值因該為0) 但實際上輸出會因為輸入u的好壞而引響不會剛好是0 故此項數值定義為PDF誤差
    L = f(r_c, theta_c ,phi_c)
    # loss on data
    # 將邊界條件(c)(c是儲存邊界條件的解 ex: u(1,y)=1 ) 與預測輸出值 相減 得到data誤差 (??邊界誤差??)
    l = mse(t_d, T_)
    # Add the loss on data and loss on equation ....
    # 將兩種誤差相加成總誤差
    loss = lambda1*L  + lambda2*l

  # 計算總損失對模型參數的梯度
  g = tape.gradient(loss, model.trainable_weights)
  # 使用優化器更新模型參數
  opt.apply_gradients(zip(g, model.trainable_weights))


  if epoch % 100 == 0 or epoch == epochs-1: # 每隔 100 個 epoch 或最後一個 epoch 記錄當前的損失值
    print(f"{epoch:5}, {loss.numpy():.9f}")
    loss_values = np.append(loss_values, loss)
    L_values = np.append(L_values, L)
    l_values = np.append(l_values, l)

# 總計算時間
end = time.time()
computation_time = {}
computation_time["pinn"] = end - start
print(f"\ncomputation time: {end-start:.3f}\n")
# 繪製總損失 loss 、PDE損失 L 和 邊界點誤差l 隨 epoch 的變化曲線
print("-------------------------------")
# ----------------------------對數畫法------------------------
plt.figure()
plt.semilogy(loss_values, label='total Loss')
plt.xlabel("Epochs" r'($\times 10^2$)',fontsize=16)
plt.legend()
# ------------------------------------------------------------
# plt.figure()
# plt.plot(loss_values, label='total Loss')
# plt.legend()
# plt.xlabel("Epochs" r'($\times 10^2$)',fontsize=16)

# 繪製邊界點誤差 l 和 PDE 殘差損失 L 隨 epoch 的變化曲線。
# ----------------------------對數畫法------------------------
plt.figure()
# plt.legend()
plt.semilogy(l_values, label='Loss_data')
# plt.legend()

plt.semilogy(L_values, label='Loss_PDE')
plt.xlabel("Epochs" r'($\times 10^2$)',fontsize=16)
plt.legend()
#---------------------------------------------------------------
# plt.figure()
# plt.plot(l_values, label='Loss_data')
# plt.plot(L_values, label='Loss_PDE')
# plt.legend()
# plt.xlabel("Epochs" r'($\times 10^2$)',fontsize=16)

n = 100     #設置網格大小 n
r = np.linspace(a, max_r, n)
theta = np.linspace(0, np.pi, n)
phi = np.linspace(0, 2*np.pi, 2*n)
r, theta ,phi = np.meshgrid(r, theta, phi)

r = r.reshape([2*n*n*n, 1])
theta = theta.reshape([2*n*n*n, 1])
phi = phi.reshape([2*n*n*n, 1])


r_T = tf.convert_to_tensor(r)
theta_T = tf.convert_to_tensor(theta)
phi_T = tf.convert_to_tensor(phi)
SS = true_solution(r_T,theta_T,phi_T)
S = u(r_T,theta_T,phi_T)
M=SS-S

M=tf.reduce_mean(tf.abs(M))
print("Absolute error: {}".format(M.numpy()))

# plt.show()

# 測試
# 假设 x, y, true_solution(x, y) 是你的数据
# 並將其座標攤平為 (nn, 1) 維張量 X 和 Y
n = 100
theta = np.linspace(0, np.pi, n)
phi = np.linspace(0, 2*np.pi, 2*n)
theta0 ,phi0 = np.meshgrid(theta, phi)
theta = theta0.reshape([2*n*n, 1])
phi = phi0.reshape([2*n*n, 1])
# 將 theta 和 phi 轉換為 Tensorflow 張量
theta_T = tf.convert_to_tensor(theta)
phi_T = tf.convert_to_tensor(phi)


# print(theta_T.shape)
# print(phi_T.shape)
ran=np.linspace(a,max_r,5)
for i in ran:
  r_T = tf.ones_like(theta_T)*i
  R_T = u(r_T, theta_T, phi_T)
  r0 = R_T.numpy().reshape(2*n, n)
  # print(r0)
  # print(phi0.shape)

  X1 ,Y1 ,Z1 = coordinates_transform(r0, theta0, phi0)


  r1 = np.abs(true_solution(np.ones((2*n, n))*i,theta0,phi0)-r0)
  X2 ,Y2 ,Z2 = coordinates_transform(r1, theta0, phi0)
  


  r2 = true_solution(np.ones((2*n, n))*i,theta0,phi0)
  # print(true_solution(np.ones((2*n, n))*i,theta0,phi0))
  # print(r0)
  X3 ,Y3 ,Z3 = coordinates_transform(r2, theta0, phi0)

  fig = plt.figure(figsize=(18, 6))

  # # 左侧子图
  ax2 = fig.add_subplot(131, projection='3d')
  surf1 = ax2.plot_surface(X1, Y1, Z1, cmap='rainbow')
  ax2.set_xlabel("x", fontsize=16)
  ax2.set_ylabel("y", fontsize=16)
  ax2.set_zlabel("z", fontsize=16)
  ax2.set_title("PINN Solution (r={})".format(i), fontsize=16)

  # 中間子图
  ax3 = fig.add_subplot(132, projection='3d')
  surf2 = ax3.plot_surface(X2, Y2, Z2, cmap='rainbow')
  ax3.set_xlabel("x", fontsize=16)
  ax3.set_ylabel("y", fontsize=16)
  ax3.set_zlabel("z", fontsize=16)
  ax3.set_title("Absolute error (r={})".format(i), fontsize=16)
  # ax3.set_title("Solution (r={})".format(i), fontsize=16)

  # 中間子图
  ax4 = fig.add_subplot(133, projection='3d')
  surf3 = ax4.plot_surface(X3, Y3, Z3, cmap='rainbow')
  ax4.set_xlabel("x", fontsize=16)
  ax4.set_ylabel("y", fontsize=16)
  ax4.set_zlabel("z", fontsize=16)
  ax4.set_title("Solution (r={})".format(i), fontsize=16)

# plt.show()

# 獲取所有圖像編號
fig_nums = plt.get_fignums()

# 使用迴圈保存所有圖像
for i in fig_nums:
    plt.figure(i)
    plt.savefig(f'Laplace equation 3D_electromagnetism 運行結果/figure_{i}.png')

