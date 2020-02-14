import autograd.numpy as np
import autograd
import os
from autograd import grad
from autograd import jacobian
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.linalg import pinv

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--function", type=int, default=1, help="choose from three low dimensional example functions, 1-3")
opt = parser.parse_args()
function = opt.function

# GDA
def gda(z_0, alpha=0.05, num_iter=100):
    z = [z_0]
    grad_fn = grad(target)
    for i in range(num_iter):
        g = grad_fn(z[-1])
        z1 = z[-1] + g*np.array([-1,1])*alpha
        z.append(z1)
    z = np.array(z)
    return z

# Extra Gradient
def eg(z_0, alpha=0.05, num_iter=100):
    z = [z_0]
    grad_fn = grad(target)
    for i in range(num_iter):
        g = grad_fn(z[-1])
        z1 = z[-1] + g*np.array([-1,1])*alpha
        g = grad_fn(z1)
        z2 = z[-1] + g*np.array([-1,1])*alpha
        z.append(z2)
    z = np.array(z)
    return z

# Optimistic Gradient
def ogda(z_0, alpha=0.05, num_iter=100):
    z = [z_0,z_0]
    grads = []
    grad_fn = grad(target)
    for i in range(num_iter):
        g = grad_fn(z[-1])
        gg = grad_fn(z[-2])
        z1 = z[-1] + 2*g*np.array([-1,1])*alpha - gg*np.array([-1,1])*alpha
        z.append(z1)
    z = np.array(z)
    return z

# Consensus Optimization
def co(z_0, alpha=0.01, gamma=0.1, num_iter=100):
    z = [z_0]
    grads = []
    grad_fn = grad(target)
    hessian = jacobian(grad_fn)
    for i in range(num_iter):
        g = grad_fn(z[-1])
        H = hessian(z[-1])
        #print(np.matmul(H,g), z[-1])
        v = g*np.array([1,-1]) + gamma*np.matmul(H,g)
        z1 = z[-1] - alpha*v
        z.append(z1)
    z = np.array(z)
    return z

# Symplectic gradient adjustment
def sga(z_0, alpha=0.05, lamb=0.1, num_iter = 100):
    z = [z_0]
    grad_fn = grad(target)
    hessian = jacobian(grad_fn)
    for i in range(num_iter):
        g = grad_fn(z[-1])
        w = g * np.array([1,-1])
        H = hessian(z[-1])
        HH = np.array([[1, -lamb*H[0,1]],[lamb*H[0,1],1]])
        v = HH @ w
        z1 = z[-1] - alpha*v
        z.append(z1)
    z = np.array(z)
    return z

# Follow the ridge
def follow(z_0, alpha=0.05, num_iter = 100):
    z = [z_0]
    grad_fn = grad(target)
    hessian = jacobian(grad_fn)
    for i in range(num_iter):
        g = grad_fn(z[-1])
        H = hessian(z[-1])
        v = np.array([g[0], -g[1]-H[0,1]*np.squeeze(pinv(H[1:,1:]))*g[0]])
        z1 = z[-1] - alpha*v
        z.append(z1)
    z = np.array(z)
    return z

def f1(z):
    x = z[0]
    y = z[1]
    f = -3*x*x-y*y+4*x*y
    return f

def f2(z):
    x = z[0]
    y = z[1]
    f = 3*x*x+y*y+4*x*y
    return f

def f3(z):
    x = z[0]
    y = z[1]
    f = (0.4*x*x-0.1*(y-3*x+0.05*x*x*x)**2-0.01*y*y*y*y)*np.exp(-0.01*(x*x+y*y))
    return f

# Select target function
if function==1:
    target = f1              # (0,0) is local minimax and global minimax
    z_0 = np.array([5., 7.]) # Set initial point
    plot_width = 12          # Set range of the plot
    root_dir = 'results/f1.pdf'
elif function==2:
    target = f2         # (0,0) is not local minimax and not global minimax
    z_0 = np.array([6., 5.])
    plot_width = 12
    root_dir = 'results/f2.pdf'
elif function==3:
    target = f3         # (0,0) is local minimax
    z_0 = np.array([7., 5.])
    plot_width = 8
    root_dir = 'results/f3.pdf'


# Run all algorithms on target
zfr=follow(z_0, num_iter = 1000, alpha = 0.05)
zgda=gda(z_0, num_iter = 1000, alpha = 0.05)
zogda=ogda(z_0, num_iter = 1000, alpha = 0.05)
zeg=eg(z_0, num_iter = 1000, alpha = 0.05)
zco=co(z_0, num_iter = 1000, alpha = 0.05, gamma=0.1)
zsga=sga(z_0, num_iter = 1000, alpha = 0.01, lamb=1.0)


# Plot trajectory with contour
plt.rcParams.update({'font.size': 14})
def_colors=(plt.rcParams['axes.prop_cycle'].by_key()['color'])


#plot_width=12
plt.figure(figsize=(5,5))
axes = plt.gca()
axes.set_xlim([-plot_width,plot_width])
axes.set_ylim([-plot_width,plot_width])

x1 = np.arange(-plot_width,plot_width,0.1)
y1 = np.arange(-plot_width,plot_width,0.1)
X,Y = np.meshgrid(x1,y1)
Z = np.zeros_like(X)
for i in range(len(x1)):
    for j in range(len(y1)):
        Z[j][i] = target(np.array([x1[i] ,y1[j]]))
plt.contourf(X,Y,Z,30,cmap=plt.cm.gray)

lw = 2
hw = 0.7
line6,=plt.plot(zfr[:,0],zfr[:,1],'-',color='r',linewidth=lw,zorder=10)
line1,=plt.plot(zgda[:,0],zgda[:,1],'--',linewidth=lw,color=def_colors[9],zorder=2)
line2,=plt.plot(zogda[:,0],zogda[:,1],'--',linewidth=lw,color=def_colors[1])
line3,=plt.plot(zeg[:,0],zeg[:,1],'--',linewidth=lw,color=def_colors[2])
line4,=plt.plot(zsga[:,0],zsga[:,1],'--',color=def_colors[0],linewidth=lw)
line5,=plt.plot(zco[:,0],zco[:,1],'--',color='xkcd:violet',linewidth=lw)
init=plt.plot(zfr[0,0],zfr[0,1],'^',zorder=20,ms=12.0,color='r')
plt.legend((line6,line1, line2, line3, line4, line5), ('FR','GDA', 'OGDA', 'EG', 'SGA', 'CO'), loc=4)


os.makedirs('results/', exist_ok=True)
plt.savefig(root_dir, dpi=300)
#plt.show()