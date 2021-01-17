import numpy as np
import matplotlib.pyplot as plt


def prop(a, b, d):
    return 1/(1+(1/b*d)**a)


n = 3
alphas = [2, 3, 8]
betas = [2, 3, 5]
colors = ['r', 'g', 'b']

d = np.arange(0.0, 10.0, 0.1)

ax = plt.subplot()
for i in range(n):
    res = prop(alphas[i], betas[i], d)
    label = 'alpha= {}, beta= {}'.format(alphas[i], betas[i])
    ax.plot(d, res, color=colors[i], label=label)

plt.xlabel('d(i,j)')
plt.ylabel('p(i,j)')
plt.legend()
plt.show()
