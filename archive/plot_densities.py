import numpy as np
import matplotlib.pyplot as plt

delta = 0.02
w1 = 0.5

def f_a(w, theta):
    term1 = (1 - theta) * (1 + 1/theta) * (1 - w)**(1 / theta)
    term2 = theta * (1 + 1 / (1 - theta)) * w**(1 / (1 - theta))
    return term1 + term2

def f_b(w, theta):
    if theta < 0.999:
        z = 1 - (1 - 1e-9) * (
            (1 - theta) * (1 - w1 - delta/2)**(1/theta + 1) \
            + theta * (w1 + delta/2)**(1 / (1-theta) + 1) \
            - (1 - theta) * (1 - w1 + delta/2)**(1/theta + 1) \
            - theta * (w1 - delta/2)**(1 / (1 - theta) + 1)
        )
        ind2 = (w > (w1 - delta/2)) & (w < (w1 + delta/2))
        ind1 = 1 - ind2
        return f_a(w, theta) / z * (ind1 + 1e-9 * ind2)
    if theta >= 0.999:
        return f_a(w, theta)


xx = np.arange(0, 1, 0.001)
theta_list = [0.2, 0.4, 0.6, 0.8]

plt.figure(figsize=(12, 3))
for i, theta in enumerate(theta_list):
    plt.subplot(1, 4, i+1)
    yy = f_a(xx, theta)
    plt.plot(xx, yy)
    plt.ylim((-0.2, None))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f"$\\theta = {theta}$", fontsize=18)
    plt.xlabel("$w$", fontsize=15)
    plt.ylabel("$f^a(w; \\theta)$", fontsize=15)
plt.tight_layout()
plt.savefig("figs/density_fa.pdf")

plt.figure(figsize=(12, 3))
for i, theta in enumerate(theta_list):
    plt.subplot(1, 4, i+1)
    yy = f_b(xx, theta)
    plt.plot(xx, yy)
    plt.ylim((-0.2, None))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f"$\\theta = {theta}$", fontsize=18)
    plt.xlabel("$w$", fontsize=15)
    plt.ylabel("$f^b(w; \\theta)$", fontsize=15)
plt.tight_layout()
plt.savefig("figs/density_fb.pdf")