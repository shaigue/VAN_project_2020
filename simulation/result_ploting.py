from matplotlib import pyplot as plt

corr_acc = [0.9888, 0.97595, 0.9765625, 0.971472, 0.961288]
corr_msde = [0.00816, 0.01802, 0.02172, 0.02328, 0.031168]
ind_acc = [0.9895, 0.97681, 0.97725, 0.972288, 0.962064]
ind_msde = [0.009944, 0.022867, 0.022117, 0.026714, 0.037416]

exp = [1,2,3,4,5]
plt.figure()
plt.title("accuracy")
plt.plot(exp, corr_acc, color='b', label="correlated")
plt.plot(exp, ind_acc, color='r', label="independend")
plt.xlabel("experiment number")
plt.ylabel("accuracy")
plt.legend()

plt.figure()
plt.title("MSDE")
plt.plot(exp, corr_msde, color='b', label="correlated")
plt.plot(exp, ind_msde, color='r', label="independend")
plt.xlabel("experiment number")
plt.ylabel("MSDE")
plt.legend()

plt.show()