from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

A = -0.75, -0.25, 0, 0.25, 0.5, 0.75, 1.0
B = 0.73, 0.97, 1.0, 0.97, 0.88, 0.73, 0.54
label='ha','ee','we','sd','wr','cv','sg'

plt.plot(A,B,'ro')
for ind,xy in enumerate(zip(A, B)):
    ax.annotate(label[ind], xy=(xy[0],xy[1]), textcoords='data')

plt.grid()
plt.show()


