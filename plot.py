import pickle
import numpy as np
import matplotlib.pyplot as plt

fig=plt.figure()
ax=fig.add_subplot(111)
ax.set_title('trainning loss per epoch')
ax.set_ylabel('loss value')
ax.set_xlabel('epoch')

with open('data/loss_list.pickle', 'rb') as handle:
    b = pickle.load(handle)

print(type(b),b)

avgs=[]
for epoch in b:
    avgs.append(np.mean(epoch))

x=[i+1 for i in range(len(avgs))]
y=avgs
plt.plot(x,y)
plt.show()