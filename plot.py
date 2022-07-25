import pickle
import numpy as np
import matplotlib.pyplot as plt

fig=plt.figure()
ax=fig.add_subplot(111)
ax.set_title('trainning loss per epoch')
ax.set_ylabel('loss value')
ax.set_xlabel('epoch')

with open('data/train_loss_list.pickle', 'rb') as handle:
    train = pickle.load(handle)
with open('data/test_loss_list.pickle', 'rb') as handle:
    test = pickle.load(handle)
print('train',type(train),train)
print('test',type(test),test)

train_avgs=[]
test_avgs=[]
for tr,tst in zip(train,test):
    train_avgs.append(np.mean(tr))
    test_avgs.append(np.mean(tst))

x=[i + 1 for i in range(len(train_avgs))]

plt.plot(train_avgs,'go-')
plt.plot(test_avgs,'ro-')
plt.show()