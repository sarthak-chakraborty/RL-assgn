import numpy as np
import pickle 
import matplotlib.pyplot as plt

scores=pickle.load(open('train_scores.pkl','rb'), encoding='latin1')
n_scores = np.load('test_scores.npy')


scores_deque = []
x = [i for i in range(len(scores)-1000)]


y=[]
j=0
for i in range(len(scores)-1000):
	if i < 100:
		scores_deque.append(scores[i])
		y.append(np.mean(scores_deque))
	else:
		for j in range(len(scores_deque)-1):
			scores_deque[j] = scores_deque[j+1]
		scores_deque[len(scores_deque)-1]=scores[i]
		y.append(np.mean(scores_deque))




plt.figure()
plt.plot(x, scores[:9001], color='y', alpha=0.5)
plt.plot(x,y,linewidth=0.9,color='k',linestyle='--')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Scores vs Episode')
plt.savefig('Training_graph.png')