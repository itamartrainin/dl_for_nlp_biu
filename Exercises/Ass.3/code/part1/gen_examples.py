import random
import numpy as np
from xeger import Xeger


LIMIT_REGEX = 100

def generating_data(num_positive=500, num_negative=500):
	pos_samples, neg_samples = [], []
	for i in range(num_positive):
		sample = Xeger(limit=np.floor(LIMIT_REGEX/9))
		sample = sample.xeger("([1-9]+)a+([1-9]+)b+([1-9]+)c+([1-9]+)d+([1-9]+) 1")
		pos_samples.append(sample)
	for i in range(num_negative):
		sample = Xeger(limit=np.floor(LIMIT_REGEX/9))
		sample = sample.xeger("([1-9]+)a+([1-9]+)c+([1-9]+)b+([1-9]+)d+([1-9]+) 0")
		neg_samples.append(sample)
	return pos_samples, neg_samples		

def saving_data(pos_samples, neg_samples, train_size=400, dev_size=0, test_size=100):
	data_size = [train_size, dev_size, test_size]
	files_names = ["train", "dev", "test"]
	if np.sum(data_size) > len(pos_samples) + len(neg_samples):
		raise Exception("Didn't get enough samples for dividing them to the wanted sizes")
	for k in range(len(data_size)):
		data_arr = []
		for i in range(data_size[k]):
			if len(pos_samples) > 0 and len(neg_samples) > 0:
				coin = random.randint(0, 1)
			elif len(pos_samples) > 0:
				coin = 1
			else:
				coin = 0
			if coin:
				sample = pos_samples.pop(0)
			else:
				sample = neg_samples.pop(0)
			data_arr.append(sample)
		if data_size[k] > 0:
			with open(files_names[k], mode="w", encoding="utf-8") as f:
				for samp in data_arr:
					f.write(samp + "\n")



if __name__=="__main__":
	pos_samples, neg_samples = generating_data(5000, 5000)
	print(pos_samples[:3])
	print(neg_samples[:3])
	saving_data(pos_samples, neg_samples, train_size=3000, test_size=1000)
