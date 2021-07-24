import random
import numpy as np
from xeger import Xeger

LIMIT_REGEX = 100


def generating_data(challenge_name, num_positive=500, num_negative=500):
    pos_samples, neg_samples = [], []
    for i in range(num_positive):
        if challenge_name == "Palindroms":
            sample = Xeger(limit=np.floor(LIMIT_REGEX / 2))
            sample = sample.xeger("([0-1]+)")
            sample += sample[::-1] + " 1"
            pos_samples.append(sample)
        if challenge_name == "start_end_same":
            sample = Xeger(limit=np.floor(LIMIT_REGEX - 2))
            sample = sample.xeger("([0-1]+)")
            coin = str(random.randint(0, 1))
            sample = coin + sample + coin + " 1"
            pos_samples.append(sample)
        if challenge_name == "primes":
            counter = 0
            num = 2
            while counter < num_positive:
                print(counter)
                num += 1
                flag = True
                for i in range(2, num):
                    if (num % i) == 0:
                        flag = False
                        break
                if flag:
                    counter += 1
                    pos_samples.append(bin(num)[2:] + " 1")
        else:
            raise Exception("There is no challenge called {}".format(challenge_name))
    for i in range(num_negative):
        if challenge_name == "Palindroms":
            sample = Xeger(limit=LIMIT_REGEX)
            sample = sample.xeger("([0-1]+)") + " 0"
            neg_samples.append(sample)
        if challenge_name == "start_end_same":
            sample = Xeger(limit=np.floor(LIMIT_REGEX - 2))
            sample = sample.xeger("([0-1]+)")
            coin = random.randint(0, 1)
            sample = str(coin) + sample + str(1 - coin) + " 0"
            neg_samples.append(sample)
        if challenge_name == "primes":
            counter = 0
            num = 2
            while counter < num_negative:
                num += 1
                for i in range(2, num):
                    if (num % i) == 0:
                        counter += 1
                        neg_samples.append(bin(num)[2:] + " 0")
                        break
        else:
            raise Exception("There is no challenge called {}".format(challenge_name))
    return pos_samples, neg_samples


def saving_data(pos_samples, neg_samples, challenge_name, train_size=400, dev_size=0, test_size=100):
    data_size = [train_size, dev_size, test_size]
    files_names = ["train_{}".format(challenge_name), "dev_{}".format(challenge_name), "test_{}".format(challenge_name)]
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


if __name__ == "__main__":
    pos_samples, neg_samples = generating_data("primes", 200, 200)
    saving_data(pos_samples, neg_samples, "primes", train_size=300, test_size=100)
