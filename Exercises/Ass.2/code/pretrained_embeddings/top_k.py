import numpy as np
import sklearn.metrics.pairwise as pairwise

wv = np.loadtxt("wordVectors")

vocabf = open('vocab', 'r', encoding='utf-8')
vocab = [line.strip() for line in vocabf.readlines()]

word_to_ix = {}
for word in vocab:
    if word not in word_to_ix:
        word_to_ix[word] = len(word_to_ix)

words = ['dog', 'england', 'john', 'explode', 'office']
for word in words:
    sim = {}
    u = wv[word_to_ix[word]]
    for vocab_word in vocab:
        if vocab_word != word:
            v = wv[word_to_ix[vocab_word]]
            computed_sim = 1 - pairwise.cosine_similarity(u.reshape(1, -1), v.reshape(1, -1))[0][0]
            sim[vocab_word] = computed_sim
    sim = {k: v for k, v in sorted(sim.items(), key=lambda item: item[1])}
    top_5 = list(sim)[:5]
    print(word + ':')
    for w in top_5:
        print('\t{} - {}'.format(w,sim[w]))