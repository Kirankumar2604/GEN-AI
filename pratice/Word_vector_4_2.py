import gensim.downloader as api
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
 
model = api.load('word2vec-google-news-300')
words = ['king', 'man', 'woman', 'queen']
vectors = [model[word] for word in words]

king = model['king']
man = model['man']
woman = model['woman']
queen = king - man + woman

similar_words = model.most_similar(queen, topn=5)
print("Most similar words to 'queen' (result of king - man + woman):")
for word, similarity in similar_words:
   print(f"{word}: {similarity}")
 
pca = PCA(n_components=2)
result = pca.fit_transform(vectors)

plt.figure(figsize=(6, 6))
plt.scatter(result[:, 0], result[:, 1])

for i, word in enumerate(words):
   plt.text(result[i, 0] + 0.05, result[i, 1] + 0.05, word, fontsize=12)
    
plt.title("2D Visualization of Word Relationships (king, man, woman, queen)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()