from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from Utils import merge_files
from Utils import evaluation
 




def visualize_word_embeddings(texts_original, texts_reconstructed,title=""):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Φτιάχνουμε embeddings
    embeddings_original = model.encode(texts_original)
    embeddings_reconstructed = model.encode(texts_reconstructed)

    all_emb = list(embeddings_original) + list(embeddings_reconstructed)
    pca = PCA(n_components=2)
    all_reduced = pca.fit_transform(all_emb)
    reduced_original = all_reduced[:len(embeddings_original)]
    reduced_reconstructed = all_reduced[len(embeddings_original):]
    # Plot
    plt.figure(figsize=(10,5)) 
    plt.subplot(1, 3, 1)
    plt.scatter(reduced_original[:,0], reduced_original[:,1], c='blue', label='Original')
    for i, word in enumerate(texts_original):
        plt.text(reduced_original[i,0], reduced_original[i,1], word, color='blue', fontsize=9)
    plt.legend()
    plt.title("Word Embeddings: Original")

    plt.subplot(1, 3, 2)
    plt.scatter(reduced_reconstructed[:,0], reduced_reconstructed[:,1], c='red', label='Reconstructed')
    for i, word in enumerate(texts_reconstructed):
        plt.text(reduced_reconstructed[i,0], reduced_reconstructed[i,1], word, color='red', fontsize=9)
    plt.legend()
    plt.title("Word Embeddings: Reconstructed")

    plt.subplot(1, 3, 3)
    plt.scatter(reduced_reconstructed[:,0], reduced_reconstructed[:,1], c='red', label='Reconstructed')
    for i, word in enumerate(texts_reconstructed):
        plt.text(reduced_reconstructed[i,0], reduced_reconstructed[i,1], word, color='red', fontsize=9)
    plt.scatter(reduced_original[:,0], reduced_original[:,1], c='blue', label='Original')
    for i, word in enumerate(texts_original):
        plt.text(reduced_original[i,0], reduced_original[i,1], word, color='blue', fontsize=9)
    plt.legend()
    plt.title("Word Embeddings: Reconstructed")

    plt.suptitle(title)
    plt.show()
    


# print("reading files .........")
scores=[]
labels=[]
_data=merge_files("data/reconstructionsA.json","data/reconstructionsB.json","data/reconstructionsAB.json")
for data in _data:
    print("---------------------------------------------ModelName: ",data['model_name'],"--------------------------------------------------")
    print("ModelName: ",data['model_name'])
    print("OriginalText: ",data['original_text'])
    print("Reconstruction: ",data['reconstruction'])
    score=evaluation(hypothesis=data['reconstruction'],premise=data['original_text']) #print indisde
    visualize_word_embeddings(data['original_text'].split(" "),data['reconstruction'].split(" "),"reconstructed with : "+data['model_name']+"with cosine Similarity : "+str(score))
    scores.append(score)
    labels.append(data['model_name'])

plt.figure(figsize=(12, 6))
bars=plt.bar(labels, scores)

for bar, score in zip(bars, scores):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01,
             f"{score:.2f}", ha="center", va="bottom", fontsize=9)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Cosine Similarity")
plt.title("Evaluation of Reconstructions (A + B merged)")
plt.tight_layout()
plt.show()

# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings1 = model.encode("καλημερα")
# print(embeddings1)