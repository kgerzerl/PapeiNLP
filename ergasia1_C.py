import matplotlib.pyplot as plt
from Utils import merge_files
from Utils import evaluation






# print(evaluation(hypothesis=resultGPT2[0]["generated_text"],explanation=text1))

scores=[]
labels=[]
print("reading files .........")
_data=merge_files("data/reconstructionsA.json","data/reconstructionsB.json","data/reconstructionsAB.json")
for data in _data:
    print("---------------------------------------------ModelName: ",data['model_name'],"--------------------------------------------------")
    print("ModelName: ",data['model_name'])
    print("OriginalText: ",data['original_text'])
    print("Reconstruction: ",data['reconstruction'])
    score=evaluation(hypothesis=data['reconstruction'],premise=data['original_text']) #print indisde
    scores.append(score)
    labels.append(data['model_name'])
    # print(score)


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

