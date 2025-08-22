from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from Utils import save_reconstructions_to_json as store
import nltk

# Αν δεν έχεις το nltk punkt
nltk.download('punkt')

def paraphrase_sentence_by_sentence(paragraph,tok,mdl):
    sentences = nltk.sent_tokenize(paragraph)
    
    paraphrased_sentences = []
    
    for sentence in sentences:
        tokens = tok(sentence, truncation=True, padding='longest', return_tensors="pt")
        summary_ids = mdl.generate(**tokens, num_beams=10, max_length=512)
        paraphrased_sentence = tok.decode(summary_ids[0], skip_special_tokens=True)
        paraphrased_sentences.append(paraphrased_sentence)
    
    return " ".join(paraphrased_sentences)

texts = [
"""Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in
our lives. Hope you too, to enjoy it as my deepest wishes.
Thank your message to show our words to the doctor, as his next contract checking, to all of us.
I got this message to see the approved message. In fact, I have received the message from the
professor, to show me, this, a couple of days ago. I am very appreciated the full support of the
professor, for our Springer proceedings publication""",
"""During our final discuss, I told him about the new submission — the one we were waiting since
last autumn, but the updates was confusing as it not included the full feedback from reviewer or
maybe editor?
Anyway, I believe the team, although bit delay and less communication at recent days, they really
tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance
and efforts until the Springer link came finally last week, I think.
Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before
he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so.
Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future
targets”
"""]

_data=[]





#############      PEGASUS MODEL ################
model_name_pegasus = "tuner007/pegasus_paraphrase"
tokenizer_pegasus = PegasusTokenizer.from_pretrained(model_name_pegasus)
model_pegasus = PegasusForConditionalGeneration.from_pretrained(model_name_pegasus)
for i,text in enumerate(texts):
    result_pegasus = paraphrase_sentence_by_sentence(text,tokenizer_pegasus,model_pegasus)
    _data.append({
            "model_name":model_name_pegasus+"id: "+str(i+1),
            "original_text": text,
            "reconstruction": result_pegasus
        })    
    print(result_pegasus)
    print("\n---\n")




from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name_t5 = "humarin/chatgpt_paraphraser_on_T5_base"
tokenizer_t5 = AutoTokenizer.from_pretrained(model_name_t5)
model_t5 = AutoModelForSeq2SeqLM.from_pretrained(model_name_t5)
for i,text in enumerate(texts):
    result_t5 = paraphrase_sentence_by_sentence(text,tokenizer_t5,model_t5)
    _data.append({
            "model_name":model_name_t5+"id: "+str(i+1),
            "original_text": text,
            "reconstruction": result_t5
        })
    print(result_t5)
    print("\n---\n")



from transformers import BartForConditionalGeneration, BartTokenizer

model_name_bart = "facebook/bart-large-cnn"
tokenizer_bart = BartTokenizer.from_pretrained(model_name_bart)
model_bart = BartForConditionalGeneration.from_pretrained(model_name_bart)
for i,text in enumerate(texts):
    result_bart = paraphrase_sentence_by_sentence(text,tokenizer_bart,model_bart)
    _data.append({
            "model_name":model_name_bart+"id: "+str(i+1),
            "original_text": text,
            "reconstruction": result_bart
        })
    print(result_bart)
    print("\n---\n")

store(data=_data,file_path="data/reconstructionsB.json")

    
# print("all reconstructions are store in reconstructions.json please run ergasia1_C to evaluate")

# print("evaluating ....................")
# print(evaluation(explanation=texts[0],hypothesis=result_pegasus))
# print(evaluation(explanation=texts[0],hypothesis=result_t5))
# print(evaluation(explanation=texts[0],hypothesis=result_bart))
