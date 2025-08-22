import re
from Utils import save_reconstructions_to_json as store

full_text1="""Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in
our lives. Hope you too, to enjoy it as my deepest wishes.
Thank your message to show our words to the doctor, as his next contract checking, to all of us.
I got this message to see the approved message. In fact, I have received the message from the
professor, to show me, this, a couple of days ago. I am very appreciated the full support of the
professor, for our Springer proceedings publication"""
text1="""Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes."""
text2="""During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor?"""


_data=[]
def recreate_text1_using_regex(text):
    # Use regex to find and replace specific patterns in the text
    text = re.sub(r"in our Chinese culture", "a special celebration in Chinese culture", text)
    text = re.sub(r"to celebrate it with all safe and great in our lives.", "to honor safety, health, and blessings in our lives.", text)
    text = re.sub(r" as my deepest wishes.", " ,sending you my deepest wishes!", text)
    return text



def recreate_text2_using_regex(text):
    # Use regex to find and replace specific patterns in the text
    text = re.sub(r"During our final discuss", "During our final discussion", text)
    text = re.sub(r", I told him about the new submission", ". I mentioned the new submission", text)
    text = re.sub(r"the one we were waiting since last autumn,", "the one we had been waiting for since last autumn.", text)
    text = re.sub(r"but the updates was confusing as it not included", "However, the updates were confusing, as they did not include", text)
    text = re.sub(r"the full feedback from reviewer or maybe editor?", "the full feedback from the reviewer, or perhaps the editor?", text)

    return text
result1 = recreate_text1_using_regex(text1)
print("Result for text1:\n")
print(result1,"end=\n")
result2 = recreate_text2_using_regex(text2)
print("Result for text2:\n")
print(result2)


_data.append({
            "model_name":"recreate_text1_using_regex id: 1",
            "original_text": text1,
            "reconstruction": result1
        })
_data.append({
            "model_name":"recreate_text2_using_regex id: 2",
            "original_text": text2,
            "reconstruction": result2
        })
store(data=_data,file_path="data/reconstructionsA.json")