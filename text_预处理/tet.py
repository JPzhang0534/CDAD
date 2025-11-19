# import os
# import torch
#
# print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
# print("torch.cuda.device_count():", torch.cuda.device_count())
#
# for i in range(torch.cuda.device_count()):
#     print(f"Device {i}: {torch.cuda.get_device_name(i)}")

# stanza 方式
# import stanza
# nlp = stanza.Pipeline(lang='en')
# doc = nlp('She is a cute woman.')
# print(doc)

#spacy 方式
# import spacy
#
# nlp = spacy.load("en_core_web_sm")
# text = "a photo of a man in a red shirt is riding a black bicycle near the beach."
#
# doc = nlp(text)
#
# for token in doc:
#     print(f"{token.text:15} POS={token.pos_:10} DEP={token.dep_:10} TAG={token.tag_}")
#
# # 提取名词、形容词等
# nouns = [token.text for token in doc if token.pos_ == "NOUN"]  # 名词
# adjectives = [token.text for token in doc if token.pos_ == "ADJ"] # 属性
#
# print(nouns)
# print(adjectives)