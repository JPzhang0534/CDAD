import stanza
# 加载英文模型
nlp = stanza.Pipeline('en', model_dir="/home/zjp/stanza_resources", processors='tokenize,pos,lemma,depparse',
                      download_method=None)


# 输入句子
sentence = "the women who is holding a tennis racket in red vest and white hat and short skirt"

# 分析句子
doc = nlp(sentence)

# 遍历句子的词元，寻找名词及其修饰词
for sent in doc.sentences:
    for word in sent.words:
        # 找到主名词，例如 table 或 chairs
        if word.upos == 'NOUN':
            noun = word.text
            modifiers = []

            # 查找修饰当前名词的形容词等修饰词
            for w in sent.words:
                if w.head == word.id and w.deprel in ['amod', 'compound']:
                    modifiers.append(w.text)

            # 按顺序输出（修饰词 + 名词）
            phrase = modifiers + [noun]
            print(' '.join(phrase))
