#  import nltk
#  nltk.download('averaged_perceptron_tagger')
# 토큰화 : 텍스트를 토큰이라 불리는 단위로 나누는 작업

#문장 토큰화
from nltk.tokenize import sent_tokenize
text = "The most beautiful things in the world cannot be seen or touched. They are felt with the heart."
text.lower()
tokenized_sentences = sent_tokenize(text)
print(tokenized_sentences)

#단어 토큰화
from nltk.tokenize import word_tokenize
text = "The most beautiful things in the world cannot be seen or touched. They are felt with the heart."
tokenized_words = word_tokenize(text)
print(tokenized_words)


#품사태깅
from nltk.tag import pos_tag
tagged_words = pos_tag(tokenized_words)
print(tagged_words)