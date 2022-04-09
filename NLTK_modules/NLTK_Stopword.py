# import nltk
# nltk.download('stopwords')

#불용어: 의미가 없는 단어들을 제거하는 작업

from nltk.corpus import stopwords
e_stopwords = stopwords.words("english")
print("불용어 개수:",len(e_stopwords))
print(e_stopwords[:179])

#불용어가 아닌 단어들 만을 포함할 리스트
no_stopword =[]
from NLTK_token import tokenized_words
for word in tokenized_words:
  if word not in e_stopwords: no_stopword.append(word)

  print("추출 전:", tokenized_words)
  print("추출 후:", no_stopword)