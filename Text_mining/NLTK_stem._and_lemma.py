
from nltk.stem import SnowballStemmer
from NLTK_Stopword import no_stopword
from nltk.stem import WordNetLemmatizer

#어간추출 : 단어의 어간을 추출하는 작업
ss = SnowballStemmer(language="english")
# for문 사용방식 2가지
# stemmed = []
# for word in no_stopword: stemmed.append(ss.stem(word))
stemmed = [ss.stem(word) for word in no_stopword]

print("추출 전:", no_stopword)
print("추출 후:", stemmed)


# 표제어 추출: 사전의 기본 단어인 표제어를 추출하는 작업
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in no_stopword]

print("추출 전:", no_stopword)
print("추출 후:", lemmatized)

