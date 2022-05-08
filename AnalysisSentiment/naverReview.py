#데이터 내려받기
from urllib.request import urlretrieve
reviewDataUrl ="https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt"
urlretrieve(reviewDataUrl, filename  ="ratings_total.txt")

#데이터 로딩
import pandas as pd
review_df = pd.read_csv("./SentimentAnalysis/data/ratings_total.txt",sep="\t",names=["rating","review"])
review_df.to_csv("./SentimentAnalysis/data/1.review_data.csv",sep="\t",index=False)

#pandas 패키지
# 파이썬에서 대중적으로 사용되는 데이터 분석 및 처리용 패키지
# csv(Command Seperated Values)나 xls(eXceL Spreadsheet)등 다양한 형식의 데이터 파일을 읽고 저장할 수 있음
# 테이블 형식의 데이터를 DataFrame 객체로 표현, 정제, 변환, 탐색할 수 있음

customers = pd.DataFrame()
customers["id"] = ["Kim","Lee","Park","Choi"]
customers["visit"] = [27,129,36,14]
customers["mileage"] = [101.5, 300.0, 23.4, 59.1]

print(customers)

customers.to_csv("./SentimentAnalysis/data/comma_customers.csv",index=False)
customers.to_csv("./SentimentAnalysis/data/tab_customers.csv",sep="\t",index=False)

# 전처리 Preprocessing
# 데이터를 모형개발에 사용하기 전에, 입력 형식을 맞추거나, 처리속도와 품질
# 향상 등을 위해서 데이터를 조작하는 모든 작업을 지칭

# 데이터의 규모와 내용 확인
print(len(review_df))
print(review_df.shape)
print(review_df.head())

# 칼럼별로 값의 종류와 발생 빈도 확인
# 리뷰 등급별 카운팅
print(review_df["rating"].value_counts())

#차트로 보기

import matplotlib.pyplot as plt
review_df["rating"].value_counts().plot(kind="bar")
plt.show()


## 값이 없는 결측치가 존재하는지 확인
# 결측치가 있으면 모형을 학습할 수 없는 경우가 대부분임
# 결측치가 있으면 0이나 열의 평균값 같이 이를 대체할 수 있는 구체적인 값을 입력하거나, 결측치가 잇는 해당 행 또는 열을 제거
# 보통 행을 제거하며, 열에 많은 결측치가 있을 경에만 열을 제거
  # 결측치 -> 데이터의 값을 말함
# 결측치가 있는지 없는지 판별 
print(review_df.isna()) # isna => na는 not available의 약어
print(review_df.isna().sum())
print(review_df.isna().sum().sum())

## 데이터 정제하기
# 리뷰내용에서 한글과 공백을 제외하고 모두 제거
review_df = pd.read_csv("./SentimentAnalysis/data/ratings_total.txt",sep="\t",names=["rating","review"])
review_df["review"] = review_df["review"].str.replace("[^ㄱ-ㅎ|ㅏ-ㅣ|가-힣]","",regex=True)
review_df.to_csv("./SentimentAnalysis/data/2.onlyKorean.csv",sep="\t",index=False)
# 56번의 처리가 잘되었는지 확인
import numpy as np 
review_df["review"] = review_df["review"].replace("",np.nan) # nan -> not a number
print(review_df.isna().sum())

# 리뷰내용에서 한글 오류 수정
review_df["review"] = review_df["review"].str.replace("배공","배송")
review_df["review"] = review_df["review"].str.replace("보폴","보풀")
review_df.to_csv("./SentimentAnalysis/data/3.Edittypo.csv",sep="\t",index=False)
# def findStr(str):
#   if str == "배공": 
#     return True

#     return False

# filter(findStr,review_df["review"])

# 토큰화/불용어 제거
from konlpy.tag import Okt
okt = Okt()

x =[]
stopwords = ["도", "는", "다", "의", "가", "이", "은", "한", "에", "하", "고", "을","를", "인", "듯", "과", "와", "네", "들", "듯", "지", "임", "게"]

# for i,r in enumerate(review_df["review"]):
#   tokenized = okt.morphs(r,stem=True)
#   tokenized_with_us = [word for word in tokenized if not word in stopwords]
#   x.append(tokenized_with_us)
  
#   if i % 50000 == 0: print(i)

for indexOfReview, review in enumerate(review_df["review"]): 
  tokenized = okt.morphs(review,stem=True) # 토큰화
  tokenized_with_us = [word for word in tokenized if not word in stopwords] # 불용어 제거
  x.append(tokenized_with_us)
  
  if indexOfReview % 50000 == 0: print(indexOfReview)

review_df["tokenized"] = x
del review_df["review"]
review_df.to_csv("./SentimentAnalysis/data/4.tokenizedDataWithlabel.csv",sep="\t",index=False)
# 감독학습으로 감성분석을 하기 위한 레이블 설정
# 3이상이면 1 3점미만 0 -> 선호도 분석
review_df["label"] = [1 if x > 3 else 0 for x in review_df["rating"]]
del review_df["rating"] # rating을 label로 대체
print(review_df.head())
review_df.to_csv("./SentimentAnalysis/data/dataWithLabel.csv",sep="\t",index=False)

# scikit-learn 패키지
# 기계학습을 위한 효율적인 패키지
# 감독학습과 비감독 학습을 망라한 다양한 알고리즘들과 함께, 모형 개발, 모형 선택, 전처리 등을 위한 보조적인 함수들도 다수제공

import sklearn.naive_bayes
clf = sklearn.naive_bayes.MultinomialNB()

## 모형 개발과 검증
# 모형 개발을 위해 데이터집합을 훈련 데이터집합과 검증 데이터집합을 분할
# 전자를 이용해서 모형을 구축한 후, 후자를 이용해서 검증을 해야 함

# 사례 scikit-learn 패키지의 Model_selection 모듈의 train_test_split 함수를
# 이용하여 데이터 집합의 25%를 검증용으로, 나머지를 훈련용으로 분할
from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(review_df,test_size=0.25,random_state=42)
print("훈련용 리뷰의 수:", len(train_df))
print("검증용 리뷰의 수:", len(valid_df))

# 입력 형태 반환
#  -> 컴퓨터는 숫자만을 처리할 수 있으므로, 문서의 단어들을 숫자의 형태로 변환
#  -> 컴퓨터는 크게 두 가지지 방법이 있음
#     1. 문서의 단어 출현 빈도를 나타내는 카운트 벡터로 변환하는 방법, 그리고 이것을 개선한 Tfidf 벡터로 변환하는 방법
#     2. 전체 문서에 대한 단어 집합을 구하고, 각각의 단어에 하나의 정수를 부여하여, 각 문서를 이 정수들로
#        표현하되, 최대 크기를 정하여 이를 넘으면 제거, 미달한 경우에는 가상의 단어 PAD를 지칭하는 0으로 채워 줌

# 카운트 백터
# Column에 단어, Row에 문서를 각각 배치하여 테이블을 만들고, 각 교차지점에 해당단어가 해당문서에 출현한 횟수를 기입한 벡터
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = lambda x:x)
vector = vectorizer.fit_transform(train_df["tokenized"]).toarray()

# 출현 문서 수가 너무 작은 경우와 출현 문서수가 너무 큰 경우에는 분석에 도움이 안되기 때문에 제거

vectorizer = CountVectorizer(analyzer = lambda x:x, min_df=5,max_df=0.8)# min=5개 문서이상 출현, max=전체 문서들의 80%이하에서 출현해야함
train_vector = vectorizer.fit_transform(train_df["tokenized"]).toarray()
print(train_vector.shape) # (150000,10664)

valid_vector = vectorizer.transform(valid_df["tokenized"]).toarray()
print(valid_vector.shape) #  (50000,10664)

# Tfidf 벡터
# 각 문서에서 각 단아가 출현한 횟수 TF(Term Frequency)에 IDF(Inverse Document Frequency)를 곱해서 구한 벡터
# IDF - log(전체 문서 수 / (1+해당 단어가 출현한 문서 수) *단 log의 밑은 10이나 상수 e를 사용할 수 있음
# 한 단어에 대한 IDF 값은 모든 문서에 대하여 동일
# 사이킷런 패키지를 이용한 Tfidf 벡터 만들기
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer=lambda x:x, min_df=5, max_df=0.8)
train_df = vectorizer.fit_transform(train_df["tokenized"]).toarray()
valid_df = vectorizer.fit_transform(valid_df["tokenized"]).toarray()


# Tfidf 벡터
# 나이브 베이즈 모형 개발과 검증
# 일반적으로 Tfidf 벡터를 이용하는 것이 카운트 벡터를 사용하는 것 보다 좋은 결과를 보여주지만 항상 그런 것은 아니다.
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(train_vector,train_df["label"])
pred = clf.predict(valid_vector)
correct = sum(valid_df["label"] == pred)
print(correct)

accuracy = correct/len(valid_df)
print(accuracy)
