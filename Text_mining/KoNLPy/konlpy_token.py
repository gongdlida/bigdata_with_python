# konlpy 한국어 자연어 처리를 위한 파이썬 패키지
# 한국어 토큰화
# 빈칸만으로 구별해서 토큰화를 할 수 없는 경우가 대부분, 형태소를 이해해야 함
# 뜻을 갖는 가장 작은 말의 단위인 형태소를 분석하기 위해서 Okt, Mecab등이 제공


from konlpy.tag import Okt
#토큰화
okt = Okt()
ktext = "열심히 코딩한 당신, 연휴에는 여행을 가봐요"
ktokens = okt.morphs(ktext)
print(ktokens)
#품사태깅
tagged_ktokens = okt.pos(ktext)
print(tagged_ktokens)

#불용어 제거
# 조사, 접속사 등을 제거하는 것만으로는 부족하며, 명사, 형용사 등에서도
# 제거하기 위해서는 불영어 사전이 필요

kstopwords=  ["가", "휴", "을", "에는", "아이고", "당신", "나", "우리", "저희"]
no_kstopword= [word for word in ktokens if not word in kstopwords]
print(no_kstopword)

# 한국어 어간 추출
# okt의 경우에는 morphs 함수에서 stem 패러미터를 true로 설정하면 되
print("before",ktokens)
ktokens= okt.morphs(ktext,stem=True)
print("after",ktokens) # "가봐요" => "가보다" 로 변경된 것을 확인할 수 있음

#한국어 표제어 추출은 제공하지 않기 때문에 사용할 수 없음