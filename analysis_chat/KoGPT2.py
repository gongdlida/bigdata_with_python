from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import torch # a package for GPU acceleration
U_TKN = "<usr>"
S_TKN = "<sys>"
EOS = "</s>"
MASK = "<unused0>"
SENT = "<unused1>"
PAD = "<pad>"
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
eos_token = EOS, unk_token ="<unk>",pad_token=PAD,mask_token=MASK
)
model = GPT2LMHeadModel.from_pretrained("eunjin/kogpt2-finetuned-wellness")

with torch.no_grad(): # 훈련에서 주로 사용하는 그레디먼트 계산을 사용하지 않음
  qs=[]
  while True:
    q = input("나> ").strip() 
    if q == "quit": 
      break 
    qs.append(q)

    userQuery = U_TKN + q + SENT
    encoded = tokenizer.encode(userQuery)
    inputIDs = torch.LongTensor(encoded).unsqueeze(dim=0)
    output = model.generate(inputIDs,max_length=50, num_beams=10, do_sample=False,
                            top_k=50, no_repeat_ngram_size=2, temperature=0.85) 
    allResponse=tokenizer.decode(output[0])
    idx = torch.where(output[0]==tokenizer.encode(S_TKN)[0])
    systemResponse = tokenizer.decode(output[0][int(idx[0])+1:], skip_special_tokens=True) 
    if '답변' in allResponse: # 긍정답변, 부정답변
      userQuery = U_TKN + ''.join(qs[-2:]) + SENT # 직전의 질문들을 추가함
      encoded = tokenizer.encode(userQuery)
      inputIDs = torch.LongTensor(encoded).unsqueeze(dim=0)
      output = model.generate(inputIDs,max_length=50, num_beams=10, do_sample=False, top_k=50, no_repeat_ngram_size=2, temperature=0.85) 
      #a_new = tokenizer.decode(output[0], skip_special_tokens=True)
      idx = torch.where(output[0]==tokenizer.encode(S_TKN)[0])
      systemResponse= tokenizer.decode(output[0][int(idx[0])+1:], skip_special_tokens=True) 
    print(systemResponse.strip())
