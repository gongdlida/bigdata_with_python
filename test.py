x = "hello world"
print(x)
x.split()

i = [1,2,3]
print(i[1])
print(list(range(1,10,1)))

c = 0
while True:
  if c%3 != 0:
    print(c) 
  
  elif c> 4:
    break
  
  c+=1

for i in range(1,10):
  if i%3 == 0:
    print(i)
  else:
    print(0)
