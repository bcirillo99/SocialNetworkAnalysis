
sigma = 1 
mu = 5
rewards=[]
sigma1=[]
for reward in range(1,10):
  n=reward
  rewards.append(reward)
  sigma = (sigma * (n - 2) + (reward - mu) ** 2) / (n - 1) if n > 1 else 1
  print(sigma)
  s = 0
  for r in rewards:
    s+=(r-mu)**2 if n > 1 else 1
  sigma1.append(s/n-1 )
print(sigma1)


p =[20.0,24.0,28.0,30.0,32.0,36.0]
n=1
mu=sum(p)/len(p)
for reward in p:
  sigma = (sigma * (n - 2) + (reward - mu) ** 2) / (n - 1) if n > 1 else 1
  n+=1
#u=E(p)
u=sum(p)/len(p)
# X-u
x_u=[x-u for x in p]
# (X-u)^2
x_u2=[x**2 for x in x_u]
# E[(X-u)^2]
print(sigma)
print(sum(x_u2)/(len(x_u2)-1))