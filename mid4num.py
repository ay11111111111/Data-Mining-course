import numpy as np

A = np . array ([[[1 ,2 ,3] ,[4 ,5 ,6] ,[7 ,8 ,9] ,[10 ,11 ,12]]])

print(A.shape)
print ( A[0 ,: ,:])
print()
print ( A [: ,1:4 ,:]) # Line C
print()
print ( A [0 , : ,:-1])


desired = np.array ([2 ,1 ,0 ,0 ,1 ,1])
predicted = np.array ([2 ,2 ,2 ,1 ,1 ,1])


tf = 0
tp = 0
for i in range(len(predicted)):
	if predicted[i] == desired[i] and desired[i] != 0:
		tp += 1
	elif predicted[i] != desired[i] and desired[i] !=0:
		tf += 1	
print(tp)
print(tf)
