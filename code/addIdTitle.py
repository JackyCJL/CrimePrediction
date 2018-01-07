import pandas as pd

predicted = pd.read_csv(r'result.csv')

'''
id = []
for i in range(0,len(predicted)):
    id.append([i])
'''

cnt = 0
for line in predicted:
    print(line)
    break
    line.insert(0,cnt)
    cnt = cnt + 1

#result= pd.concat([id,predicted], axis=1)
#print(result)
print(predicted)