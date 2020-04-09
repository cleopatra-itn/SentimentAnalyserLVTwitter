import numpy as np
import pandas as pd

import sys

file_path = sys.argv[1] 

metric = {}
stastics = {}

with open(file_path) as input_file:
    current_epoch=None
    for line in input_file:
        line= line.strip()

        if line.find("Bert Model") >-1:
            stastics["Bert Model"] =line
        if line.find("Current date and time") >-1:
            stastics["Current date and time"] =line
        if line.find("Train file") >-1:
            stastics["Train file"] =line
        if line.find("Valid file") >-1:
            stastics["Valid file"] =line
        if line.find("Test file") >-1:
            stastics["Test file"] =line
        if line.find("Train size") >-1:
            stastics["Train size"] =line
        if line.find("Valid size") >-1:
            stastics["Valid size"] =line
        if line.find("Test size") >-1:
            stastics["Test size"] =line
        tokens = line.split()
        for token in tokens:
            if  token.find("epoch")==0:
                metric[token]=[]
                current_epoch=token
                continue
            if  token.find("train_loss")>-1:
                metric[current_epoch].append(token)
            if  token.find("val_loss")>-1:
                metric[current_epoch].append(token)
            if  token.find("test_loss")>-1:
                metric[current_epoch].append(token)
            if  token.find("train_acc")>-1:
                metric[current_epoch].append(token)
            if  token.find("val_acc")>-1:
                metric[current_epoch].append(token)
            if  token.find("test_acc")>-1:
                metric[current_epoch].append(token)
results =[]
for item in metric.items():
    result=[]
    result.append(item[0].replace('epoch=',""))
    for fig in item[1]:
        result.append(fig.split("=")[-1].replace(",",""))
    results.append(result)

for item in stastics.items():
    print(item[0],item[1].split()[-1])

#lets convert that to numpy array as np.array
num = np.array(results)

#now construct a beautiful table
df = pd.DataFrame(num, columns=["EPOCH","Trn loss","Val Acc" ,"Tst loss","Trn Acc","Val loss","Tst Acc"]) #
dash = 62
print("-"*dash)
print("| ".join(df.columns), "|")
for index,row in df.iterrows():
    print("-"*dash)
    print("|",row["EPOCH"]," |", row["Trn loss"]," |", row["Val loss"]," |",row["Tst loss"], " |", row["Trn Acc"]," |",row["Val Acc"],"  |",row["Tst Acc"]," |")
    
print("-"*dash)


# 