import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

store_data = pd.read_csv('C:\\Users\\ZDani\\Downloads\\prod.csv', header=None)  #keeping header as None
#print(store_data)
#store_data = pd.read_csv('store_data.csv', header=None)  #keeping header as None
#support - как часто встречается
#confidence - как часто срабатывается правило
#lift - поддержка ( независимость)
num_records = len(store_data)
print(num_records)
records = []

for i in range(0,num_records):
    itemInRecords = []
    for j in range(len(store_data.values[i])):
        if( str(store_data.values[i,j]) !='nan'):
           itemInRecords.append(str(int(store_data.values[i,j])))
    records.append(itemInRecords)


association_rules = apriori(records, min_support=0.1, min_confidence=0.01, min_lift=0, min_length=0)
association_results = list(association_rules)


print(len(association_results))  #to check the Total Number of Rules mined
#section = str(association_results).split(')]),')
#for i in section:
#    print(i)  # to print the first item the association_rules list to see the first rule


number=1
for item in range(0, len(association_results)):
    current=str(association_results[item]).split('=')
    #Напоминаю, что у нас были пропуски, которые при преобразовании стали текстовым значением "nan". Пропустим наборы, где они участвуют, т.к. алгоритм подсчитал nan как одно из наименований:
    if not "nan" in current[1]:
        #print("|Поддержка:"+str(current[2])[:4])
        #  #Для множества с мощностью 1:
        #if (current[1].count("'"))==2:
        #    print("|Достоверность:"+str(round(float((current[6])[:-6])*100,2))+"%")
        #    print("Лифт:" + (current[7])[:-3])
        #Для множества с мощностью 2:
        if(current[1].count("'"))==4:
            print(f"-----------------Набор № {number} -----------------"[:50])
            print("|Набор:" + (current[1])[11:-11])
            print("|Достоверность прямого правила:"+str(round(float((current[6])[:-6])*100,2))+"%")
            print("|Достоверность обратного правила:"+str(round(float((current[10])[:-6])*100,2))+"%")
            print("Лифт:" + (current[7])[:-30])
            print("Лифт:" + (current[11])[:-30])
            print("----------------------------------------")
        #Для множества с мощностью 3:
        #if (current[1].count("'"))==6:
            #    print("|Достоверность для двух последних к первому:"+str(round(float((current[6])[:-6])*100,2))+"%")
            #print("|Достоверность первых двух к последнему:"+str(round(float((current[10])[:-6])*100,2))+"%")
            #     print("|Достоверность двух крайних к среднему:"+str(round(float((current[14])[:-6])*100,2))+"%")
            #print("Лифт:" + (current[7])[:-30])
            #     print("Лифт:" + (current[11])[:-30])
            #      print("Лифт:" + (current[15])[:-30])
        #     print("Лифт:" + (current[19])[:-30])
    number=number+1

print("-----КОНЕЦ---РАБОТЫ-----")

