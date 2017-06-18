#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:40:47 2017

@author: SichengZhou
"""

import csv
import statistics
import pandas as pd
import sys
from collections import Counter
import numpy as np

#hospital_readmission = csv.reader('Hospital_Readmissions_Reduction_Program.csv')

csvfile =  open('totalcomplete.csv',newline = '') 
reader = csv.DictReader(csvfile)

#for row in reader:
#    print(row)





data = pd.read_csv('totalcomplete.csv')
count = 0
countnot = 0


print (data.head())
wd_order = Counter()
wd_notorder = Counter()
total_counts = Counter()

for i in range(521):
    if data.iloc[i]['iforder'] == 'prescribe/order':
        count += 1
        for word in data.iloc[i]['detail'].split(' '):
            if word.lower() in wd_order:
                wd_order[word.lower()] += 1
            if word.lower() not in wd_order:
                wd_order[word.lower()] = 1
            if word.lower() in total_counts:
                total_counts[word.lower()] += 1
            if word.lower() not in total_counts:
                total_counts[word.lower()] = 1
    elif data.iloc[i]['iforder'] == 'notprescribe/order':
        countnot += 1
        for word in data.iloc[i]['detail'].split(' '):
            if word.lower() in wd_notorder:
                wd_notorder[word.lower()] += 1
            if word.lower() not in wd_notorder:
                wd_notorder[word.lower()] = 1                
            if word.lower() in total_counts:
                total_counts[word.lower()] += 1
            if word.lower() not in total_counts:
                total_counts[word.lower()] = 1
                
#            
#for w in total_counts.most_common()[:200]:
#    print (w)
        
for i in range(521):
    if data.iloc[i]['iforder'] == 'notprescribe/order':
        countnot += 1
        for word in data.iloc[i]['detail'].split(' '):
            if word.lower() in wd_notorder:
                wd_notorder[word.lower()] += 1
            if word.lower() not in wd_notorder:
                wd_notorder[word.lower()] = 1
                
#for w in wd_notorder.most_common()[0:100]:
#    print (w)
#    
#for w in wd_order:
#    if w not in wd_notorder:
#        print (w)
  
        
order_not_ratios = Counter()
for term,cnt in list(total_counts.most_common()):
    if(cnt >= 1):
        order_not_ratio = wd_order[term] / float(wd_notorder[term]+1)
        order_not_ratios[term] = order_not_ratio
        
        
print  (order_not_ratios.most_common()[0:100])
        
for word,ratio in order_not_ratios.most_common():
    order_not_ratios[word] = np.log(ratio)
    
#print (list(reversed(order_not_ratios.most_common()))[0:30])
#print (countnot)
        
        #data.iloc[1]['detail']


#for row in data.iterrows():
#    print (row['detail'])
    
#    
#    
#    if row['iforder'] == 'prescribe/order':
#        count += 1
#        

#for i in data:
#    i.encode('utf-8').strip()
#    print (i)
    
#print (reader)

#print (data.head())



#
#for row in reader:
#    if (row['Measure Name'] == 'READM-30-HF-HRRP') \
#       and (row['State'] == 'CA'):
#           
#           try:        
#               ex_re_ratio.append(float(row['Excess Readmission Ratio']))
#               
#           except:
#               pass
    

feature_list = ['stage','alarm','allergy','allergic','barcode','been given','bin',
'bradycardic','calculate','chamber','changed','communicate','communicated',
'communication','connected to','continuous','different patient','discharge',
'discharge medication list','discharge orders','dispensed','dispensed as',
'dizziness','dizzy','dose was given','drew up','duplicates','dyspneic','emar',
'emr','expired medication','faint','found','hemorrhagic stroke','hives','hold off',
'hung','infusing','infusion rate','infusion setting','interpret','interprets',
'itching','label','lack of availability','leaking','mar','med rec','mix','mixed up', 
'nauseated','new order','normally','noted to','npo','omni','omnicell','on his tray',
'on the counter','order','ordered','ordered as','ordering','orders were ignored',
'overdose','override','pca pump','pharmacy','pick up','policy','prior to administration',
'programmed','programmed into','proper protocol','pump','pyxis','quickmar','ready',
'received','redness','redose','removed from','scanned','signed out','sitting on',
'slot','stock','storage','suddenly stopped','supposed to','supposed to receive',
'to be given','too early','transcribed','tube station','turned off','vomiting',
'vomitting','wasted','were ignored','were missing','were missed','continuous']
            
#print (len(feature_list))
#data_feature = data.loc[:,feature_list]
#allergic = data_feature['allergic']
#print (data_feature.head())
#print (data_try)
#print (data.iloc[0:1,:]) 
#print (allergic)
