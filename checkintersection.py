import pandas as pd

df=pd.read_csv("labelattack.csv")
data = {}
for col in df.columns.values.tolist():
    data[col] = df[col].T.tolist()

def isnan(value):
  try:
      import math
      return math.isnan(float(value))
  except:
      return False
for key in data:
	for key1 in data:
		if key != key1:
			
			result = set(data[key]).intersection(set(data[key1]))
                        if len(list(result)) != 0:
				if isnan(list(result)[0]) is False:
                                        interlist = [str(x)  for x in list(result) if isnan(x) is False]
					#print "Tactics " + key + " | " + key1 + "| Number of common techniques: " + str(len(interlist))
					#print
					#print "Common techniques: " + ",".join(interlist )
					#print 
					#print

print " techniques by the tactics they belong to"
alldata = set()
for col in df.columns.values.tolist():
    alldata.update(df[col].T.tolist())

tech = {}
for k in alldata:
        if isnan(k) is False:
		for key in data:
			if k.strip() in str(data[key]):
		                
				if k.strip() in tech:
					tech[k.strip()].append(key)

				else:
					tech[k.strip()] = []
                                        tech[k.strip()].append(key) 
for j in tech:
	if len(tech[j]) > 1:
		print j, len(tech[j])
                print


