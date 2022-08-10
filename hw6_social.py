"""
Social Media Analytics Project
Name:
Roll Number:
"""

import hw6_social_tests as test

project = "Social" # don't edit this

### PART 1 ###

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
endChars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]

'''
makeDataFrame(filename)
#3 [Check6-1]
Parameters: str
Returns: dataframe
'''
def makeDataFrame(filename):
    df=pd.read_csv(filename)
    return df
   


'''
parseName(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseName(fromString):
    start=fromString.find(":") + \
    len(":")
    end=fromString.find("(")
    return fromString[start:end].strip()


'''
parsePosition(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parsePosition(fromString):
    start=fromString.find("(")+\
        len(":")
    end=fromString.find("from")
    return fromString[start:end].strip()


'''
parseState(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseState(fromString):
    start=fromString.find("from")+\
        len("from")
    end=fromString.find(")")
    return fromString[start:end].strip()




'''
findHashtags(message)
#5 [Check6-1]
Parameters: str
Returns: list of strs
'''
def findHashtags(message):
    hash_words=[]
    for word in message.split("#")[1:]:
        hashtag=""
        for letter in word:
            if letter not in endChars:
                hashtag+=letter
            else:
                break
        hash_words.append("#"+hashtag)
    return hash_words


'''
getRegionFromState(stateDf, state)
#6 [Check6-1]
Parameters: dataframe ; str
Returns: str
'''
def getRegionFromState(stateDf, state):
    Region=stateDf.loc[stateDf['state']==state,"region"]
#   df.loc[df['known column name'] == 'known value to match', 'column name to return']  
    return Region.values[0]


'''
addColumns(data, stateDf)
#7 [Check6-1]
Parameters: dataframe ; dataframe
Returns: None
'''
def addColumns(data, stateDf):
    names=[]
    positions=[]
    states=[]
    regions=[]
    hashtags=[]
    for index,row in data.iterrows():
        l=row["label"]
        t=row["text"]
        name=parseName(l)
        position=parsePosition(l)
        state=parseState(l)
        region=getRegionFromState(stateDf,state)
        hashtag=findHashtags(t)
        names.append(name)
        positions.append(position)
        states.append(state)
        regions.append(region)
        hashtags.append(hashtag)
    data["name"]=names
    data["position"]=positions
    data["state"]=states
    data["region"]=regions
    data["hashtags"]=hashtags
    return None

### PART 2 ###

'''
findSentiment(classifier, message)
#1 [Check6-2]
Parameters: SentimentIntensityAnalyzer ; str
Returns: str
'''
def findSentiment(classifier, message):
    score = classifier.polarity_scores(message)['compound']
    if score< (-0.1):
        return "negative"
    elif score>(0.1):
        return "positive"
    else:
        return "neutral"

   


'''
addSentimentColumn(data)
#2 [Check6-2]
Parameters: dataframe
Returns: None
'''
def addSentimentColumn(data):
    classifier = SentimentIntensityAnalyzer()
    sentiments=[]
    for index,row in data.iterrows():
        message=row["text"]
        sentiment=findSentiment(classifier,message)
        sentiments.append(sentiment)
    data["sentiment"]=sentiments
    return None


'''
getDataCountByState(data, colName, dataToCount)
#3 [Check6-2]
Parameters: dataframe ; str ; str
Returns: dict mapping strs to ints
'''
def getDataCountByState(data, colName, dataToCount):
    dict={}
    for index,row in data.iterrows():
        if len(colName)!=0 and len(dataToCount)!=0:
            if row[colName]==dataToCount:
                if row["state"] in dict:
                    dict[row["state"]]+=1
                else:
                    dict[row["state"]]=1
        else:
            if  row["state"] in dict:
                    dict[row["state"]]+=1
            else:
                dict[row["state"]]=1
    
    
    return dict


'''
getDataForRegion(data, colName)
#4 [Check6-2]
Parameters: dataframe ; str
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def getDataForRegion(data, colName):
    outer_dictionary={}
    for index,row in data.iterrows():
        if row["region"] not in outer_dictionary:
            outer_dictionary[row["region"]]={}
        if row[colName] in outer_dictionary[row["region"]]:
            outer_dictionary[row["region"]][row[colName]]+=1
        else:
            outer_dictionary[row["region"]][row[colName]]=1


    return outer_dictionary


'''
getHashtagRates(data)
#5 [Check6-2]
Parameters: dataframe
Returns: dict mapping strs to ints
'''
def getHashtagRates(data):
    hashtag_dictionary={}
    for index,row in data.iterrows():
        for hashtag in row["hashtags"]:
            if hashtag in hashtag_dictionary:
               hashtag_dictionary[hashtag]+=1
            else:
               hashtag_dictionary[hashtag]=1

    return hashtag_dictionary


'''
mostCommonHashtags(hashtags, count)
#6 [Check6-2]
Parameters: dict mapping strs to ints ; int
Returns: dict mapping strs to ints
'''
def mostCommonHashtags(hashtags, count):
    hashtags_value={k:v for k,v in sorted(hashtags.items(),key=lambda v:v[1],reverse=True)}
    tophashtags_dictionary={}
    for i in hashtags_value:
        if len(tophashtags_dictionary)<count:
            tophashtags_dictionary[i]=hashtags_value[i]


    return tophashtags_dictionary


'''
getHashtagSentiment(data, hashtag)
#7 [Check6-2]
Parameters: dataframe ; str
Returns: float
'''
def getHashtagSentiment(data, hashtag):
    mes=0
    count=0
    for index,row in data.iterrows():
        if hashtag in findHashtags(row["text"]):
            if row["sentiment"]=="positive":
                count+=1
            elif row["sentiment"]=="negative":
                count -= 1
            elif row["sentiment"]=="neutral":
                count +=0
            mes+=1
    return count/mes


### PART 3 ###

'''
graphStateCounts(stateCounts, title)
#2 [Hw6]
Parameters: dict mapping strs to ints ; str
Returns: None
'''
def graphStateCounts(stateCounts, title):
    import matplotlib.pyplot as plt
    dict_list=list(stateCounts.items())
    for k,v in dict_list:
        labels=k
        yvalues=v
        plt.bar(labels,yvalues,color='yellow')
        plt.xlabel(title,loc='center')
        plt.xticks(rotation='vertical')
        plt.title(title)
    plt.show()
    return


'''
graphTopNStates(stateCounts, stateFeatureCounts, n, title)
#3 [Hw6]
Parameters: dict mapping strs to ints ; dict mapping strs to ints ; int ; str
Returns: None
'''
def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    return


'''
graphRegionComparison(regionDicts, title)
#4 [Hw6]
Parameters: dict mapping strs to (dicts mapping strs to ints) ; str
Returns: None
'''
def graphRegionComparison(regionDicts, title):
    return


'''
graphHashtagSentimentByFrequency(data)
#4 [Hw6]
Parameters: dataframe
Returns: None
'''
def graphHashtagSentimentByFrequency(data):
    return


#### PART 3 PROVIDED CODE ####
"""
Expects 3 lists - one of x labels, one of data labels, and one of data values - and a title.
You can use it to graph any number of datasets side-by-side to compare and contrast.
"""
def sideBySideBarPlots(xLabels, labelList, valueLists, title):
    import matplotlib.pyplot as plt

    w = 0.8 / len(labelList)  # the width of the bars
    xPositions = []
    for dataset in range(len(labelList)):
        xValues = []
        for i in range(len(xLabels)):
            xValues.append(i - 0.4 + w * (dataset + 0.5))
        xPositions.append(xValues)

    for index in range(len(valueLists)):
        plt.bar(xPositions[index], valueLists[index], width=w, label=labelList[index])

    plt.xticks(ticks=list(range(len(xLabels))), labels=xLabels, rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Expects that the y axis will be from -1 to 1. If you want a different y axis, change plt.ylim
"""
def scatterPlot(xValues, yValues, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xValues, yValues)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xValues[i], yValues[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.ylim(-1, 1)

    # a bit of advanced code to draw a line on y=0
    ax.plot([0, 1], [0.5, 0.5], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    test.week1Tests()
    print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    test.runWeek1()

    ## Uncomment these for Week 2 ##
    print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()
    
   

    ## Uncomment these for Week 3 ##
    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()
