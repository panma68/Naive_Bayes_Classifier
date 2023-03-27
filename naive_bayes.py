import os
import math

PATH_FOR_VOCAB = "aclImdb\\imdb.vocab"
PATH_FOR_TRAINING_DATA = "aclImdb\\train\\labeledBow.feat"

#Vocabulary , returns a list with the words
def vocabulary():
    vocab = list()
    
    with open(PATH_FOR_VOCAB,encoding = "utf-8") as file:
        for line in file:
            #We remove the \n char ( [:-1] )
            vocab.append(line[:-1])
    return vocab

#Here we set the probabilities for each word , depending if the review
#has a positive or negative review.
def initialization(vocab):     
    
    #Length of our Vocabulary (we set it 1 so the probability is not 0)
    posVocab=[1 for x in range(len(vocab))]
    negVocab=[1 for x in range(len(vocab))]

    totalSumOfNegativeWords=0 #for indexing negative words
    totalSumOfPositiveWords=0 #for indexing positive words
    
    """
    Using the labeledBow to get the count of words labeledBow.feat file is an
    already-tokenized bag of words (BoW) features that were used in our experiments.
    
    The .feat file is in LIBSVM format, an ascii sparse-vector format for labeled data.  
    
    The feature indices in these files start from 0, and the text
    tokens corresponding to a feature index is found in [imdb.vocab]. So a
    line with 0:7 in a .feat file means the first word (in [imdb.vocab] or vocab[0] variable in main)
    (the) appears 7 times in that review.
    
    For each line , the first number is the rating of a review.
    """  
    
    with open(PATH_FOR_TRAINING_DATA,encoding = "utf-8") as file:
        for review in file:
            review = review.split(" ",1) #LIBSVM format
            rating = review[0]  #the rating of review
            rating = int(rating)
            
            listOfArgs = review[1].split()
            
            for word in listOfArgs:
                indexing = word.split(":")
                wordIndexInVocab = int(indexing[0])
                wordSumInCurrentReview = int(indexing[1])

                if(rating <=4):
                    negVocab[wordIndexInVocab] = negVocab[wordIndexInVocab] + wordSumInCurrentReview
                    totalSumOfNegativeWords += wordSumInCurrentReview

                elif(rating >=7):
                    posVocab[wordIndexInVocab] = posVocab[wordIndexInVocab] + wordSumInCurrentReview
                    totalSumOfPositiveWords += wordSumInCurrentReview        

        #returning probability arrays
        posProbabilities=[int(x)/(totalSumOfPositiveWords+len(vocab)) for x in posVocab]
        negProbabilities=[int(x)/(totalSumOfNegativeWords+len(vocab)) for x in negVocab]
        
        return posProbabilities,negProbabilities

#For not letting our probability reach 0 (due to number representation constraints)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


#M is the most common words in each posibilities list.
#data is the train or test data 
def calculations(M,data,posProbabilities,negProbabilities):
    print(f"\nStart calculations for {data} data , ignoring the {M} most common words..")

    #Initializing numberOfReview
    numberOfReview = 0

    #Initializing success rate
    posEstimations = 0

    #Selecting the most common range amongst M words
    #Here M is the number inside my func
    tempPos = [x for x in posProbabilities]
    tempNeg = [x for x in negProbabilities]
    
    tempPos.sort()
    tempNeg.sort()
    
    #Here we don't consider the M most common words.
    tempPos = tempPos[-M:]
    tempNeg = tempNeg[-M:]

    #Here we set tempPos and tempNeg == to their first element
    #If the word probability is greater than the tempPos or tempNeg
    #probability , it means the chosen word prob/ty is within the M most common words
    tempPos = tempPos[0]
    tempNeg = tempNeg[0]

    filePath = f"{os.getcwd()}/aclImdb//{data}//labeledBow.feat"
    with open(filePath,"r",encoding = "utf-8") as file:
        for review in file:
            review = review.split(" ",1)

            #Rating of the review
            rating = int(review[0])
            review = review[1].split()  #contains the word indexes and occurances 

            #Here we will save the probabilities
            #We initialize them to positive/negative ReviewProbability
            #0.5 is the probability of the review being positive/negative

            positiveResult = 0.5
            negativeResult = 0.5

            for words in review:

                indexing = words.split(":")
                ind1 = int(indexing[0])
                ind2 = int(indexing[1])
                #Calculating if review is positive or negative
                
                if (posProbabilities[ind1] < tempPos):
                #Here if we want to calculate the times the word is present in the
                #current testing review , we simply do
                #posProbabilities[ind1]**ind2 instead of posProbabilities[ind1]
                #same with the negativeResult
                    temp = positiveResult
                    positiveResult = positiveResult * posProbabilities[ind1]
                    
                    if(positiveResult == 0.0):
                        temp = sigmoid(temp)
                        positiveResult = temp
                        positiveResult = positiveResult * posProbabilities[ind1]
                        negativeResult = sigmoid(negativeResult)
                        
                if (negProbabilities[ind1] < tempNeg):
                    temp = negativeResult
                    negativeResult = negativeResult * negProbabilities[ind1]
                    
                    if(negativeResult == 0.0):
                        temp = sigmoid(temp)
                        negativeResult = temp
                        negativeResult = negativeResult * negProbabilities[ind1]
                        positiveResult = sigmoid(positiveResult)
            
            #Calculating success rate
            if(positiveResult > negativeResult):
                numberOfReview += 1
                if(rating >= 7):
                    posEstimations += 1
        
            else:
                numberOfReview += 1
                if(rating <= 4):
                    posEstimations += 1
                    
            
                    
    
    #Final Results
    successRate = posEstimations/numberOfReview
    successRate = round(successRate , 3)
    successRate = successRate * 100
    
    return "Naive Bayes success rate is: " + str(successRate) +"%"
    
    
def main():         
    #For vocab , first value represents the POSITIVE occurances, and the second the NEGATIVE occurances. 
    vocab = vocabulary()
    posProbabilities , negProbabilities = initialization(vocab)
    
    print(calculations(2000,"train",posProbabilities,negProbabilities))
    print(calculations(1000,"train",posProbabilities,negProbabilities))
    print(calculations(500,"train",posProbabilities,negProbabilities))
    print(calculations(50,"train",posProbabilities,negProbabilities))
    
    print(calculations(2000,"test",posProbabilities,negProbabilities))
    print(calculations(1000,"test",posProbabilities,negProbabilities))
    print(calculations(500,"test",posProbabilities,negProbabilities))
    print(calculations(50,"test",posProbabilities,negProbabilities))


main()


