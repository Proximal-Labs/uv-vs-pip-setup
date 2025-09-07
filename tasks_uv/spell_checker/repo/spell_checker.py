from textblob import TextBlob    # importing textblob library


a = "contwol"	 # incorrect spelling
print("original text: "+str(a))     #printing original text

b = TextBlob(a)  #correcting the text

# prints the corrected spelling
print("corrected text: "+str(b.correct())) # corrected spelling