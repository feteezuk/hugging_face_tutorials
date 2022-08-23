from transformers import pipeline

classifier = pipeline("sentiment-analysis")
res = classifier("Lick my Stinky Ballsack")

print(res)