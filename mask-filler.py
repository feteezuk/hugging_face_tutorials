from transformers import pipeline

#fills in the blank
#top_k argument 
#controls how many possibilities you want to be displayed.
unmasker = pipeline("fill-mask")
res = unmasker("This course will teach you all about <mask> models.", top_k=2)
print(res)