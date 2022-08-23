from transformers import pipeline

generator = pipeline("text-generation")
gen = generator("In this course, we will teach you how to")

print(gen)

#Printed Text
#In this course, we will teach you how to properly use Google Forms
#  3.4 and use it to access Google forms and add, delete, edit or
#  delete items. 
# You will also use it to add, remove, add and delete items on the