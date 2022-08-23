#step1: pip install datasets transformers
# step2 : Load dataset
from datasets import load_dataset

ds = load_dataset('beans')
# print(ds)

#Step3 => look at the 400th image features
#notice image is a PIL, is a cache

ex = ds['train'][400]
# print(ex)

#STEP4 Look at the image 
image = ex['image']
# print(image)

#STEP5 Look at the labels

labels = ds['train'].features['labels']
print(labels)