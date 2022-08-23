from transformers import LevitFeatureExtractor, LevitForImageClassificationWithTeacher
from PIL import Image
from torchvision import transforms

import requests

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

image = Image.open('images/d.jpg')
convert_tensor = transforms.ToTensor()

convert_tensor(image)

feature_extractor = LevitFeatureExtractor.from_pretrained('facebook/levit-128S')
model = LevitForImageClassificationWithTeacher.from_pretrained('facebook/levit-128S')

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
