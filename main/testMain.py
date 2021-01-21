import test as ef
import extract_features as pf
from pathlib import Path
import os
import glob

sentiment = 0
count = 0
countImg = 0
valance = 0
positive = 0
negative = 0
dict = {}
imageA = []
imgA = []

for img in glob.glob("images/yeet.jpg"):
    imgA.append(img)
    path_file = os.path.basename(img).replace(".jpg", "")
    imageA.append(path_file)
    Path("cropped_faces/" + path_file).mkdir(parents=True, exist_ok=True)
    countImg = ef.crop_faces(img)
    count = 0

for x in range(len(imageA)):
    Path("Data4/" + imageA[x]).mkdir(parents=True, exist_ok=True)
    path_to = "Data4/" + imageA[x]
    for img_a in glob.glob("cropped_faces/"+imageA[x]+"/*.jpg"):
        count = ef.data(imgA[x], img_a, count)
    valance = ef.predict(path_to)
    valance = (valance*33.5)+33.5
    # color = pf.extract_colors(imgA[x])*10
    # print(color)
    # valance = valance + color
    if valance > 50:
        positive += 1
    else:
        negative += 1
    dict[imgA[x]] = valance
    sentiment += valance
    print(valance)

print(dict)
print(positive)
print(negative)
print(sentiment/(positive+negative))
