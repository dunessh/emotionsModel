from __future__ import absolute_import
import os
import cv2
import colorgram
import numpy as np
from fer import FER

# def extract_face(image_a):
#     detector = FER(mtcnn=True) # or with mtcnn=False for Haar Cascade Classifier
#     image = cv2.imread(image_a)
#     try:
#         result = detector.detect_emotions(image)
#         for x in range(len(result)):
#             bounding_box = result[x]["box"]
#             emotions = result[x]["emotions"]
#             cv2.rectangle(
#                 image,
#                 (bounding_box[0], bounding_box[1]),
#                 (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
#                 (0, 155, 255),
#                 2,
#             )
#             for idx, (emotion, score) in enumerate(emotions.items()):
#                 color = (211, 211, 211) if score < 0.01 else (0, 255, 0)
#                 emotion_score = "{}: {}".format(
#                     emotion, "{:.2f}".format(score) if score > 0.01 else ""
#                 )
#                 cv2.putText(
#                     image,
#                     emotion_score,
#                     (bounding_box[0], bounding_box[1] + bounding_box[3] + 30 + idx * 15),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     color,
#                     1,
#                     cv2.LINE_AA,
#                 )
#         if len(result) > 0:
#             image_name = os.path.basename(image_a)
#             path = 'C:/Users/User/PycharmProjects/FYP/main/drawn_images'
#             cv2.imwrite(os.path.join(path, image_name + '_drawn.jpg'), image)
#
#             negative = 0
#             positive = 0
#             neutral = 0
#
#             for y in range(len(result)):
#                 Tv = result[y]["emotions"]
#                 maximum = max(Tv, key=Tv.get)
#                 if maximum == "angry" or maximum == "disgust" or maximum == "fear" or maximum == "sad":
#                     negative += 1
#                 elif maximum == "happy" or maximum == "surprise":
#                     positive += 1
#                 else:
#                     neutral += 1
#
#                 if positive > negative:
#                    return "Image is positive"
#                 elif negative > positive:
#                     return "Image is negative"
#                 else:
#                     return "Image is neutral"
#         else:
#             return None
#     except:
#         pass
    # Result is an array with all the bounding boxes detected. We know that for 'justin.jpg' there is only one.

def hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100

    valence = (0.69*v)+(0.22*s)

    return valence

def extract_colors(image_a):
    # Extract 6 colors from an image.
    colors = colorgram.extract(image_a, 6)
    colors.sort(key=lambda c: c.hsl.h)
    # colorgram.extract returns Color objects, which let you access
    # RGB, HSL, and what proportion of the image was that color.
    valence = 0
    for j in range(len(colors)):
        rgb = colors[j].rgb
        r, g, b = rgb
        valence += hsv(r, g, b)

    valence = (valence/len(colors))/175.95

    return valence


def extract_objects(image_a):
    # Load Yolo
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    # Loading image
    img = cv2.imread(image_a)
    try:
        img = cv2.resize(img, None, fx=1, fy=1)
        height, width, channels = img.shape
        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

        for j in range(len(class_ids)):
           class_ids = list(dict.fromkeys(class_ids))

        for r in range(len(class_ids)):
            if len(class_ids) == 0:
                return None
            else:
                return str(classes[class_ids[r]])
    except:
        pass
