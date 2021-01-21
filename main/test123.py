#! C:\Users\User\miniconda3\envs\tensorflow\python.exe
import glob
import test as zf
from flask import Flask, request, Response
import numpy
import sys
import json
import cv2
import extract_features as ef

# filename = open("C:/Users/User/PycharmProjects/FYP/main/output.txt", 'w')

# sys_out = sys.stdout
# sys.stdout = filename


app = Flask(__name__)

@app.route('/webhook', methods=['GET'])
def respond():
    valence = 0
    count = 0
    error = ""
    for img in glob.glob("E:/gambar/211.jpg"):
        # print(img)"C:/Users/User/Desktop/twitterhehe/storage/app/*.jpg"

        if ef.extract_face(img) == None or ef.extract_colors(img) == None or ef.extract_objects(img) == None:
            error = "Images rejected"
        else:
            count += 1
            if ef.extract_face(img) == "Image is Positive":
                valence += 50 + ef.extract_colors(img)
            else:
                valence += 25 + ef.extract_colors(img)

    sentiment = valence/count


    # some JSON:
    a = {'sentiment': sentiment, 'message': error}

    x = json.dumps(a)
    # parse x:
    y = json.loads(x)

    return x

    # return Response(status=200)

if __name__ == '__main__':
    app.run(debug=True)



# filename.close()