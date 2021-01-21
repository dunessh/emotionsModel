#! C:\Users\User\miniconda3\envs\tensorflow\python.exe
import glob
import test as ef
from flask import Flask, request, Response
import numpy
from pathlib import Path
import os
import sys
import json
import cv2
import extract_features as pf
import mysql.connector
from mysql.connector import Error

# filename = open("C:/Users/User/PycharmProjects/FYP/main/output.txt", 'w')

# sys_out = sys.stdout
# sys.stdout = filename


app = Flask(__name__)

@app.route('/webhook', methods=['GET'])

def respond():
    username = request.args.get('username')
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='emotions',
                                             user='root',
                                             password='')

        sql_select_Query = "select name from image where id_str=%s"
        id_str = (username,)
        cursor = connection.cursor()
        cursor.execute(sql_select_Query, id_str)
        record = cursor.fetchall()
        for row in record:
            print()

    except Error as e:
        print("Error reading data from MySQL table", e)
    finally:
        if (connection.is_connected()):
            connection.close()
            cursor.close()
            print("MySQL connection is closed")


    sentiment = 0
    count = 0
    countImg = 0
    valance = 0
    positive = 0
    negative = 0
    dict = {}
    error =""

    for img in glob.glob("images/*.jpg"):
        path_file = os.path.basename(img).replace(".jpg", "")
        Path("cropped_faces/" + path_file).mkdir(parents=True, exist_ok=True)
        countImg = ef.crop_faces(img, countImg)
        print(countImg)
        count = 0
        for img_a in glob.glob("cropped_faces/" + path_file + "/*.jpg"):
            Path("Data4/" + path_file).mkdir(parents=True, exist_ok=True)
            path_to = "Data4/" + path_file
            count = ef.data(img, img_a, count)
            valance = ef.predict(path_to)

        valance += pf.extract_colors(img)
        if valance > 40:
            positive += 1
        else:
            negative += 1
        dict[os.path.basename(img)] = valance
        sentiment += valance
        print(valance)

    print(dict)
    print(positive)
    print(negative)
    sentiment = sentiment / (positive + negative)

    # some JSON:
    a = {'sentiment': sentiment, 'message': error, 'positive': positive, 'negative': negative, 'dict': dict}

    x = json.dumps(a)
    # parse x:
    y = json.loads(x)

    return x

    # return Response(status=200)

if __name__ == '__main__':
    app.run(debug=True)



# filename.close()
