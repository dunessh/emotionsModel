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


app = Flask(__name__)

@app.route('/webhook', methods=['GET'])

def respond():
    username = request.args.get('username')
    sentiment = 0
    count = 0
    color = 0
    countImg = 0
    valance = 0
    positive = 0
    negative = 0
    dict = {}
    error =""
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='emotions',
                                             user='root',
                                             password='')

        for img in glob.glob("C:/xampp/htdocs/emotionsImage/public/images/*.jpg"):
            # C:/xampp/htdocs/emotionsImage/storage/app/*.jpg
            valance = 0
            path_file = os.path.basename(img).replace(".jpg", "")
            sql_select_Query = "select sentiment from image where name=%s"
            name = (path_file,)
            cursor = connection.cursor()
            cursor.execute(sql_select_Query, name)
            record = cursor.fetchone()
            if record is None:
                Path("cropped_faces/" + path_file).mkdir(parents=True, exist_ok=True)
                print(path_file)
                print(img)
                countImg = ef.crop_faces(img)
                count = 0
                for img_a in glob.glob("cropped_faces/" + path_file + "/*.jpg"):
                    Path("Data4/" + path_file).mkdir(parents=True, exist_ok=True)
                    path_to = "Data4/" + path_file
                    count = ef.data(img, img_a, count)
                    valance = ef.predict(path_to)
                valance = (valance*33.5)+33.5
                print(valance)
                color = pf.extract_colors(img)*10
                print(color)
                valance = valance + color
                print(valance)
                if valance > 50:
                    positive += 1
                else:
                    negative += 1
                dict[os.path.basename(img)] = valance
                sentiment += valance
                mySql_insert_query = """INSERT INTO image (name, id_str, sentiment) 
                                       VALUES 
                                       (%s, %s, %s) """
                recordTuple = (path_file, username, valance)
                cursor = connection.cursor()
                cursor.execute(mySql_insert_query,recordTuple)
                connection.commit()
                cursor.close()
            else:
                valance = float('.'.join(str(ele) for ele in record))
                sentiment += valance
                if valance > 50:
                    positive += 1
                else:
                    negative += 1


        sentiment = sentiment / (positive + negative)

    except Error as e:
        print("Error reading data from MySQL table", e)

    finally:
        if (connection.is_connected()):
            connection.close()
            cursor.close()
            print("MySQL connection is closed")




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


