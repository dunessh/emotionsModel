import os
os.getcwd()
collection = "C:/Users/User/PycharmProjects/FYP/main/yeet"
for i, filename in enumerate(os.listdir(collection)):
    os.rename(os.path.join("C:/Users/User/PycharmProjects/FYP/main/yeet", filename), os.path.join("C:/Users/User/PycharmProjects/FYP/main/yeet", str(i) + "ho.jpg"))

