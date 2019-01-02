from PIL import Image
import face_recognition
import os

"""
用于将人脸裁剪出来的小程序
"""
def find_and_save_face(web_file, face_file):
    # 用face_recognition打开图片
    image = face_recognition.load_image_file(web_file)
    # 找出图中所有人脸
    face_locations = face_recognition.face_locations(image)

    # print("I found {} face(s) in this photograph.".format(len(face_locations)))

    for face_location in face_locations:
        # 脸的坐标
        top, right, bottom, left = face_location
        # print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # 切分原图并保存
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.save(face_file)


list = os.listdir("D:\房悦\课程资料\人工智能\CNN_face\Images_原图\\")

for image in list:

    web_file = "D:/mycode/pycharm/face/CNN2/Images/" + image
    face_file = "D:/mycode/pycharm/face/CNN2/Images_resize/" + image
    try:
        find_and_save_face(web_file, face_file)
    except:
        print("fail")
