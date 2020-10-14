import cv2
import sqlite3
from os import path
import os
import src.align.detect_face as detect_face
import src.facenet as facenet
import tensorflow as tf

# detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Hàm cập nhật tên và ID vào CSDL


def insertOrUpdate(id, name):
    conn = sqlite3.connect("FaceBaseNew.db")
    cursor = conn.execute('SELECT * FROM People WHERE ID='+str(id))
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
        break

    if isRecordExist == 1:
        cmd = "UPDATE people SET Name=' "+str(name)+" ' WHERE ID="+str(id)
    else:
        cmd = "INSERT INTO people(ID,Name) Values(" + \
            str(id)+",' "+str(name)+" ' )"

    conn.execute(cmd)
    conn.commit()
    conn.close()


def main_app(user_name, user_id):
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #id = input('Nhập mã nhân viên:')
    #name = input('Nhập tên nhân viên;')
    #print("Bắt đầu chụp ảnh nhân viên, nhấn q để thoát!")

    insertOrUpdate(user_id, user_name)

    sampleNum = 0
    MAX_IMG = 10
    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    FACENET_MODEL_PATH = 'src/Models/20180402-114759.pb'

    recog_graph = tf.Graph()
    recog_sess = tf.Session(graph=recog_graph)

    with recog_graph.as_default():
        with recog_sess.as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)
            pnet, rnet, onet = detect_face.create_mtcnn(
                recog_sess, "src/align")

            img_num = 0

            while(img_num <= MAX_IMG):

                ret, img = cam.read()

                # Lật ảnh cho đỡ bị ngược
                img = cv2.flip(img, 1)

                # Kẻ khung giữa màn hình để người dùng đưa mặt vào khu vực này
                centerH = img.shape[0] // 2
                centerW = img.shape[1] // 2
                sizeboxW = 300
                sizeboxH = 400
                cv2.rectangle(img, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
                              (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 5)

                # Đưa ảnh về ảnh xám
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Nhận diện khuôn mặt
                # faces = detector.detectMultiScale(gray, 1.3, 5)
                bounding_boxes, _ = detect_face.detect_face(
                    img, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
                faces = bounding_boxes[:, 0:4]
                for (x1, y1, x2, y2) in faces:
                    x1 = int(x1)
                    x2 = int(x2)
                    y1 = int(y1)
                    y2 = int(y2)
                    img_num += 1
                    # Vẽ hình chữ nhật quanh mặt nhận được
                    cv2.rectangle(img, (x1, y1),
                                  (x2, y2), (255, 0, 0), 2)
                    sampleNum = sampleNum + 1
                    # Ghi dữ liệu khuôn mặt vào thư mục dataSet
                    if not path.isdir(os.getcwd() + "/Dataset/FaceData/processed/" + user_id):
                        os.makedirs(
                            os.getcwd() + "/Dataset/FaceData/processed/" + user_id)
                    cv2.imwrite(os.getcwd() + "/Dataset/FaceData/processed/" + user_id + "/User." + user_id + '.' + str(sampleNum) +
                                ".jpg", gray[y1:y2, x1:x2])

                cv2.imshow('frame', img)
                # Check xem có bấm q hoặc trên 100 ảnh sample thì thoát
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                elif sampleNum > 100:
                    break

            cam.release()
            cv2.destroyAllWindows()
