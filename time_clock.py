import tensorflow as tf
from imutils.video import VideoStream

import argparse
import src.facenet as facenet
import imutils
import os
import sys
import math
import pickle
import src.align.detect_face as detect_face
import numpy as np
import cv2
import collections
# from sklearn.svm import SVC

from keras.models import load_model
from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
import time

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, default='liveness.model',
                    help="path to trained model")
    ap.add_argument("-l", "--le", type=str, default='le.pickle',
                    help="path to label encoder")
    ap.add_argument("-d", "--detector", type=str, default='face_detector',
                    help="path to OpenCV's deep learning face detector")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'src/Models/facemodel.pkl'
    FACENET_MODEL_PATH = 'src/Models/20180402-114759.pb'

    recog_graph = tf.Graph()
    liveness_graph = tf.Graph()

    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        recog_model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    recog_sess = tf.Session(graph=recog_graph)
    liveness_sess = tf.Session(graph=liveness_graph)

    with recog_graph.as_default():
        with recog_sess.as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            pnet, rnet, onet = detect_face.create_mtcnn(
                recog_sess, "src/align")

            people_detected = set()
            person_detected = collections.Counter()
            recog_init = tf.local_variables_initializer()

    # Load model nhan dien fake/real
    print("[INFO] loading liveness detector...")
    with liveness_graph.as_default():
        with liveness_sess.as_default():
            liveness_model = load_model(args["model"])
            liveness_init = tf.local_variables_initializer()

    le = pickle.loads(open(args["le"], "rb").read())
    cap = VideoStream(src=0).start()
    #cap = cv2.VideoCapture('http://192.168.1.101:4747/video')

    while (True):
        cropped_array = []
        bb_array = []
        name_array = []
        frame = cap.read()
        (h, w) = frame.shape[:2]
        frame = imutils.resize(frame, width=600)
        frame = cv2.flip(frame, 1)

        # ?===================Face Recognition=====================================
        with recog_graph.as_default():
            with recog_sess.as_default():
                recog_sess.run(recog_init)
                bounding_boxes, _ = detect_face.detect_face(
                    frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
                if faces_found > 0:
                    det = bounding_boxes[:, 0:4]
                    bb = np.zeros((faces_found, 4), dtype=np.int32)
                    for i in range(faces_found):
                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]
                        # print(bb[i][3]-bb[i][1])
                        # print(frame.shape[0])
                        # print((bb[i][3]-bb[i][1])/frame.shape[0])
                        if (bb[i][3]-bb[i][1])/frame.shape[0] > 0.25:
                            cropped = frame[bb[i][1]:bb[i]
                                            [3], bb[i][0]:bb[i][2], :]
                            scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                interpolation=cv2.INTER_CUBIC)
                            scaled = facenet.prewhiten(scaled)
                            scaled_reshape = scaled.reshape(
                                -1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                            feed_dict = {
                                images_placeholder: scaled_reshape, phase_train_placeholder: False}
                            emb_array = recog_sess.run(
                                embeddings, feed_dict=feed_dict)

                            predictions = recog_model.predict_proba(
                                emb_array)
                            best_class_indices = np.argmax(
                                predictions, axis=1)
                            best_class_probabilities = predictions[
                                np.arange(len(best_class_indices)), best_class_indices]
                            best_name = class_names[best_class_indices[0]]
                            # print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                            if best_class_probabilities > 0.8:
                                cropped_array.append(cropped)
                                bb_array.append(bb)
                                name_array.append(
                                    class_names[best_class_indices[0]])

                                person_detected[best_name] += 1
                            else:
                                name = "Unknown"

        # ?===================Liveness=====================================
        with liveness_graph.as_default():
            with liveness_sess.as_default():
                liveness_sess.run(liveness_init)
                for cropped, bb, name in zip(cropped_array, bb_array, name_array):
                    #TODO: Liveness
                    face = cv2.resize(cropped, (32, 32))
                    face = face.astype("float") / 255.0
                    face = img_to_array(face)
                    face = np.expand_dims(face, axis=0)

                    # Dua vao model de nhan dien fake/real
                    preds = liveness_model.predict(face)[0]

                    j = np.argmax(preds)
                    label = le.classes_[j]

                    # TODO: Draw Identicated Face
                    if (j == 0):
                        cv2.rectangle(
                            frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0, 255), 2)
                        text_x = bb[i][0]
                        text_y = bb[i][3] + 20

                        #name = class_names[best_class_indices[0]]
                        cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 255), thickness=1, lineType=2)
                        cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 255), thickness=1, lineType=2)
                    else:
                        cv2.rectangle(
                            frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                        text_x = bb[i][0]
                        text_y = bb[i][3] + 20

                        #name = class_names[best_class_indices[0]]
                        cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 255), thickness=1, lineType=2)
                        cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 255), thickness=1, lineType=2)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
