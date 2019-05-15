import cv2
import numpy as np
import os
import sys

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

class FaceDetector(object):
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)
    
    def detect(self, image, biggest_only=True):
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (30, 30)
        biggest_only = True
        faces_coord = self.classifier.detectMultiScale(image,
        	scaleFactor = scale_factor,
        	minNeighbors = min_neighbors,
        	minSize = min_size,
        	flags = cv2.CASCADE_SCALE_IMAGE
        )
        return faces_coord

def cut_faces(image, faces_coord):
    faces = []
    for (x, y, w, h) in faces_coord:
        w_rm = int(0.3 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])
    return faces

def resize(images, size=(299, 299)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)
    return images_norm



def normalize_faces(image, faces_coord):
    faces = cut_faces(image, faces_coord)
    faces = resize(faces)
    return faces

def generate_pretrained_images(image, current_count, person, detector):
    out_dir = "retrain_images/" + person
    faces_coord = detector.detect(image, True)
    faces = normalize_faces(image ,faces_coord)
    for i, face in enumerate(faces):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        new_file_name = out_dir + '/%s.jpeg' % (current_count[0])
        print ("   Writing:", new_file_name)
        cv2.imwrite(new_file_name, faces[i])
        current_count[0] += 1

def collect_dataset():
    detector = FaceDetector("haarcascade_frontalface_default.xml")
    labels = []
    labels_dic = {}
    people = [person for person in listdir_nohidden("people/")]
    for i, person in enumerate(people):
        labels_dic[i] = person
        current_count = [0]
        path = "people/" + person
        for image in listdir_nohidden(path):
            print ("-Reading: " + path + '/' + image)
            current_image = cv2.imread(path + '/' + image, 0)
            generate_pretrained_images(current_image, current_count, person, detector)

            labels.append(person)
    print ("Done!")
    return (np.array(labels), labels_dic)

labels, labels_dic = collect_dataset()