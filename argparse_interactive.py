''' DOC
This is a script to train and test face recognition using opencv '''

import argparse
import os
import cv2
import numpy as np
import pyttsx

dataset_path = '/home/suhanna/work/opencv_1/opencv-2.4.9/build/lib/myscripts/image_data_set'
newpath = os.path.join(dataset_path,'s6')
face_cascade = cv2.CascadeClassifier('/home/suhanna/work/opencv_1/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_default.xml') #haarcascade_frontalface_default.xml haarcascade_frontalface_alt.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load("face_detection_trained_model.cv2")

folder_count = len(os.listdir(dataset_path)) 
label_name_dict = {i.split('_')[1]:i.split('_')[0] for i in os.listdir(dataset_path)}
parser = argparse.ArgumentParser(prog='PROG', description='description')
parser.add_argument('cmd', choices=['train','test','help','quit'])

def get_images_and_labels(path): 
    image_paths = [os.path.join(path, f) for f in os.listdir(path)] 
    images = []
    labels = [] 
    for image_path in image_paths: 
        for i in os.listdir(image_path):
            image_pil = cv2.imread(os.path.join(image_path,i))
            if image_pil is None:
                raise ValueError('Image not found')
            image_pil_gray=cv2.cvtColor(image_pil, cv2.COLOR_BGR2GRAY)
            person_label = os.path.basename(image_path).split('_')[1]
            faces = face_cascade.detectMultiScale(image_pil_gray)
            for (x, y, w, h) in faces: 
                images.append(image_pil_gray[y: y + h, x: x + w]) 
                labels.append(person_label) 
                cv2.imshow("Adding faces to traning set...", image_pil_gray[y: y + h, x: x + w]) 
                cv2.waitKey(100) 
    return images, labels

def train_base_image():
    ''' create a trainig set '''

    ''' 1. Capture image '''
    camera = cv2.VideoCapture(0)
    newpath = os.path.join(dataset_path,name+'_'+str(folder_count))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    while True:
        return_value,image = camera.read()
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        cv2.imshow('image',gray)
        i=0
        while(i<8):
            image_name = str(i) + '.jpg'
            cv2.imwrite(newpath+'/'+image_name,image)
            i = i+1
        break
    camera.release()
    cv2.destroyAllWindows()

    '''2. Detect face '''
    images, labels = get_images_and_labels(dataset_path)
    cv2.destroyAllWindows()

    '''3. Training '''
    recognizer.train(images, np.array(labels).astype('int'))
    recognizer.save("face_detection_trained_model.cv2")
    print "\n ****** Training Completed ******"

def recognize_face():
    ''' recognize face '''

    '''1. Capture image '''
    camera = cv2.VideoCapture(0)
    i = 0;
    while i<1:
        return_value,image = camera.read()
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        cv2.imshow('image',gray)
        cv2.imwrite('test_input.jpg',image)
        i=i+1
    camera.release()
    cv2.destroyAllWindows()

    '''2. Detect face '''
    test_img = cv2.imread('test_input.jpg')
    if test_img is None:
        raise ValueError('Image not found')
    test_img_gray=cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(test_img_gray)

    '''3. Recognize Face '''
    test_label = label_name_dict.keys()
    for (x, y, w, h) in faces: 
        label_predicted,result = recognizer.predict(test_img_gray[y: y + h, x: x + w])
        engine = pyttsx.init()
        if str(label_predicted) in test_label: 
            print "{} is Correctly Recognized with confidence {}".format(label_predicted, result)
            finded = "Hello {}".format(label_name_dict[str(label_predicted)])
        else:
            finded = "Sorry for the inconvinence"
        engine.say(finded)
        engine.runAndWait()
    print "\n ****** Recognition Completed ******"

print "\nselect your option :\n\t1. Train\n\t2. Test\n\t3. Help\n\t4. Quit"
while True:
    astr = raw_input('$: ')
    try:
        args = parser.parse_args(astr.split())
    except SystemExit: # trap argparse error message
        print 'exception error'
        continue
    if args.cmd in ['train', 'test']:
    	if args.cmd == 'train':
            name = raw_input('Your Good name : ')
            folder_count+=1
            label_name_dict[folder_count] = name

            train_base_image()
    	else:
    		recognize_face()
    elif args.cmd == 'help':
        parser.print_help()
    else:
        print '\n****** Thank you ******'
        break