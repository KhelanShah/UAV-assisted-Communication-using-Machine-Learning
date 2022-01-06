import pandas as pd 
import os
import numpy as np 
import cv2
from keras.preprocessing import image 
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten 
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
count = 0
 
cam = cv2.VideoCapture("C:\\Users\\MAFIA\\file_example_MP4_480_1_5MG.mp4") 

try: 
	
	# creating a folder named data 
	if not os.path.exists('images'): 
		os.makedirs('images') 

# if not created then raise error 
except OSError: 
	print ('Error: Creating directory of data') 

# frame 
currentframe = 0

while(True): 
	
	# reading from frame 
	ret,frame = cam.read() 

	if ret: 
		# if video is still left continue creating images 
		name = './images/frame' + str(currentframe) + '.jpg'
		print ('Creating...' + name) 

		# writing the extracted images 
		cv2.imwrite(name, frame) 

		# increasing counter so that it will 
		# show how many frames are created 
		currentframe += 1
	else: 
		break

# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows()


def get_image_value(path, dim): 
    '''This function will read an image and convert to a specified version and resize depending on which algorithm is being used. '''
    img = image.load_img(path, target_size = dim)
    img = image.img_to_array(img)
    return img/255

def get_img_array(img_paths, dim): 
    '''This fucntion takes a list of image paths and returns the np array corresponding to each image.  It also takes the dim and whether edge is specified in order to pass it to another function to apply these parameters.  This function uses get_image_value to perform these operations'''
    final_array = []
    from tqdm import tqdm
    for path in tqdm(img_paths):
        img = get_image_value(path, dim)
        final_array.append(img)
    final_array = np.array(final_array)  
    return final_array

def get_tts():
    '''This function will create a train test split'''  
    DIM =  (150,150) 
    np.random.seed(10)        
    pistol_paths = [f'C:/Users/MAFIA/OneDrive/Desktop/Separated/FinalImages/Pistol/{i}' for i in os.listdir('C:/Users/MAFIA/OneDrive/Desktop/Separated/FinalImages/Pistol')] 
    pistol_labels = [1 for i in range(len(pistol_paths))]
    rifle_paths = [f'C:/Users/MAFIA/OneDrive/Desktop/Separated/FinalImages/Rifle/{i}' for i in os.listdir('C:/Users/MAFIA/OneDrive/Desktop/Separated/FinalImages/Rifle')] 
    rifle_labels = [2 for i in range(len(rifle_paths))]    
    neg_paths = [f'C:/Users/MAFIA/OneDrive/Desktop/train/{i}' for i in os.listdir('C:/Users/MAFIA/OneDrive/Desktop/train')]
    np.random.shuffle(neg_paths)
    neg_paths = neg_paths[:len(pistol_paths)- 500]
    neg_labels = [0 for i in range(len(neg_paths))]

    np.random.shuffle(pistol_paths)
    pistol_paths = pistol_paths[:len(rifle_paths)+150]
    neg_paths = neg_paths[:len(rifle_paths)+150]

    pistol_labels = [1 for i in range(len(pistol_paths))]
    rifle_labels = [2 for i in range(len(rifle_paths))]
    neg_labels = [0 for i in range(len(neg_paths))]
    paths = pistol_paths + rifle_paths + neg_paths
    labels = pistol_labels + rifle_labels + neg_labels
    x_train, x_test, y_train, y_test = train_test_split(paths, labels, stratify = labels, train_size = .90, random_state = 10)

    new_x_train = get_img_array(x_train, DIM)
    new_x_test = get_img_array(x_test, DIM)
    
    print('Train Value Counts')
    print(pd.Series(y_train).value_counts())
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Test Value Counts')
    print(pd.Series(y_test).value_counts())
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('X Train Shape')
    print(new_x_train.shape)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('X Test Shape')
    print(new_x_test.shape)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_test = to_categorical(y_test)
    y_train = to_categorical(y_train)
    tts = (new_x_train, new_x_test, y_train, y_test)
    return tts

x_train, x_test, y_train, y_test = get_tts()


def get_conv_model(dim = (150,150, 3)):
    '''This function will create and compile a CNN given the input dimension'''
    inp_shape = dim
    act = 'relu'
    drop = .25
    kernal_reg = regularizers.l1(.001)
    optimizer = Adam(lr = .0001)    
    model = Sequential() 
    model.add(Conv2D(64, kernel_size=(3,3),activation=act, input_shape = inp_shape, 
                     kernel_regularizer = kernal_reg,
                     kernel_initializer = 'he_uniform',  padding = 'same', name = 'Input_Layer'))
    model.add(MaxPooling2D(pool_size=(2, 2),  strides = (3,3)))
    model.add(Conv2D(64, (3, 3), activation=act, kernel_regularizer = kernal_reg, 
                     kernel_initializer = 'he_uniform',padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides = (3,3))) 
    model.add(Conv2D(128, (3, 3), activation=act, kernel_regularizer = kernal_reg, 
                     kernel_initializer = 'he_uniform',padding = 'same'))
    model.add(Conv2D(128, (3, 3), activation=act, kernel_regularizer = kernal_reg, 
                     kernel_initializer = 'he_uniform',padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides = (3,3)))  
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(drop))
    model.add(Dense(3, activation='softmax', name = 'Output_Layer'))
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return model 
#prevents overfitting and saves models every time the validation loss improves
early_stopping = EarlyStopping(monitor='val_loss', verbose = 1, patience=10, min_delta = .00075)
model_checkpoint = ModelCheckpoint('ModelWeights.h5', verbose = 1, save_best_only=True,
                                  monitor = 'val_loss')
lr_plat = ReduceLROnPlateau(patience = 2, mode = 'min')
epochs = 50
batch_size = 32
model = get_conv_model()
model_history = model.fit(x_train, y_train, batch_size = batch_size,
            epochs = epochs, 
     callbacks = [early_stopping, model_checkpoint, lr_plat], validation_data = (x_test, y_test), verbose= 1)

def non_max_suppression(boxes, overlapThresh= .5):
    '''This image was taken from PyImageSearch... again cannot thank that guy enough'''
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]    
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1, yy1, xx2, yy2 = np.maximum(x1[i], x1[idxs[:last]]), np.maximum(y1[i], y1[idxs[:last]]), np.minimum(x2[i], x2[idxs[:last]]), np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w, h = np.maximum(0, xx2 - xx1 + 1), np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return pick

def get_img_prediction_bounding_box(path, model, dim):
    '''This function will create a bounding box over what it believes is a weapon given the image path, dimensions, and model used to detect the weapon.  Dimensions can be found within the Var.py file.  This function is still being used as I need to apply non-max suppresion to create only one bounding box'''
    img = get_image_value(path, dim)   
    img = img.reshape(1, img.shape[0], img.shape[1], 3)
    pred = model.predict(img)[0]
    category_dict = {0: 'Not Suspicious', 1: 'Suspicious', 2: 'Suspicious'}
    cat_index = np.argmax(pred)
    cat = category_dict[cat_index]
    #print(f'{path}\t\tPrediction: {cat}\t{int(pred.max()*100)}% Confident')

    #speed up cv2
    cv2.setUseOptimized(True)
    cv2.setNumThreads(10) #change depending on your computer
    img = cv2.imread(path)
    clone = img.copy() 
    clone2 = img.copy()
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()

    rects = ss.process() 
    windows = []
    locations = []
    #print(f'Creating Bounding Boxes for {path}')
    for x, y, w,h in rects[:1001]: 
        startx, starty, endx, endy = x, y, x+w, y+h 
        roi = img[starty:endy, startx:endx]
        roi = cv2.resize(roi, dsize =dim, interpolation = cv2.INTER_CUBIC)
        windows.append(roi)
        locations.append((startx, starty, endx, endy))
    windows = np.array(windows)
    windows = windows.reshape(windows.shape[0], windows.shape[1], windows.shape[2], 3)
    windows = np.array(windows)
    locations = np.array(locations)
    predictions = model.predict(windows)
    nms = non_max_suppression(locations)
    bounding_cnt = 0
    for idx in nms:
        if np.argmax(predictions[idx]) != cat_index: 
            continue
        startx, starty, endx, endy = locations[idx]
        cv2.rectangle(clone, (startx, starty), (endx, endy), (0,0,255), 2)
        text = f'{category_dict[np.argmax(predictions[idx])]}: {int(predictions[idx].max()*100)}%'
        cv2.putText(clone, text, (startx, starty+15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0),2)
        bounding_cnt += 1

    if bounding_cnt == 0: 
        pred_idx= [idx for idx, i in enumerate(predictions) if np.argmax(i) == cat_index]
        cat_locations = np.array([locations[i] for i in pred_idx])
        nms = non_max_suppression(cat_locations)
        if len(nms)==0:
            cat_predictions = predictions[:,cat_index]
            pred_max_idx = np.argmax(cat_predictions)
            pred_max = cat_predictions[pred_max_idx]
            pred_max_window = locations[pred_max_idx]
            startx, starty, endx, endy = pred_max_window
            cv2.rectangle(clone, (startx, starty), (endx, endy),  (0,0,255),2)
            text = f'{category_dict[cat_index]}: {int(pred_max*100)}%'
            cv2.putText(clone, text, (startx, starty+15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0),2)
        for idx in nms: 
            startx, starty, endx, endy = cat_locations[idx]
            cv2.rectangle(clone, (startx, starty), (endx, endy), (0,0,255), 2)
            text = f'{category_dict[np.argmax(predictions[pred_idx[idx]])]}: {int(predictions[pred_idx[idx]].max()*100)}%'
            cv2.putText(clone, text, (startx, starty+15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0),2)        
   # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    cv2.imshow('image', np.hstack([clone, clone2]))
    cv2.waitKey(1)
    ss.clear()
    
    if(cat == "Suspicious"):
        count = count + 1
        print(i,count)
        
    return predictions
 
dim = (150, 150, 3)    
normal_model = get_conv_model(dim)
normal_model.load_weights('ModelWeights.h5') #path to the model weights
test_folder = 'C:/Users/MAFIA/test' #folder where you will put your images to test
predictions = []
for idx, i in enumerate([i for i in os.listdir(test_folder) if i != 'ipynb_checkpoints']):
    img_path = f'{test_folder}/{i}'
    pred = get_img_prediction_bounding_box(img_path, normal_model, dim = (150,150))
    predictions.append(pred)
