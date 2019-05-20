import pandas as pd
import cv2
import os
import zipfile
import numpy as np

def choose_styles(styles: list):
    print("Styles are: ", *styles)
    train_file = pd.read_csv("D:/painter-by-numbers/train_info.csv")
    train_file = train_file.iloc[train_file[train_file['style'].isin(styles)].index]
    print(train_file.shape)
    return train_file

def get_pictures(folder_name,train_df):
    X_data_set = []
    Y_data_set = []
    for i, row in train_df.iterrows():
        X_data_set.append(cv2.imread(os.path.join(folder_name,row["filename"])))
        Y_data_set.append(row["style"])
    return X_data_set, Y_data_set

def resize_images_and_save(train_df,save_path,images_folder):
    x = 0
    for i, row in train_df.iterrows():
        try: 
            img = cv2.imread(os.path.join(images_folder,row["filename"]))
            img = cv2.resize(img,(224,224))
            #norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            cv2.imwrite(os.path.join(save_path,row["filename"]),norm_image)
        except:
            pass
        x = x + 1
        print(x)
    return


styles = ["Realism","Cubism","Impressionism","Expressionism","High Renaissance","Symbolism"]
style_codes = {"Realism":         [1,0,0,0,0,0],
               "Cubism":          [0,1,0,0,0,0],
               "Impressionism":   [0,0,1,0,0,0],
               "Expressionism":   [0,0,0,1,0,0],
               "High Renaissance":[0,0,0,0,1,0],
               "Symbolism":       [0,0,0,0,0,1]}

train_df = choose_styles(styles=styles)
#resize_images_and_save(train_df,"D:/painter-by-numbers/train_prep","D:/painter-by-numbers/train")

def get_pictures_from_zip(zip,train_df):
  x = 0
  print("func started")
  X_data_set = []
  Y_data_set = []
  X_test = []
  Y_test = []
  with zipfile.ZipFile(zip, 'r') as train:
    print("zip loaded")
    for i,row in train_df.iterrows():
      x = x + 1
      if x >= 20000:
        try:
          data = train.read(row["filename"])
          X_test.append(cv2.imdecode(np.frombuffer(data, np.uint8), 1).transpose(2,0,1))
          Y_test.append(row["style"])
          #print("image_loaded")
        except:
          print(row["filename"])
          del X_test[-1]
          del Y_test[-1]

      try:
        data = train.read(row["filename"])
        X_data_set.append(cv2.imdecode(np.frombuffer(data, np.uint8), 1).transpose(2,0,1))
        Y_data_set.append(row["style"])
        #print("image_loaded")
      except:
        print(row["filename"])
        del X_data_set[-1]
        del Y_data_set[-1]
  
  return X_data_set, Y_data_set, X_test, Y_test

#X_train, y_train = get_pictures("D:/painter-by-numbers/train_prep",train_df)

#print(X_train)