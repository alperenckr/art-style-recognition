import pandas as pd
import cv2
import os

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
            norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            cv2.imwrite(os.path.join(save_path,row["filename"]),norm_image)
        except:
            pass
        x = x + 1
        print(x)
    return


styles = ["Realism","Cubism","Impressionism","Expressionism","High Renaissance","Sembolism"]
train_df = choose_styles(styles=styles)
resize_images_and_save(train_df,"D:/painter-by-numbers/train_prep","D:/painter-by-numbers/train")



#X_train, y_train = get_pictures("D:/painter-by-numbers/train_prep",train_df)

print(X_train)