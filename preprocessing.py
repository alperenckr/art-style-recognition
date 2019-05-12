import pandas as pd
import cv2


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
        X_data_set.append(cv2.imread(folder_name+"/"+row["filename"]))
        Y_data_set.append(row["style"])
    return X_dataset, Y_dataset



styles = ["Realism","Cubism","Impressionism","Expressionism","High Renaissance","Sembolism"]
train_df = choose_styles(styles=styles)
X_train, y_train = get_pictures("D:/painter-by-numbers/train",train_df)

print(X_train)