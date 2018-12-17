import pandas as pd
import os
import io

def read_to_pandas(path_to_folder, ID):
    files = os.listdir(path_to_folder)
    file_information = []
    count = 0
    for file in files:
        path_to_file = os.path.join(path_to_folder, file)
        with io.open(path_to_file,'r',encoding='utf-8') as f:
            text = f.read()
            count = count +1
            
        file_information.append([text, ID])

    print ("Process {} files".format(count))
    pandas_data =  pd.DataFrame(file_information, columns=['content', 'ID'])
    return pandas_data

folder_list = [
    "./data_fb_status/normal",
    "./data_fb_status/sensitive"
]

classID = 0
df = pd.DataFrame()
save_to = "./data_fb_status.csv"
for path_to_folder in folder_list:
    print ("Processing {}, classID = {}".format(path_to_folder, classID))
    df = df.append(read_to_pandas(path_to_folder, classID))
    classID = classID + 1
    print ("Done!!!")
df.to_csv(save_to, index=False, encoding='utf-8', escapechar='\\')