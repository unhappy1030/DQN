import os.path
import pandas as pd
import os

if __name__ == "__main__":
    directory = '/Users/white/Desktop/valak/Valak/Main_ValaK/DQN_project/data'
    file_names = os.listdir(directory)
    for filename in file_names:
        path = os.path.join(directory, filename)
        df = pd.read_csv(path)
        df = df.drop(['term'], axis=1)  # 열 제거
        df['time'] = pd.to_datetime(df['time'])
        grouped = df.groupby(df['time'].dt.date)
        new_path = os.path.join('/Users/white/Desktop/valak/Valak/Main_ValaK/DQN_project/filtered_data', filename)
        df.to_csv(new_path, index=False, mode='w')
        