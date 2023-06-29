import os.path
import pandas as pd
import os

if __name__ == "__main__":
    directory = '/Users/white/Desktop/valak/Valak/Main_ValaK/DQN_project/data'
    file_names = os.listdir(directory)
    for filename in file_names:
        path = os.path.join(directory, filename)
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'])
        code = filename[11:17]
        date = df["time"].dt.strftime('%Y%m%d')
        date = date[0]
        oprc = df['stock_price'].values[0]
        df['stock_price'] = ((df['stock_price'] - oprc) / oprc).round(5)
        for i in range(1,11):
            idx = 'icln' + str(i)
            df[idx] = ((df[idx] - oprc) / oprc).round(5)
        df['akpr'] = df['akpr'].round(5)
        df['t_akpr'] = df['t_akpr'].round(5)
        new_path = os.path.join('/Users/white/Desktop/valak/Valak/Main_ValaK/DQN_project/filtered_data', filename)
        df.to_csv(new_path, index=False, mode='w')
