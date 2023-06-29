import os.path
import numpy as np
import pandas as pd
import datetime
import os




def save_data_to_csv(path, filename, data):
    columns=['time',
            'stock_hold',
            'stock_price',
            'trade_volume',
            'contract_volume',
            'akpr', 't_akpr', 'a_akpr', 'b_akpr',
            'icln1', 'icln2','icln3','icln4','icln5','icln6', 'icln7','icln8','icln9','icln10',
            'term']
    if not os.path.isfile(path):  # 파일이 존재하지 않으면
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(path, index=False, mode='w')  # 파일을 새로 만들어서 저장하고
    else:  # 파일이 이미 존재하면
        df = pd.DataFrame(data)
        df.to_csv(path, index=False, header=False, mode='a')  # 데이터만 추가해서 저장
if __name__ == "__main__":
    directory = '/Users/white/Desktop/valak/Valak/Main_ValaK/DQN_project/data'
    file_names = os.listdir(directory)
    for filename in file_names:
        path = '/Users/white/Desktop/valak/Valak/Main_ValaK/DQN_project/data/' + filename
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'])
        grouped = df.groupby(df['time'].dt.date)
        for date, group_df in grouped:
            # 파일 이름 생성
            new_filename = f'{date}_{filename}'
            print(new_filename)
            # 해당 날짜의 데이터를 파일로 저장
            new_path = '/Users/white/Desktop/valak/Valak/Main_ValaK/DQN_project/data/' + new_filename
            group_df.to_csv(new_path, index=False, mode='w')
