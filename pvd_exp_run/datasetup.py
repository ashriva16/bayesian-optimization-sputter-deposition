import numpy as np
import pandas as pd
import re
import os


def stress_resist_data_loader():

    stress_dir = "../../data/raw/old_format/stress_data/"

    df_stress = pd.DataFrame(columns=['Pressure', 'Final Film Stress_MPa'])
    POWER = []
    for path, subdirs, files in os.walk(stress_dir):
        for i, file in enumerate(files):
            arr = re.split('[^0-9a-zA-Z]+', file)
            power = int(re.findall(r'\d+', arr[-4])[0])
            press = int(re.findall(r'\d+', arr[-3])[0])
            if power == 750:
                colnames = ['Ar Pressure_mtorr', 'Final Film Stress_MPa']
            else:
                colnames = ['Argon Pressure_mTorr', 'Final Film Stress_MPa']
            if(i == 0):
                temp_df = pd.read_csv(stress_dir + file, delimiter=',',
                                      usecols=colnames)
                df_stress = pd.concat([df_stress,
                                       temp_df.rename(columns={colnames[0]:'Pressure'})],
                                      ignore_index=True)
            else:
                temp_df = pd.read_csv(stress_dir + file, delimiter=',',
                                      usecols=colnames)
                df_stress = pd.concat([df_stress,
                                       temp_df.rename(columns={colnames[0]:'Pressure'})],
                                      ignore_index=True)

            POWER.append(power)

    df_stress['Power'] = POWER
    df_stress['Pressure'] = df_stress['Pressure'].astype(int)

    import pathlib

    data_dir = "../../data/raw/old_format/resistivity"
    batches = os.listdir(data_dir)

    resist_df = pd.DataFrame()
    for batch in os.listdir(data_dir):
        df_temp = pd.read_csv(
            os.path.join(data_dir, batch),
            usecols=['Sputter Power (W)', 'Ar Pressure (mTorr)',
                     'Avg Sheet Resistance (ohm/sq)'],
        )
        resist_df = pd.concat([resist_df, df_temp], ignore_index=True)

    resist_df = resist_df.rename(columns={'Ar Pressure (mTorr)': 'Pressure'})
    resist_df = resist_df.rename(columns={'Sputter Power (W)': 'Power'})

    # data_df = pd.concat([resist_df, df_stress],  ignore_index=True)
    # print(data_df.to_string())

    resist_df_new = resist_df.sort_values(
        ['Pressure', 'Power'], ascending=[True, True])
    df_new = df_stress.sort_values(
        ['Pressure', 'Power'], ascending=[True, True])

    df_new['Avg Sheet Resistance (ohm/sq)'] = list(
        resist_df_new['Avg Sheet Resistance (ohm/sq)'])
    df_new.reset_index(inplace=True, drop=True)
    df_new = df_new.drop(
        np.where(pd.isnull(df_new).iloc[:, 1])[0])

    x_train = np.array(df_new.iloc[:, [0, 2]]).astype(float)
    stress_y_train = np.array(df_new.iloc[:, 1]).astype(float)
    resist_y_train = np.array(df_new.iloc[:, 3]).astype(float)

    # x_train, stress_y_train, resist_y_train = append_new_data(
    #     x_train, stress_y_train, resist_y_train)

    return x_train, stress_y_train.reshape(-1,1), resist_y_train.reshape(-1,1)

def append_new_data():

    NEW_DATA_DIR = "/home/ashriva/work/beyond_finger_printing/ankit/Experiments/Data/raw/new_bayes_exp/"
    resist_file = "BayesianOptimization_ResistivityData.csv"
    stress_file = "BayesianOptimization_StressData.csv"

    columnnames = ['Ar Pressure (mTorr)', 'Sputter Power (W)', 'Avg Sheet Resistance (ohm/sq)']
    resist_df = pd.read_csv(NEW_DATA_DIR+resist_file, delimiter=',', usecols=columnnames)

    columnnames = ['Ar Pressure (mTorr)', 'Sputter Power (W)', 'Initial Film Stress (MPa)']
    stress_df = pd.read_csv(NEW_DATA_DIR+stress_file, delimiter=',', usecols=columnnames)

    combine_df = resist_df
    assert all(resist_df['Sputter Power (W)'] == stress_df['Sputter Power (W)'])
    assert all(resist_df['Ar Pressure (mTorr)'] == stress_df['Ar Pressure (mTorr)'])
    combine_df['Stress'] = stress_df['Initial Film Stress (MPa)']

    params_new = combine_df[['Ar Pressure (mTorr)','Sputter Power (W)']].to_numpy()
    stress_new = combine_df['Stress'].to_numpy()
    resistivity_new = combine_df['Avg Sheet Resistance (ohm/sq)'].to_numpy()
    return params_new, stress_new, resistivity_new