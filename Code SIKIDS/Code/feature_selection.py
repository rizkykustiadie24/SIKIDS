from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'

"""
Fungsi informationGain: 
Fungsi featureSelection: 
Fungsi sigmoidNormalization: 
"""

def informationGain(data, result):
    print("1. \t Perhitungan Nilai Information Gain Pada Tiap Fitur")
    # Melakukan seleksi fitur dengan menggunakan metode information gain melalui fungsi mutual_info_classif
    X = data.iloc[:, 0:41]
    Y = data.iloc[:, 41]
    mutual_info = mutual_info_classif(X, Y)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = X.columns
    print("\t Nilai Information Gain dari Tiap Fitur \t ")
    for index, value in mutual_info.items():
        print("\t - " + index + " " + str(value))
    # Melakukan plot feature dengan nilai information gain-nya masing-masing
    plt.figure(figsize=(6,6))
    ax = plt.axes()
    ax.bar(mutual_info.index,mutual_info.values)
    plt.xticks(rotation=90)
    plt.title("Hasil Information Gain")
    
    nama_file = 'Diagram Hasil Information Gain.png'
    plt.savefig(nama_file)
    print("\t "+ nama_file+ " Berhasil Disimpan!!!")
    result.append(nama_file)

    # Menyeleksi feature dengan nilai threshold yang ditentukan
    newData, selected_feature, result = featureSelection(mutual_info, data, result)

    # Melakukan plot feature yang telah diseleksi
    plt.figure(figsize=(6,6))
    ax = plt.axes()
    ax.bar(selected_feature.index,selected_feature.values)
    plt.xticks(rotation=90)
    plt.title("Diagram Visualisasi Fitur")
    nama_file = 'Diagram Visualisasi Fitur yang Digunakan.png'
    plt.savefig(nama_file)
    print(nama_file+ " Berhasil Disimpan!!!")
    result.append(nama_file)
    return newData, selected_feature, result
    
def featureSelection(mutual_info, dataset, result):
    print("\n")
    print("2. \t Menetapkan Nilai Threshold Yang Akan Digunakan Untuk Seleksi Fitur (Nilai Rata-Rata Gain)")
    mean = mutual_info.values.mean() # Mendapatkan nilai rata-rata dari semua nilai information gain pada feature
    print("\t Nilai Threshold (Mean) = ", mean)
    print("\n")
    print("3. \t Melakukan Seleksi Fitur Yang Memiliki Nilai Lebih Besar dari Mean")
    print("\t Fitur Yang Digunakan \t ")
    selected_feature = mutual_info[mutual_info > mean] # Mendapatkan feature yang memiliki nilai IG lebih besar dari nilai rata-rata
    for index, value in selected_feature.items():
        print("\t - " + index)
    newData= dataset[selected_feature.index] # Menyeleksi dataset dengan hanya berisi feature hasil seleksi
    newClass = dataset.iloc[:, -1]
    newData["Class"] = newClass
    newData, result= sigmoidNormalization(newData, result) # Melakukan normalisasi dengan dataset yang baru

    return newData, selected_feature, result

def sigmoidNormalization(newData,result):
    print("\n")
    print("-------------------------------------------------- \t Tahap Normalisasi Data dengan Sigmoid \t --------------------------------------------------")
    # Melakukan normalisasi dengan metode sigmoid dengan cara menjadikan dataset ke dalam array numpy terlebih dahulu
    nameColumns = list(newData)
    X = newData.loc[:, newData.columns != 'Class']
    X = X.to_numpy()
    sig = 1/(1 + np.exp(-X))
    normalData = pd.DataFrame(sig)
    normalData["Class"] = newData["Class"]
    normalData.columns = nameColumns
    # Menyimpan data hasil normalisasi ke dalam data berbentuk csv yang baru
    nama_file = 'Dataset Hasil Normalisasi.csv'
    print(nama_file+ " Berhasil Disimpan!!!")
    normalData.to_csv(nama_file)
    result.append(nama_file)
    return normalData, result