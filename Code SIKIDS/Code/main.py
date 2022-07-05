import os
import os.path 
import shutil
from pathlib import Path
import pandas as pd
from feature_selection import sigmoidNormalization, informationGain
from prepocessing_data import preData
from classification import randomForest, kCross, treeVisualization, testingData

"""
Fungsi getData: Melakukan preprocessing dataset dan mendapatkan feature yang akan digunakan pada klasifikasi melalui Information Gain
Fungsi splitData: Membagi dataset menjadi 80% data latih dan 20% data uji
Fungsi testData: Melakukan uji model dengan dataset yang lain2

"""
result = []

def getData(dir, key, result):
    print("\n")
    print("-------------------------------------------------- \t Tahap Pembacaan Dataset NSL-KDD \t --------------------------------------------------")
    pathname, extension = os.path.splitext(dir)
    filename_train = pathname.split('/')
    dataset = pd.read_csv(dir, header=None)
    print("Dataset Yang Digunakan \t\t = ", filename_train[-1])
    print("Jumlah Dataset Yang Digunakan \t = ",dataset.shape[0])
    print("\n")
    dataset, result = preData(dataset, key, result) # Preprocessing data meliputi mengganti nama kolom, data cleaning, dan data transform
    opt = "try"
    while(opt!= 1 or opt!=2):
        print("Pilihan Seleksi Fitur: ")
        print("1. Dengan Seleksi Fitur")
        print("2. Tanpa Seleksi Fitur")
        opt = int(input("Masukan Pilihan Anda (1 / 2) = "))
        if(opt==1):
            print("-------------------------------------------------- \t Tahap Seleksi Fitur dengan Information Gain \t --------------------------------------------------")
            newData, selected_feature, result = informationGain(dataset, result) # Mendapatkan feature melalui function informationGain
            name_file = 'List Fitur yang Digunakan Dalam Klasifikasi.csv'
            result.append(name_file)
            print(name_file+ " Berhasil Disimpan!!! \n")
            selected_feature.to_csv(name_file) # Menyimpan feature yang telah diseleksi ke dalam sebuah file agar dapat digunakan saat testing dengan dataset yang berbeda
            break
        elif(opt==2):
            newData, result = sigmoidNormalization(dataset, result)
            cols = newData.shape[1]
            cols = int(cols)
            print("Jumlah Fitur yang Digunakan = ", cols-1)
            break
        else:
            opt == 3
            break
    return newData, result

def splitData(dataset):
    print("-------------------------------------------------- \t Tahap Split Data Latih dan Data Uji\t --------------------------------------------------")
    print("Data Latih 80% & Data Uji 20%")
    trainData = dataset.sample(frac=0.8) # Mendapatkan 80% dataset yang akan digunakan sebagai data latih
    testData = dataset.drop(trainData.index) # Sisa 20% dataset akan digunakan sebagai data uji
    x_train = trainData.loc[:, trainData.columns != 'Class']
    y_train = trainData["Class"]
    x_test = testData.loc[:, testData.columns != 'Class']
    y_test = testData["Class"]
    print("Data Latih Yang Digunakan \t\t = ", x_train.shape[0])
    print("Data Uji Yang Digunakan \t\t = ", x_test.shape[0])
    print("\n")
    return x_train, y_train, x_test, y_test

def testData(dir, key, result):
    print("-------------------------------------------------- \t Tahap Pembacaan Dat Uji \t--------------------------------------------------")
    pathname, extension = os.path.splitext(dir)
    filename_train = pathname.split('/')
    dataset = pd.read_csv(dir, header=None)
    print("Dataset Yang Digunakan \t\t = ", filename_train[-1])
    print("Jumlah Dataset Yang Digunakan \t = ",dataset.shape[0])
    print("\n")
    dataset, result = preData(dataset, key, result) # Preprocessing data meliputi mengganti nama kolom, data cleaning, dan data tranform
    opt = "try"
    while(opt!= 1 or opt!=2):
        print("Pilih Jenis Model Yang Sebelumnya Berhasil Dibangun")
        print("1. Model Dengan Seleksi Fitur")
        print("2. Model Tanpa Seleksi Fitur")
        opt = int(input("Masukan Pilihan Anda (1 / 2) = "))
        if(opt == 1):
            selected_feature = pd.read_csv("./File Hasil Training/List Fitur yang Digunakan Dalam Klasifikasi.csv") # Mengambil feature yang telah disimpan dalam file eksternal
            print("-------------------------------------------------- \t Tahap Selecting Data dengan Fitur Yang Telah Ditentukan \t --------------------------------------------------")
            print("\t Fitur Yang Digunakan \t ")
            selected_feature.columns = ["Feature", "IG"]
            print(selected_feature["Feature"])
            dataTesting = dataset[selected_feature["Feature"]] # Selecting dataset dengan hanya menggunakan feature yang telah diseleksi sebelumnya
            dataTesting["Class"] = dataset["Class"]
            dataNormal, result = sigmoidNormalization(dataTesting, result) # Data testing baru tidak melalui proses seleksi fitur dan hanya akan dilakukan normalisasi data
            break
        elif(opt==2):
            dataNormal, result = sigmoidNormalization(dataset, result) # Data testing baru tidak melalui proses seleksi fitur dan hanya akan dilakukan normalisasi data
            break
        else:
            opt == 3
            break
    acc = testingData(dataNormal) # Mendapatkan nilai akurasi dari dataset yang telah diklasifikasikan dengan model yang sebelumnya telah disimpan
    print('Akurasi Model :', acc)
    return result, acc

if __name__ == "__main__":
    print("-------------------------------------------------- \t KLASIFIKASI DENGAN RANDOM FOREST \t--------------------------------------------------")
    opt = "no exit"
    while(opt != "exit"):
        print("Pilihan: ")
        print("1. Training Data")
        print("2. Testing Data")
        print("3. Keluar Program")
        answer = int(input("Ketik Pilihan 1 / 2: "))
        if(len(result) != 0):
            del result[0: len(result)]
        if(answer == 1):
            dir = './KDD-Dataset.csv' # Nama file dataset yang digunakan
            dataset, result = getData(dir, 1, result)
            x_train, y_train, x_test, y_test = splitData(dataset) # Membagi dataset menjadi 80% data latih dan 20% data uji
            result, nama_model = randomForest(x_train, y_train, x_test, y_test, result)
            kCross(dataset, nama_model) # Menghitung nilai akurasi dengan K Cross Validation
            result = treeVisualization(x_train, y_train,result, nama_model) # Memvisualisasikan tree pada model Random Forest
            print("\n")
            pathlist = "File Hasil Training"
            isdir = os.path.isdir(pathlist) 
            if(isdir == True):
                shutil.rmtree(pathlist)
                os.makedirs(pathlist)
            else:
                 os.makedirs(pathlist)
            file_source ='./'
            file_destination ='./File Hasil Training'
            i = 1
            print("File yang Berhasil Disimpan: \n")
            for data in result:
                print(str(i) +"\t" + data)
                for file in Path(file_source).glob(data):
                    shutil.move(os.path.join(file_source,file),file_destination)
                i +=1
        elif(answer == 2):
            dir = './KDD-Dataset.csv' # Nama file dataset yang digunakan
            result, acc = testData(dir, 2, result) # Menguji dengan dataset yang berbeda
            if(acc!=0):
                print("\n")
                pathlist = "File Hasil Testing"
                isdir = os.path.isdir(pathlist) 
                if(isdir == True):
                    shutil.rmtree(pathlist)
                    os.makedirs(pathlist)
                else:
                    os.makedirs(pathlist)
                file_source ='./'
                file_destination ='./File Hasil Testing'
                print("File yang Berhasil Disimpan: \n")
                i = 1
                for data in result:
                    print(str(i) +"\t" + data)
                    for file in Path(file_source).glob(data):
                        shutil.move(os.path.join(file_source,file),file_destination)
                    i +=1
        else:
            print("Tidak Ada Pilihan")
        opt = str(input('Ketik "exit" Jika Ingin Mengakhiri Program: '))

