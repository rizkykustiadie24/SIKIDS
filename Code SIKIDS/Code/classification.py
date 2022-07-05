from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn import tree
import timeit

"""
Fungsi randomForest: Membangun model Random Forest dengan menggunakan dataset yang telah dinormalisasi 
Fungsi kCross: Menghitung nilai akurasi dengan K Cross Validation
Fungsi testingData: Melakukan uji data dengan dataset yang lain, tetapi menggunakan model yang sebelumnya telah dibangun dan disimpan
Fungsi treeVisualization: Memvisualisasikan tree pada model Random Forest
0 : class normal
1 : class tidak normal
"""

def randomForest(x_train, y_train, x_test, y_test, result):
    print("-------------------------------------------------- \t Tahap Pelatihan Model Random Forest \t --------------------------------------------------")
    print("1. \t Membuat Model Random Forest \n")
    # Membuat model Random Forest dengan data training sebesar 80% dari dataset yang telah siap
    start_make_model = timeit.default_timer()
    rfc = RandomForestClassifier()
    rfc.fit(x_train,y_train)
    stop_make_model = timeit.default_timer()
    lama_eksekusi_model = stop_make_model - start_make_model 
    print("\t Lama Proses Pembangunan Model = ", lama_eksekusi_model)
    print("2. \t Melakukan Prediksi Terhadap Data Uji \n")
    # Melakukan prediksi dengan data testing sebesar 20% dari dataset
    start_prediksi_model = timeit.default_timer()
    y_pred = rfc.predict(x_test)
    stop_prediksi_model = timeit.default_timer()
    lama_eksekusi_prediksi = stop_prediksi_model - start_prediksi_model
    print("\t Lama Proses Prediksi Data dengan Model = ", lama_eksekusi_prediksi) 

    print("3. \t Menyimpan Model Random Forest \n")
    # Menyimpan model Random Forest agar dapat digunakan untuk klasifikasi pada data lainnya
    nama_model = 'Model-Random-Forest.joblib'
    joblib.dump(rfc, "./"+nama_model)
    print("\t "+ nama_model+ " Berhasil Disimpan!!!")
    result.append(nama_model)

    print("4. \t Menghitung Nilai Akurasi dan Confussion Matrix \n")
    # Menghitung akurasi model dan confussion matrix
    print('Akurasi Model : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix\n\n', cm)
    print(classification_report(y_test, y_pred))
    print("\n")
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True)
    nama_file = 'Confussion Matrix.png'
    print(nama_file+ "Berhasil Disimpan!!! \n")
    plt.savefig(nama_file)
    result.append(nama_file)
    return result, nama_model

def kCross(dataset, nama_model):
    print("-------------------------------------------------- \t Tahap K Cross Validation \t --------------------------------------------------")
    print("K Yang Digunakan = 5 \n")
    # Melakukan testing data untuk mendapatkan nilai akurasi dengan K Cross Validation, K yang dipilih adalah 5
    k = 5
    x = dataset.loc[:, dataset.columns != 'Class']
    y = dataset["Class"]
    kf = KFold(n_splits=k, random_state=None)
    acc_score = []
    loaded_rfc = joblib.load("./"+nama_model)

    # Melakukan prediksi/testing untuk setiap data yang berbeda (pada setiap k)
    for train_index , test_index in kf.split(x):
        x_train , x_test = x.iloc[train_index,:],x.iloc[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]
        pred_values = loaded_rfc.predict(x_test)
        acc = accuracy_score(pred_values , y_test)
        acc_score.append(acc)
    avg_acc_score = sum(acc_score)/k
    # Menghitung akurasi tiap fold dan juga rata-rata akurasinya
    print('Nilai Akurasi Pada Tiap Fold - {}'.format(acc_score))
    print('Rata-rata Nilai Akurasi : {}'.format(avg_acc_score))
    print("\n")

# Melakukan testing dengan dataset lainnya
def testingData(dataset):
    print("-------------------------------------------------- \t Tahap Pengujian Data dengan Model Random Forest Yang Telah Disimpan \t --------------------------------------------------")
    x = dataset.loc[:, dataset.columns != 'Class']
    y = dataset["Class"]
    print("1. \t Load Model Random Forest \n")
    try:
        loaded_rfc = joblib.load("./File Hasil Training/Model-Random-Forest.joblib")
        print("2. \t Melakukan Prediksi Terhadap Data Uji \n")
        start_prediksi_model = timeit.default_timer()
        pred_values = loaded_rfc.predict(x)
        stop_prediksi_model = timeit.default_timer()
        lama_eksekusi_prediksi = stop_prediksi_model - start_prediksi_model
        print("\t Lama Proses Prediksi Data dengan Model = ", lama_eksekusi_prediksi) 
        print("3. \t Menghitung Nilai Akurasi\n")
        acc = accuracy_score(pred_values , y)
    except FileNotFoundError:
        print("File Tidak Ditemukan")
        acc = 0
    return acc

def treeVisualization(x_train, y_train,result, nama_model):
    print("-------------------------------------------------- \t Tahap Visualisasi Tree \t --------------------------------------------------")
    # Kelas Data
    cls = ["Normal", "Tidak Normal"]
    # Load model Random Forest yang sebelumnya telah disimpan
    loaded_rfc = joblib.load("./"+nama_model)
    # Mendapatkan jumlah estimators dan kedalama tree, di mana estimator yang digunakan pada model adalah 100
    print("N Estimators = ", len(loaded_rfc.estimators_))
    print("Depth = ", loaded_rfc.estimators_[0].tree_.max_depth)
    # Membuat plot untuk memvisualisasikan tree
    plt.figure(figsize=(6,6))
    # dp = int(input("Ketikan Kedalaman Tree Yang Akan Divisualisasikan = "))
    # Ubah max_depth sesuai dengan keinginana seberapa dalam ingin memvisualisasikan tree, apabila memvisualisasikan semua maka hilangkan max_depth
    tree.plot_tree(loaded_rfc.estimators_[0], feature_names=x_train.columns, class_names=cls, filled=True, max_depth=3, fontsize=6) 
    plt.title("Random Forest Visualization")
    name_file = 'Visualisasi Tree.png'
    print(name_file+ " Berhasil Disimpan!!! \n")
    plt.savefig('Visualisasi Tree.png')
    result.append(name_file)
    print("-------------------------------------------------- \t Tahapan Selesai \t --------------------------------------------------")
    return result