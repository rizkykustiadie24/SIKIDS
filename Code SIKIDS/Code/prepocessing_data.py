from numpy import append
import pandas as pd
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=UserWarning)

"""
Fungsi preData: Proses preprocessing data meliputi mengganti nama kolom, mengganti data dengan tipe kategori ke numerik, data cleaning, data transform, dan proses K Means Clustering 
Fungsi renameColumn: dataset yang digunakan belum memiliki nama kolom, maka akan diberikan nama kolom yang sesuai dengan nama feature yang seharusnya
Fungsi cekNull: Proses data cleaning, yaitu mengecek apakah ada data dengan value 0/Nan sehingga data tersebut akan dihapus
Fungsi getClass: Mengkategorikan class ke dalam 2 jenis, yaitu Normal dan Tidak Normal
Fungsi changeToNumerik: Mengganti data dengan tipe data categorical menjadi numerik
Fungsi elbowMethod: Metode yang digunakan untuk menemukan K optimal yang akan digunakan saat clustering dengan KMeans 
Fungsi kMeans: Mengelompokkan data dengan metode KMeans clustering
"""

def preData(dataset,key, result):
    print("-------------------------------------------------- \t Tahap Preprocessing Data \t --------------------------------------------------")
    print("1. \t Mengganti Nama Kolom \n")
    dataset = renameColumn(dataset) # Mengganti nama kolom dengan nama feature yang seharusnya
    print("2. \t Mengategorikan Class Menjadi Normal dan Tidak Normal \n")
    dataset = getClass(dataset) # Mengategorikan class ke dalam 2 janis, Normal dan Tidak Normal
    print("3. \t Mengecek Data Null \n")
    dataset = cekNull(dataset) # Data cleaning
    print("4. \t Mengganti Data Categorical Menjadi Numerik")
    a, b, c, d = changeToNumerik(dataset['protocol_type'], dataset['Service'], dataset['Flag'], dataset['Class']) # Mengganti data categorical menjadi numerik
    dataset['protocol_type'], dataset['Service'], dataset['Flag'], dataset['Class'] = a, b, c, d
    name_file = 'Dataset Final.csv'
    print("\t "+ name_file+ " Berhasil Disimpan!!!")
    print("\n")
    dataset.to_csv(name_file)
    result.append(name_file)
    if(key == 1):   
        print("5. \t Mencari K Optimal Yang Akan Digunakan Pada K-Means Clustering")
        print("\t K Yang Digunakan = 6")
        result = elbowMethod(dataset,result) # Mencari K optimal yang akan digunakan pada K Means clustering
        print("\n6. \t Dikritisasi dengan Menggunakan K-Means Clustering")
        dataset = kMeans(dataset)
    elif(key == 2):
        print("5. \t Dikritisasi dengan Menggunakan K-Means Clustering \n")
        print("\t K Yang Digunakan = 6")
        dataset = kMeans(dataset)
    name_file = 'Dataset Hasil K-Means.csv'
    result.append(name_file)
    print("\t "+ name_file+ " Berhasil Disimpan!!!\n")
    dataset.to_csv(name_file) # Menyimpan dataset setelah proses K Means
    return dataset, result

def renameColumn(data):
    """
    feature pada dataset mulanya masih dipisahkan dengan koma (,) untuk itu dataset akan dibagi 
    ke dalam kolom sesuai dengan banyak featurenya. selain itu juga akan diberikan nama kolom sesuai nama featurenya 
    dan mengganti tipe data feature sesuai dengan yang seharusnya.
    """
    data.columns = ["Data"]
    data['Data'].astype(str)
    data = data['Data'].str.split(',', expand=True)
    data.columns = [
        "Duration", 
        "protocol_type", 
        "Service", 
        "Flag", 
        "src_bytes",
        "dst_bytes",
        "Land",
        "wrong_fragment",
        "Urgent",
        "Hot",
        "num_failed_logins",
        "logged_in",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "is_host_login",
        "is_guest_login",
        "Count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
        "Sub Class",
        "Difficulty Level"
    ]
    data = data.astype(
        {"Duration": int, 
        "src_bytes": int, 
        "dst_bytes":int,
        "Land": int,
        "wrong_fragment": int,
        "Urgent": int,
        "Hot": int,
        "num_failed_logins": int,
        "logged_in": int,
        "num_compromised": int,
        "root_shell": int,
        "su_attempted": int,
        "num_root": int,
        "num_file_creations": int,
        "num_shells": int,
        "num_access_files": int,
        "num_outbound_cmds": int,
        "is_host_login": int,
        "is_guest_login": int,
        "Count": int,
        "srv_count": int,
        "dst_host_count": int,
        "dst_host_srv_count": int,
        "Difficulty Level": int
        })
    data = data.astype(
        {"serror_rate": float,
        "srv_serror_rate": float,
        "rerror_rate": float,
        "srv_serror_rate": float,
        "srv_rerror_rate": float,
        "same_srv_rate": float,
        "diff_srv_rate": float,
        "srv_diff_host_rate": float,
        "dst_host_same_srv_rate": float,
        "dst_host_diff_srv_rate": float,
        "dst_host_same_src_port_rate": float,
        "dst_host_srv_diff_host_rate": float,
        "dst_host_serror_rate": float,
        "dst_host_srv_serror_rate": float,
        "dst_host_rerror_rate": float,
        "dst_host_srv_rerror_rate": float
        })
    data = data.astype(
        {"Sub Class": 'string',
        "protocol_type": 'string',
        "Service": 'string',
        "Flag": 'string'
        })
    return data

def cekNull(data):
    data.dropna(axis=0) # Menghapus data yang mengandung value null/nan
    return data

def getClass(data):
    data.loc[(data['Sub Class'] != 'normal'), 'Class'] = 'Tidak Normal' # Apabila value pada kolom sub class bukan normal, maka value dalam kolom class adalah tidak normal
    data.loc[(data['Sub Class'] == 'normal'), 'Class'] = 'Normal' # Sedangkan apabila value pada kolom sub class adalah normal, maka value dalam kolom class adalah normal
    data = data.drop(columns=['Sub Class', 'Difficulty Level']) # Menghapus kolom sub class dan difficulity level karena tidak dibutuhkan
    return data

def changeToNumerik(a, b, c, d):
    # Mengkategorikan data dengan tipe categorical menjadi numerik
    le = LabelEncoder()
    a = le.fit_transform(a)
    b = le.fit_transform(b)
    c = le.fit_transform(c)
    d = le.fit_transform(d)
    return a, b, c, d

def elbowMethod(data, result):
    datakMeans = data.iloc[:, 0:41] # Data yang akan dikelompokkan adalah data selain kolom class
    categorical_features = ['protocol_type', 'Service','Flag']
    # Untuk menggunakan fitur categorical, kita perlu mengonversi fitur ini ke biner menggunakan panda get dummies.
    for col in categorical_features:
        dummies = pd.get_dummies(datakMeans[col], prefix=col)
        datakMeans = pd.concat([datakMeans, dummies], axis=1)
        datakMeans.drop(col, axis=1, inplace=True)
    # Melakukan feature scaling dengan menggunakan MinMax
    mms = MinMaxScaler()
    mms.fit(datakMeans)
    data_transformed = mms.transform(datakMeans)
    Sum_of_squared_distances = []
    K = range(1,10)
    # Untuk setiap nilai k, akan diinisialisasi k-means 
    # dan menggunakan atribut inersia untuk mengidentifikasi jumlah kuadrat jarak sampel ke pusat cluster terdekat.
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data_transformed)
        Sum_of_squared_distances.append(km.inertia_)
    # Plot jumlah kuadrat jarak untuk setiap k
    plt.figure(figsize=(16,8))
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    nama_file = 'Diagram Hasil Elbow Method.png'
    plt.savefig(nama_file)
    print("\t "+ nama_file+ " Berhasil Disimpan!!!")
    result.append(nama_file)
    return result

def kMeans(data):
    # Data yang akan dikelompokan adalah data selain kolom class
    datakMeans = data.iloc[:, 0:41]
    # Melakukan dikritisasi dengan k = 6
    disc = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='kmeans')
    disc.fit(datakMeans)
    datakMeans = disc.transform(datakMeans)
    datakMeans = pd.DataFrame(datakMeans)
    # Memberikan nama kolom kembali sesuai dengan nama featurenya
    datakMeans.columns = [
        "Duration", 
        "protocol_type", 
        "Service", 
        "Flag", 
        "src_bytes",
        "dst_bytes",
        "Land",
        "wrong_fragment",
        "Urgent",
        "Hot",
        "num_failed_logins",
        "logged_in",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "is_host_login",
        "is_guest_login",
        "Count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate"
    ]
    # Menambahkan kolom class ke dalam data 
    datakMeans["Class"] = data.iloc[:, 41]
    return datakMeans