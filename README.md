# Eksperimen_SML_Eliv-Kurniawan

Eksperimen preprocessing dataset **Titanic - Machine Learning from Disaster** untuk submission _Membangun Sistem Machine Learning_ (SMSML)

## 📁 Struktur Folder

```
Eksperimen_SML_Eliv-Kurniawan/
├── .github/
│   └── workflows/
│       └── preprocessing.yml          # GitHub Actions otomasi preprocessing
├── data_titanic_raw/                  # Dataset mentah (Kaggle Titanic)
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
├── preprocessing/
│   ├── Eksperimen_Eliv-Kurniawan.ipynb     # Notebook EDA + preprocessing manual
│   ├── automate_Eliv-Kurniawan.py          # Script otomasi preprocessing
│   └── dataset_preprocessing/              # (auto-generated) hasil preprocessing
│       ├── titanic_clean.csv
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## 🧪 Tahapan Preprocessing

1. **Missing values**: `Age` → median, `Embarked` → modus, `Cabin` → drop.
2. **Hapus duplikat**.
3. **Feature engineering**: `Title` (dari Name), `FamilySize` (SibSp + Parch + 1), `IsAlone`.
4. **Drop kolom non-informatif**: `PassengerId`, `Name`, `Ticket`.
5. **Encoding kategorikal**: `Sex` (binary), `Embarked` & `Title` (one-hot).
6. **Outlier handling**: IQR capping pada `Fare`.
7. **Standarisasi**: `StandardScaler` pada fitur numerik.
8. **Train/test split**: 80/20 stratified by `Survived`.

## 🚀 Cara Menjalankan Lokal

```bash
# 1. Install dependency
pip install -r requirements.txt

# 2. Jalankan preprocessing otomatis
python preprocessing/automate_Eliv-Kurniawan.py \
    --input data_titanic_raw/train.csv \
    --output-dir preprocessing/dataset_preprocessing

# 3. Atau buka notebook EDA
jupyter notebook preprocessing/Eksperimen_Eliv-Kurniawan.ipynb
```

## ⚙️ GitHub Actions

Workflow `.github/workflows/preprocessing.yml` akan ter-trigger saat:

- `push` ke branch `main` yang mengubah `data_titanic_raw/**`, script otomasi, atau workflow itu sendiri.
- `pull_request` ke `main`.
- Manual via **workflow_dispatch**.
- Terjadwal harian (cron `0 0 * * *`).

Hasil yang dikembalikan setiap kali workflow berjalan:

- ✅ **Artifact** `titanic-preprocessed-dataset` (dapat di-download dari tab Actions).
- ✅ **Auto-commit** dataset hasil preprocessing ke folder `preprocessing/dataset_preprocessing/` di repository.

## 📦 Dataset

Dataset Titanic dapat diunduh dari [Kaggle - Titanic Competition](https://www.kaggle.com/c/titanic/data) dan diletakkan di folder `data_titanic_raw/`.

## 👤 Author

**Eliv Kurniawan** — Submission SMSML Dicoding.
