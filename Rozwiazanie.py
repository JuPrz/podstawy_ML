import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder 
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

#zaczytanie danych z pliku
df_train = pd.read_csv("podstawy_ML/train.csv" , sep = "," , encoding = 'utf-8')
# #sprawdzenie rozmiaru
print(df_train.shape)
# #sprawdzenie nazw kolumn i ich typów
print(df_train.info())
# #wyświetl część tabeli
print(df_train.head())
# #usunięcie wierszy z duplikatami id
df_train.drop_duplicates(subset="ID", inplace=True) 

# #nazwy kolumn dla danych docelowo numerycznych
FeaturesToConvert = ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment',
                    'Changed_Credit_Limit', 'Outstanding_Debt', 'Amount_invested_monthly',
                     'Monthly_Balance']

# #nazwy kolumn dla danych kategorycznych 

# liczności danych kategorii
sns.countplot(df_train['Credit_Score'])

#sprawdzenie danych
for i in df_train.columns: 
	print(df_train[i].value_counts()) 
	print('*'*50)
 
# # wyświetl liczbę pustych wartosci
print(df_train.isnull().sum().sort_values(ascending=False))

########## dane numeryczne #########################
# usuń zbędne znaki '-’ , '_'
for feature in FeaturesToConvert: 
	df_train[feature] = df_train[feature].str.strip('-_')
# puste kolumny zastąp NAN 
for feature in FeaturesToConvert: 
	df_train[feature] = df_train[feature].replace({'':np.nan}) 
# zmien typ zmiennych ilościowych
for feature in FeaturesToConvert: 
	df_train[feature] = df_train[feature].astype('float64')
 
# #uzupełnij braki średnią
df_train['Monthly_Inhand_Salary']= df_train['Monthly_Inhand_Salary'].fillna(method='pad')
df_train['Monthly_Balance']= df_train['Monthly_Balance'].fillna(method='pad')
df_train['Type_of_Loan']= df_train['Type_of_Loan'].fillna(method='ffill')
df_train['Credit_History_Age']= df_train['Credit_History_Age'].fillna(method='pad')
df_train['Num_of_Delayed_Payment']= df_train['Num_of_Delayed_Payment'].fillna(method='pad')
df_train['Amount_invested_monthly']= df_train['Amount_invested_monthly'].fillna(method='pad')
df_train['Changed_Credit_Limit']= df_train['Changed_Credit_Limit'].fillna(method='pad')
df_train['Num_Credit_Inquiries']= df_train['Num_Credit_Inquiries'].fillna(method='pad')

print(df_train.isnull().sum().sort_values(ascending=False))

# #zastąpienie nierealnych wartości medianą
for i in df_train.Age.values:
    if (i > 118 or i< 0 ):
        df_train.Age.replace(i, np.median(df_train.Age), inplace=True)
        
# ############ zmienne kategoryczne #####################
# # stwórz obiekt enkodera
le = LabelEncoder()

CatFeatures = ['Month','Occupation', 'Credit_Mix', 'Payment_of_Min_Amount',
               'Payment_Behaviour', 'Credit_Score']

# # zakoduj etykiety słowne numerycznymi

df_train[CatFeatures] = df_train[CatFeatures].apply(LabelEncoder().fit_transform)

# # usuń nieistotne kolumny
irrelevant = ['Type_of_Loan','ID','Customer_ID','Name','SSN']
df_train = df_train.drop(irrelevant, axis=1)

#######################################################
# sprawdź transformacje
print(df_train.shape)
print('\n**********************************\n')
print(df_train.info())
print('\n**********************************\n')
print(df_train.describe().transpose())
print('\n**********************************\n')
######################################################

df_train['Credit_History_Age'].head()

df_train['Credit_History_Age'] = (12 * (df_train["Credit_History_Age"].str.split( " ", expand=True)[0])+(df_train["Credit_History_Age"].str.split( " ", expand=True)[3]))

df_train['Credit_History_Age'] = df_train['Credit_History_Age'].astype(float)


# Drop outlier by IQR calculation
Q1 = df_train.Annual_Income.quantile(0.25)
Q3 = df_train.Annual_Income.quantile(0.75)
IQR = Q3 - Q1
df_train = df_train.drop(df_train.loc[df_train['Annual_Income'] > (Q3 + 1.5 *IQR)].index)
df_train = df_train.drop(df_train.loc[df_train['Annual_Income'] < (Q1 -1.5 * IQR)].index)

scaler = StandardScaler() #scaler = MinMaxScaler() # stanadryzacja bierze 0 za punkt wyjscia to jest np że najwięcej est 40 latków więc ich wartość jest na 0  - odejmuje od danej zmienna ze zbioru i dzieli przez odchylenie standardowe 
col_float = ['Age','Annual_Income','Delay_from_due_date','Num_of_Delayed_Payment','Outstanding_Debt','Credit_History_Age','Total_EMI_per_month','Monthly_Balance']
for i in df_train[col_float]:
    df_train[i] = scaler.fit_transform(df_train[[i]])
print(df_train.head())


# korelecaja wartość do wartości
# sns.set(style="white")
# fig, ax = plt.subplots(figsize=(10, 8))
# sns.heatmap(df_train[['Age','Annual_Income']].corr(), annot=True, linewidths=0.1, cmap='Blues', ax=ax)
# ax.set_title('Age Correlation')
# plt.show()

# sns.set(style="white")
# fig, ax = plt.subplots(figsize=(10, 8))
# sns.heatmap(df_train.corr(), annot=True, linewidths=0.1, cmap='Blues', ax=ax)
# ax.set_title('Numerical Features Correlation')
# plt.show()


pd.DataFrame(abs(df_train.corr()['Credit_Score'].drop('Credit_Score')*100).sort_values(ascending=False)).plot.bar(figsize = (10,8))
plt.show()

df_train = df_train.drop(['Num_of_Loan','Num_Credit_Card'], axis=1) # axis = 1 to kolumny - axis  = 0 to są wiersze 