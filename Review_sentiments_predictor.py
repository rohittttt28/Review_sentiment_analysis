from PyQt5 import QtWidgets,uic
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def review_sentiments_predictor(x):
    
    df=pd.read_csv('Amazon_Unlocked_Mobile.csv')
#Taking a small fraction of the dataset
    df = df.sample(frac=0.1, random_state=10)
    df.dropna(inplace=True)
# Remove any Neutral ratings equal to 3
    df = df[df['Rating'] != 3]
#Assigning positive and the negative rating accordingly
    df['Positively Rated'] = np.where(df['Rating'] > 3, 1, 0)


# Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['Reviews'], 
                                                        df['Positively Rated'], 
                                                        random_state=0)
    vect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)

    X_train_vectorized = vect.transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)

    predictions = model.predict(vect.transform(X_test))
    feature_names = np.array(vect.get_feature_names())

    sorted_coef_index = model.coef_[0].argsort()
#print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
#print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

#print('AUC: ', roc_auc_score(y_test, predictions))

#len(vect.get_feature_names())
#print('X_train last entry:\n\n', X_train.iloc[-1])
#print('\n\nX_train shape: ', X_train.shape)
#print some values 
#df.head()
    return model.predict(vect.transform([x]))[0]




    
def printt():
    v=call.lineEdit.text()
    v1=call.lineEdit_2.text()
    v2=call.lineEdit_3.text()
    
    
    if review_sentiments_predictor(v2)==1:
        print("Positive Response is given by {} {}".format(v,v1))
    else:
        print("Negative Response is given by {} {}".format(v,v1))


app=QtWidgets.QApplication([])
call=uic.loadUi("function.ui")



call.pushButton.clicked.connect(printt)

call.show()
app.exec()
