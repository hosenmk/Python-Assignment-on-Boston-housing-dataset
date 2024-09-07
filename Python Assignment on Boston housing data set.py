#import libraries for data manipulation
import pandas as pd
import numpy as np

#import libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import ProbPlot

#import libraries for building linear regression model
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

#import library for preparing data
from sklearn.model_selection import train_test_split

#import library for data preprocessing
from sklear.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

#from sklearn import datasets
#boston = datasets.load_boston()

data = pd.read_csv("Bostonhousing.csv")
data = data.rename(columns={'medv': 'Price'})
df =data
df.head()

#plotting all the columns to look at their distribution
for i in df.columns:
    plt.figure(figsize = (7, 4))
    sns.histplot(data = df, x = i, kde = True)
    plt.show()

#for bar diagram
objects = ('High', 'Low')
x_pos = np.arrange(len(objects))
status_fre = [296, 215]
plt.bar(x_pos, status_fre)
plt.xticks(x_pos, objects)
plt.ylabel('No of house')
plt.title('House price')
plt.show()


#Correlation matrix heatmap
correlation_matrix = df.corr()
plt.figure(figsize = (12, 9))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Boston Housing Data')
plt.show()

#Construct pair plots with different colors for different house prices.
df['Price_Category'] = df['Price'].apply(lambda x: 'High' if x > 20 else 'Low')
#pair plot
sns.pairplot(df, hue='Price_Category', plot_kws={'alpha':0.5}, palette={'High': 'blue', 'Low':'red'})
plt.subtitle('Pair Plots of Variables in Boston Housing Data', y=1.02)
plt.show()


#Create price catogories
data['PriceCat'] = 0
data.loc[data['medv'] > 20, 'PriceCat'] = 1

x = data.drop('PriceCat', axis=1)
y = data['PriceCat']

scaler = StandardScaler()

x = scaler.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)

logreg = LogisticRegression(max_iter=1000, solver='liblinear')
logreg.fit(x_train, y_train)

print(f'Intercept: {logreg.intercept_[0]:.3f}')
print('Coefficients:')
for i, coef in enumerate(logreg.coef_[0]):
    print(f,' - x{i}: {coef:.3f}')
print('Positive coefficients increase log odds of high prices')



#Create target
data['PriceCat'] = (data['medv'] > 20).astype(int)

#split data
x = data.drop('PriceCat', axis=1)
y = data['PriceCat']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=113)

#Scale features
scaler = StnadardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Logistic Regression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
print('Logistic Regression')
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=113)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)
y_pred=logmodel.predict(x_test)
fromsklearn.metrics import classification_report, confusion matrix, accuracy_score, precision_score
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))
print('Accuracy : ', accuracy_score(y_test, y_pred))
print('Precision : ', precision_score(y_test, y_pred))

specificity = cm[0,0]/(cm[0,0]+cm[0,1])
print('Specificity : ', specificity)

sensivity = cm[1,1]/(cm[1,0]+cm[1,1])
print('Sensivity : ', sensivity)


probs = logmodel.predict_proba(x_test)
predLR = probs[:,1]
from sklearn.metrics import roc_curve, roc_auc_score
auc = roc_auc_score(y_test, predLR)
print('LR AUC:', auc)

fpr, tpr, thresholds = roc_curve(y_test, predLR)
plt.figure
lw = 2
plt.plot(fpr, tpr, color='Green', lw=lw, label='LR (AUC = %0.4f)' % auc)
plt.plot([0,1], [0,1], color='red', lw-lw, linestyle='_')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


#Fit Decision tree
dt - DecisionTreeClassifier()
dt.fit((x_train, y_train))

#check train and test accuracy
train_acc = dt.score(x_train, y_train)
text_acc = dt.score(x_test, y_test)
print('Training accuracy:', train_acc)
print('Testing accuracy:', test_acc)

#Tune max depth
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(x_train, y_train)

test_acc_new = dt.score(x_test, y_test)
print('New testing accuracy:', test_acc_new)

#Learning curves
train_sizes, train_scores, test_scores = learning_curve(dt, x, y, train_sizes=np.linspace(0.1, , 10), scoring='accuracy', cv=5)

plt.plot(train_sizes, test_scores.mean(1), c='b')
plt.title("Decision Tree Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.show()


probsDT = DTclf.predict_proba(x_test)
probsDT = probsDT[:, 1]
auc2 = roc_auc_score(y_test, probsDT)
print('DT AUC2:', auc2)


fpr2, tpr2, thresholds2 = roc_curve(y_test, probsDT)
plt.figure()
lw = 2

plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='_')
plt.plot(fpr2, tpr2, color='green',, lw=lw, label='DT()AUC = %0.4f' % auc2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic')
plt.legend(log="lower right")
plt.show()

#Random Forest
from sklearn.enseble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
print('Random Forest')

#Confusion Matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix")
print(cnf_matrix)

#ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
print("AUC:", auc)

#Other metrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
ppv = tp / (tp + fp)
print("Predictive Value Negative:", ppv)

npv = tn / (tn + fn)
print("Predictive Value Negative:", npv)

accuracy = (tp + tn) / (tp + fp + fn + tn)
print("Accuracy:", accuracy)

sensitivity = tp / (tp + fn)
print("Sensitivity:", sensitivity)

specificity = tn / (tn + fp)
print("Specificity:", specificity)



RFclf.fit(x_train, y_train)
probsFR = RFclf.predict_proba(x_test)
probsRF = probsRF[:, 1]
from sklearn.metrics import roc_curve, roc_auc_score
auc3 = roc_auc_score(y_test, probsRF)
print('RF AUC3:', auc3)


fpr3, tpr3, thresholds3 = roc_curve(y_test, probsRF)
plt.figure()
lw = 2
plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='_')
plt.plot(fpr3, tpr3, color='green', lw=lw, label,='RF(AUC = %0.4f)' % auc3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

#plt.title('Receiver operating charactertitis')
plt.legend(loc="lower right")
plt.show()













