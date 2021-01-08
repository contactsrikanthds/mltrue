#!/usr/bin/env python
# coding: utf-8

# In[9]:


import streamlit as st

import pandas as pd 
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifer

#from sklearn.discriminant_anaysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt 
import matplotlib
import datetime
from dateutil.parser import parse
import seaborn as sns
#import statsmodels.api as sm
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_absolute_percentage_error
from itertools import combinations
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from itertools import combinations
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.metrics import roc_auc_score


import time
#import datefinder
#import dateparser
from sklearn import preprocessing 

matplotlib.use('Agg')


# In[17]:


def main():
    
    """True ML - A Truely Automated Machine Learning """
    
    st.title("True-ML")
    st.markdown(" No Code Superior Machine Learning Model Building")
    st.write('<style>h1{color: white; text-align: left;font-family: sans-serif;font-size: 45px}</style>', unsafe_allow_html=True)
    st.write('<style>h2{color: white; text-align: center;font-family: sans-serif;}</style>', unsafe_allow_html=True)

    st.write('<style>h3{color: white;text-align: left;font-family: sans-serif;font-size: 11px}</style>', unsafe_allow_html=True)
    st.write('<style>body{color: white;font-family: sans-serif;text-align: left;}</style>', unsafe_allow_html=True)
    st.write('<style>table{color: white;font-family: sans-serif;text-align: center;background-color: #eee}</style>', unsafe_allow_html=True)
    st.markdown( '<style>h2{ color: white; font-family: sans-serif;text-align: center; }</style>'
, unsafe_allow_html=True)
    #st.markdown( '<style>label{ color: white; font-family: sans-serif; }</style>',unsafe_allow_html=True) 
    

    #st.markdown('<style>markdown-text-container{font-family: cursive;}</style>', unsafe_allow_html=True)
    #st.write('<style>table{background-color: white;text-align: center;font-family: sans-serif;}</style>', unsafe_allow_html=True)
    

    page_bg_img = '<style>body{background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRWu0afLYJ6arrMqAAcw9D7mvHuU3UPs7pbFQ&usqp=CAU");background-size: cover;}</style>'

    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    #activities = ['Auto','Plots','Model Building','Prediction']
    #choice = st.sidebar.selectbox("Select Activity", activities)

    def brute_force_combinations(feature_list):
        all_combinations =[]
        for i in range(len(feature_list)):
            oc = combinations(feature_list,i+1)
            for c in oc:
                all_combinations.append(list(c))
            
        #return(all_combinations)
    def get_date(s_date):
        date_patterns = ["%m/%d/%Y","%d-%m-%Y", "%Y-%m-%d",  "%m/%d/%y",  "%d/%m/%y" ,"%d/%m/%Y",]

        for pattern in date_patterns:
            try:
                return datetime.datetime.strptime(s_date, pattern)
            except:
                return dateparser(s_date)
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    




    def result_summary(y_test,y_pred):
        confusionMatrix = confusion_matrix(y_test,y_pred)
        total = sum(sum(confusionMatrix))
        TN = confusionMatrix[0,0]
        FP = confusionMatrix[0,1]
        FN = confusionMatrix[1,0]
        TP = confusionMatrix[1,1]
    
        accuracy = (TN+TP)/total
        specificity = TN/(TN+FP)
        sensitivity = TP/(TP+FN)
        precision_0 = TN/(TN+FN)
        precision_1 = TP/(FP+TP)
        recall_0 = TN/(TN+FP)
        recall_1 = TP/(FN+TP)
        f1_score_0 = 2*precision_0 * recall_0/(precision_0+recall_0)
        f1_score_1 = 2*precision_1 * recall_1/(precision_1+recall_1)
        r2score = r2_score(y_test,y_pred)
    
        result = {"total":[total],"TN":[TN],"FP":[FP],"FN":[FN],"TP":[TP],"accuracy":accuracy,"specificity":specificity,"sensitivity":sensitivity,"precision_0":precision_0,
                  "precision_1":precision_1,"recall_0":recall_0,"recall_1":recall_1,"f1_score_0":f1_score_0,
                  "f1_score_1" :f1_score_1
                  
                 }

    
    def brute_force_combinations(feature_list):
        all_combinations =[]
        for i in range(len(feature_list)):
            oc = combinations(feature_list,i+1)
            for c in oc:
                all_combinations.append(list(c))
            
        return(all_combinations)

    
    def treatoutliers(df, columns=None, factor=1.5, method='IQR', treament='cap'):
    
        if not columns:
            columns = df.columns
        
        for column in columns:
            if method == 'STD':
                permissable_std = factor * df[column].std()
                col_mean = df[column].mean()
                floor, ceil = col_mean - permissable_std, col_mean + permissable_std
            elif method == 'IQR':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                floor, ceil = Q1 - factor * IQR, Q3 + factor * IQR
            
            if treament == 'remove':
                df = df[(df[column] >= floor) & (df[column] <= ceil)]
            elif treament == 'cap':
                df[column] = df[column].clip(floor, ceil)
                
        return df

    def brute_force_model(data_downsampled_df,features_combination,target_variable,model):
        final_frame = pd.DataFrame()
        for i in range(len(features_combination)):
            final_features = features_combination[i]
            X = data_downsampled_df[final_features]
            y = data_downsampled_df[target_variable]
            X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size = 0.20,random_state =123)
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[::,1]
            result = result_summary(y_test,y_pred)
            result['model'] = [str(model).split("(")[0]]
            result['feature_set'] = [str(final_features).replace(",",";")]
            result_frame = pd.DataFrame(result)
            final_frame = final_frame.append(result_frame)
        #return(final_frame)

    def brute_force_Reg_model(data,features_combination,model,target_variable):
         
        for i in range(len(features_combination)):
            final_features = features_combination[i]
            X = X
            y = Y
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state =123)
            row = ""
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            row += str(model).split("(")[0]
            row += "," +str(final_features).replace(",",";")
            row += "," + str(model.intercept_)
            r2score = r2_score(y_test,y_pred)
            row += "," + str(r2score)
            mse = mean_squared_error(y_test,y_pred)
            row += "," +str(mse)
            rmse = np.sqrt(mean_squared_error(y_test,y_pred))
            row += "," + str(rmse)
            f.write(row +"\n")
                                    
    def is_date(string, fuzzy=False):
    
        try: 
            parse(string, fuzzy=fuzzy)
            return True

        except ValueError:
            return False
    
    #if choice =='Auto':
        #st.subheader("Upload your Data")
        
        
    data = st.file_uploader("",type=["csv","txt"])
    
    
    if data is not None:
        data.seek(0)# change version V1.0
        df=pd.read_csv(data)
        df.fillna(df.mean(), inplace=True)
        st.table(df.head())
        #for i in df.columns:
            #st.line_chart(data=df[i])
        
        
        #start_row = st.slider('Start row', 0, n_rows - max_rows)
        #end_row = start_row + max_rows
        #df = df[start_row:end_row]
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
        #st.subheader("Data Exploration")
        #df.hist()
        #plt.show()
        #st.pyplot()

        
        #st.line_chart(df)

        #fig = plt.figure(1)
        #sns.pairplot(df) 
        #st.pyplot(fig)
        if st.button("Time Series"):
            
            st.subheader("Select Date Column")
            all_columns = df.columns
            selected_columns_time = st.multiselect("Select Columns",all_columns)
            #Non_selected_columns = df.drop(columns=selected_columns_time)
            #df['Date']= df[selected_columns_time].apply(lambda x:pd.to_datetime(x,errors = 'coerce',format = '%Y'))
            #df[selected_columns_time] = df[selected_columns_time].apply(lambda x: datetime.datetime.strptime(x,"%m/%d/%Y"))
            #for i in range(len(df)):
                #df[selected_columns_time][i]=dateparser.parse(df[selected_columns_time][i],date_formats=["%m/%d/%Y"])
            st.table(df[selected_columns_time])
            st.subheader("Select Data to forecast")
            selected_columns_1 = st.multiselect("Select Columns purpose",all_columns)
            
            
            new_df_new = df[selected_columns_time]
            try:
                new_df_new['Holiday'] = df['Holiday']
            except:
                st.write("Holiday list not included")
            #try:
                #new_df_new['Date_'] = get_date(new_df_new['Date_'])
            #except:
                #st.write("Date format is not valid")
            new_df_new['Date_'] = new_df_new[selected_columns_time].apply(lambda x: pd.to_datetime(x,errors = 'coerce', format = '%m/%d/%Y'))
            new_df_new['Day_of_week'] = new_df_new['Date_'].dt.dayofweek
            new_df_new['Month'] = new_df_new['Date_'].dt.month
            new_df_new['Day_of_month'] = new_df_new['Date_'].dt.day
            new_df_new['Day_of_Year'] = new_df_new['Date_'].dt.dayofyear
            new_df_new['Week_of_Year'] = new_df_new['Date_'].dt.weekofyear
            new_df_new['Year'] = new_df_new['Date_'].dt.year.astype(str)
            new_df_new['lag_1'] = df[selected_columns_1].shift(1)
            new_df_new['lag_2'] = df[selected_columns_1].shift(2)
            new_df_new['lag_3'] = df[selected_columns_1].shift(3)
            
            new_df_new.bfill(axis=0,inplace = True)
            new_df_new.drop('Date_',inplace=True,axis = 1)
            new_df_new.drop(columns=selected_columns_time,inplace=True, axis=1)
            new_df_new= pd.get_dummies(new_df_new)
            #datem = datetime.now().strftime("%Y")
            #new_df_new['Previous_Year'] = new_df_new['Year'].apply(lambda x: 1 if x < datem else 0 )
            #new_df_new.drop([0,1])
            #new_df_new[selected_columns_1] = df[selected_columns_1]
            
            
            st.dataframe(new_df_new)
            X=new_df_new
            Y = df[selected_columns_1]
            #st.subheader("Advanced Prediction Model Building")
            if st.button("Advanced Prediction Model"):
                all_x = brute_force_combinations(list(X.columns))
                #all_x = list(all_x)
                #st.dataframe(all_x[0])
                seed =123
                models =[]
                models.append(("Linear Regression",LinearRegression()))
                models.append(("Decision Tree",DecisionTreeRegressor()))
                models.append(("Ensemble",RandomForestRegressor()))
                models.append(("Support Vector Machine",SVR()))
                models.append(("XGBoost",XGBRegressor(objective="reg:linear")))
                all_models_adv =[]
                model_names_adv = []
                model_accuracy = []
                model_r2 =[]
                model_mse = []
                model_rmse = []
                model_mape = []
                feat = []
                for name,model in models:
                    for i in range(len(all_x)):
                        final_features = all_x[i]
                        X2 = X[final_features]
                        categorical_feature_mask = X2.dtypes==object
                        categorical_cols = X2.columns[categorical_feature_mask].tolist()
                        le = LabelEncoder()
                        X2[categorical_cols] = X2[categorical_cols].apply(lambda col: le.fit_transform(astype(str)))
                        #scaler = preprocessing.StandardScaler()
                        #X2 = scaler.fit_transform(X2)
                        #X= pd.get_dummies(X, prefix=None, prefix_sep='_', dummy_na=True, drop_first=True, dtype=None)
                        #X_train,X_test,y_train,y_test = train_test_split(X2,Y,test_size = 0.20,random_state =123)
                        X_train, X_test= np.split(X2, [int(.80 *len(X))])
                        y_train, y_test= np.split(Y, [int(.80 *len(Y))])
                        #row = ""
                        model.fit(X_train,y_train)
                        y_pred = model.predict(X_test)
                        #Accuracy = model.score(X_test,y_test)
                        #row += str(model).split("(")[0]
                        #row += "," +str(final_features).replace(",",";")
                        #row += "," + str(model.intercept_)
                        #r2score = r2_score(y_test,y_pred)
                        #row += "," + str(r2score)
                        mse = mean_squared_error(y_test,y_pred)
                        #row += "," +str(mse)
                        rmse = np.sqrt(mean_squared_error(y_test,y_pred))
                        #row += "," + str(rmse)
                        MAPE = mean_absolute_percentage_error(y_test,y_pred)
                        #accuracy_results_adv = {"model_name":name,"MSE":mse.mean(),"RMSE":rmse.mean(),"MAPE":MAPE.mean()}
                        #all_models_adv.append(accuracy_results_adv)
                        model_names_adv.append(name)
                        #model_accuracy.append(Accuracy)
                        #model_r2.append(r2score)
                        model_mse.append(mse)
                        model_rmse.append(rmse)
                        model_mape.append(MAPE)
                        feat.append(final_features)
                        #st.write(name,"|","Iteration",i,"MAPE",MAPE,"RMSE",rmse,"MSE",mse,"|","Model Done")
                        
                st.table(pd.DataFrame(zip(model_names_adv,feat,model_mape,model_mse,model_rmse),columns =['Model Name','Features','MAPE',' MSE','RMSE']).sort_values('MAPE',ascending=True))
                st.text(len(model_names_adv))       
                #result_df = pd.DataFrame(zip(data,names=["Model","Model_intercept", "Features", "r2_score", "MSE","RMSE"]))
                #st.dataframe(all_combinations.head())

        elif st.subheader("Select columns to remove"):
            ###Change Performed V 1.0
            all_columns = df.columns
            global selected_columns
            selected_columns = st.multiselect("Select Columns",all_columns)
            new_df = df.drop(columns=selected_columns)
            #st.table(new_df.columns.transpose())
            st.table(new_df.dtypes)
            #limitPer = len(new_df) * .80
            #new_df = new_df.dropna(thresh=limitPer,axis=1)
            st.table(new_df.head())
            st.subheader("Select Target columns")## Change - Left Indent V 1.0 - for whole chuck till else
            all_columns = new_df.columns
            global selected_Target_columns
            selected_Target_columns = st.multiselect(" ",all_columns)
            X = new_df.drop(columns=selected_Target_columns)
            #X.replace(np.nan,0)
            X.fillna(0)
            org_col = X.columns
            #X.drop(lambda x: x if (x.nunique()>15)
            #X = X.drop(lambda x: x.nunique()>15, axis=1)
            categorical_feature_mask = X.dtypes==object
            categorical_cols = X.columns[categorical_feature_mask].tolist()
            X.drop(X[list(X[categorical_cols].nunique().where(lambda x: x>15).dropna().index)].columns,axis=1,inplace =True)
            categorical_feature_mask = X.dtypes==object
            categorical_cols = X.columns[categorical_feature_mask].tolist()
            #st.dataframe(X)
            #st.text(categorical_cols)
            le = LabelEncoder()
            try:
                X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col.astype(str)))
            except:
                st.write("No Categories")
            #st.dataframe(X[categorical_cols])
            #X = X.drop(X[X[categorical_cols].nunique()>15].index,inplace = true)
            treatoutliers(X[X.columns.difference(categorical_cols)])
            #scaler = preprocessing.StandardScaler()
            #X = scaler.fit_transform(X)
            #X.apply(LabelEncoder().fit_transform)
            #X= pd.get_dummies(X,dummy_na=True, drop_first=True)
            
            
            st.subheader("Training Data - Input Features")
            st.table(X.head())
            
         
            
            #scaler = preprocessing.StandardScaler()
            #X = scaler.fit_transform(X)
            #X = pd.DataFrame(X,columns = org_col)
            Y = new_df[selected_Target_columns]
         
            
            st.subheader("Training Data - Target Feature")
            
            if Y.dtypes.any()==object or Y.iloc[:,0].nunique()<10:
                st.table(Y.head())
                XGmodel = XGBClassifier()
                model_XG = XGmodel.fit(X,Y)
                columns = X.columns
                ABG = pd.DataFrame()
                ABG['importance']=model_XG.feature_importances_
                ABG['Variable']= X.columns
                ABG_D = ABG.sort_values(by=['importance'],ascending = False)
                X_imp = ABG_D.head(13)
                X_imp_l = X_imp['Variable'].tolist()
                #importances = model_XG.feature_importances_
                #indices = np.argsort(importances)
                X_imp.set_index("Variable")
                st.table(X_imp)
                sns.barplot(x="importance", y="Variable", data=X_imp)
                #plt.figure(figsize=(10,10))
                #plt.rcParams["figure.figsize"] = (20,10)
                #fig = plt.figure(figsize=(18, 18))
                plt.xticks(size = 15)
                plt.yticks(size = 15)
                st.pyplot()
                #st.bar_chart(X_imp['Variable'])
                #Y = Y.apply(lambda col: le.fit_transform(col.astype(str)))
            #elif Y.iloc[:,0].nunique()<5:
                                
                #st.dataframe(Y)
                #st.subheader("Automated Traditional Model Building")
                #seed =123
                #models =[]
                #models.append(("Logistic Regression",LogisticRegression()))
                #models.append(("Decision Tree",DecisionTreeClassifier()))
                #models.append(("Ensemble",RandomForestClassifier()))
                #models.append(("Support Vector Machine",SVC()))
                #models.append(("XGBoost",XGBClassifier()))
                
                #fit =[]
        
                #model_names_log = []
                #model_accuracy_log =[]
                #model_f1_log = []
                #all_models = []
                #scoring = 'balanced_accuracy'
                #model_auc =[]

                #for name,model in models:
                    #kfold =model_selection.KFold(n_splits = 10, random_state = seed)
                    #cv_results = model_selection.cross_val_score(model,X,Y,cv=kfold,scoring = scoring)
                    #X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.20,random_state =123)
                    #model_fit = model.fit(X,Y)
                    #y_pred = model.predict(X_test)
                    #model_accuracy_l = accuracy_score(y_test,y_pred)
                    #model_f1_l = f1_score(y_test,y_pred,average ='weighted')
                    #auc =roc_auc_score(y_test, y_pred)
                    #model_names_log.append(name)
                    #model_accuracy_log.append(model_accuracy_l)
                    #model_f1_log.append(model_f1_l)
                    #model_auc.append(auc)

                    #model_names.append(name)
                    #model_mean.append(cv_results.mean())
                    #model_std.append(cv_results.std())

                    #accuracy_results = {"model_name":name,"model_accuracy":model_accuracy_l,"Standard Deviation":model_f1_l}
                    #all_models.append(accuracy_results)
                    #fit.append(model_fit)
                    
                    #model_dict = dict(zip(model_names_log,fit))
                    
                #Model_data =pd.DataFrame(zip(model_names_log,model_accuracy_log,model_f1_log),columns =['Model Name','Accuracy','F1 Score'])
                #st.subheader("Accuracy Metrics")# Change Version 1.0
                #st.dataframe(pd.DataFrame(zip(model_names_log,model_accuracy_log,model_f1_log),columns =['Model Name','Accuracy','F1 Score']).sort_values('Accuracy',ascending=False))
                #st.dataframe(Model_data)
                    #st.write(model_dict)
                #st.subheader("Advanced Prediction Model Building")
                if st.button("Advance Model Building"):
                    all_x = brute_force_combinations(list(X.columns))
                    
                    seed =123
                    models =[]
                    models.append(("Logistic Regression",LogisticRegression()))
                    models.append(("Decision Tree",DecisionTreeClassifier()))
                    models.append(("Ensemble",RandomForestClassifier(n_estimators=10)))
                    models.append(("Support Vector Machine",SVC()))
                    models.append(("XGBoost",XGBClassifier()))
                    st.write("Models Estimated",len(all_x)*len(models))
                    #st.text("Model upload Done")
                    #models.append(("Support Vector Machine",SVC()))
                    ind=[]
                    all_models_adv_l =[]
                    model_names_adv_l = []
                    final_features = []
                    accu =[]
                    f1= []
                    feat = []
                    sensitivity  = []
                    result=[]
                    model_auc = []
                    #st.line_chart(accu)
                    for name,model in models:
                        for i in range(len(all_x)):
                            if i%len(all_x)==0:                                st.write(name,len(all_x),"models completed..Processing")
                            final_features = all_x[i]
                            X1 = X[final_features]
                            #X= pd.get_dummies(X,dummy_na=True, drop_first=True)
                            X_train,X_test,y_train,y_test = train_test_split(X1,Y,test_size = 0.20,random_state =123)
                            #row = ""
                            model.fit(X_train,y_train)
                            y_pred = model.predict(X_test)
                            #y_pred_proba = model.predict_proba(X_test)[::,1]
                            accur = accuracy_score(y_test,y_pred)
                            #st.dataframe(y_test)
                            auc =roc_auc_score(y_test,y_pred)
                            ind.append(i)
                            f1r = f1_score(y_test,y_pred,average ='weighted')
                            model_names_adv_l.append(name)
                            #result = result_summary(y_test,y_pred)
                            #result['model'] = [str(model).split("(")[0]]
                            #result['feature_set'] = [str(final_features).replace(",",";")]
                            #result_frame = pd.DataFrame(result)
                            #final_frame = final_frame.append(result_frame)
                            #row += str(model).split("(")[0]
                            #row += "," +str(final_features).replace(",",";")
                            #row += "," + str(model.intercept_)
                            #accuracy  = accuracy()
                            #row += "," + str(r2score)
                            #specificity = specificity(y_test,y_pred)
                            #row += "," +str(mse)
                            #sensitivity = sensitivity(y_test,y_pred)
                            #row += "," + str(rmse)
                            #accuracy_results_adv_1 = {"model_name":name,"model_accuracy":r2score.mean(),"MSE":mse.mean(),"RMSE":rmse.mean()}
                            #all_models_adv_1.append(accuracy_results_adv)
                            #model_names_adv.append(name)
                            #accuracy.append(accuracy)
                            #specificity.append(specificity)
                            #sensitivity.append(sensitivity)
                            accu.append(accur)
                            f1.append(f1r)
                            feat.append(final_features)
                            model_auc.append(auc)
                            
                            

                    st.table(pd.DataFrame(zip(model_names_adv_l,feat,model_auc,accu,f1),columns =['Model Name','Features','AUC','Accuracy','F1-Score']).sort_values('AUC',ascending=False).head())
                    Result_data = pd.DataFrame(zip(model_names_adv_l,feat,model_auc,accu,f1),columns =['Model Name','Features','AUC','Accuracy','F1-Score']).sort_values('AUC',ascending=False)
                    #st.dataframe(result_frame)
                    #st.text(len(model_names_adv_l))
                    #st.text(Result_data[0,:])
                    #st.text(len(model_names_adv))
                    global result_df
                    result_df = pd.DataFrame(zip(model_names_adv_l,feat,model_auc,accu,f1),columns =['Model Name','Features','AUC','Accuracy','F1-Score']).sort_values('AUC',ascending=False)
                    collist = result_df.iloc[0][[1]]
                
                    #ft = result_df.head(1)['Features']
                    #em =[]
                    #for row in ft:
                        #for elem in row:
                           #em.append(elem)
                    #Xf = X[em]
                    #md = result_df.head(1)['Model Name'].tolist()
                    #ms = md[0]
                    
                    #for name,model in models:
                        #if name in ms:
                            #X_train_f,X_test_f,y_train_f,y_test_f = train_test_split(Xf,Y,test_size = 0.20,random_state =123)
                            #model.fit(X_train_f,y_train_f)

                            #st.subheader("Prediction on your data")    
                    if st.button("Predictions on your data"):
                        New_data = st.file_uploader("Upload New Dataset",type=["csv","txt"])
                        if New_data is not None:
                            New_df=pd.read_csv(New_data)
                            st.dataframe(New_df.head())
                            
                            test_Xy = New_df.drop(columns=selected_columns)
                            test_X = test_Xy[em]
                            test_X.replace(np.nan,0,inplace=True)
                            test_X.fillna(0)
                            org_col = test_X.columns
                            #X.drop(lambda x: x if (x.nunique()>15)
                            #X = X.drop(lambda x: x.nunique()>15, axis=1)
                            categorical_feature_mask = test_X.dtypes==object
                            categorical_cols = test_X.columns[categorical_feature_mask].tolist()
                            test_X.drop(test_X[list(test_X[categorical_cols].nunique().where(lambda x: x>15).dropna().index)].columns,axis=1,inplace =True)
                            categorical_feature_mask = test_X.dtypes==object
                            categorical_cols = test_X.columns[categorical_feature_mask].tolist()
                            #st.dataframe(X)
                            #st.text(categorical_cols)
                            le = LabelEncoder()
                            test_X[categorical_cols] = test_X[categorical_cols].apply(lambda col: le.fit_transform(col.astype(str)))
                            #st.dataframe(X[categorical_cols])
                            #X = X.drop(X[X[categorical_cols].nunique()>15].index,inplace = true)
                            treatoutliers(test_X[test_X.columns.difference(categorical_cols)])
                            #scaler = preprocessing.Standa
                            
                            #st.subheader(" Processed Data")
                            #st.dataframe(test_X.head(100))
                            #XY = X.drop(columns=selected_Target_columns)
                            #Model_select = st.multiselect("Select Model",model_names)
                            #for Model in model_names:
                                #if Model in model_dict:
                            #st.dataframe(model_dict)
                            #st.write(Model_select[0])
                            #nmod = model_dict[str(Model_select[0])]
                            
                            #prediction = model.predict(test_X)
                            #test_Xy['Prediction'] = prediction
                            #st.subheader(" Predictions ")
                            #st.dataframe(test_Xy)
            else:         
                st.table(Y.head())
                
                #st.subheader("Automated Traditional Model Building")
                #seed =123
                #models =[]
                #models.append(("Linear Regression",LinearRegression()))
                #models.append(("Decision Tree",DecisionTreeRegressor()))
                #models.append(("Ensemble",RandomForestRegressor()))
                #models.append(("Support Vector Machine",SVR(objective="reg:linear",random_state=123)))
                #https://scikit-learn.org/stable/modules/model_evaluation.html
                #fit =[]
        
                #model_names = []
                #model_mean =[]
                #model_rmse = []
                #model_rmse_c1 = []
                #all_models = []
                #model_mape_line =[]
                #scoring = 'r2'

                XGmodel = XGBRegressor()
                model_XG = XGmodel.fit(X,Y)
                columns = X.columns
                ABG = pd.DataFrame()
                ABG['importance']=model_XG.feature_importances_
                ABG['Variable']= X.columns
                ABG_D = ABG.sort_values(by=['importance'],ascending = False)
                X_imp = ABG_D.head(13)
                X_imp_l = X_imp['Variable'].tolist()
                #importances = model_XG.feature_importances_
                #indices = np.argsort(importances)
                X_imp.set_index("Variable")
                st.subheader("Important Factors")
                st.table(X_imp)
                st.subheader("Important Factors - Visual Representation")
                #sns.barplot(x="importance", y="Variable", data=X_imp)
                #st.pyplot()
                #st.bar_chart(X_imp)
                sns.barplot(x="importance", y="Variable", data=X_imp)
                #plt.figure(figsize=(10,10))
                #plt.rcParams["figure.figsize"] = (20,10)
                #fig = plt.figure(figsize=(18, 18))
                plt.xticks(size = 15)
                plt.yticks(size = 15)
                st.pyplot()
                #st.write(X_imp_l)
                #for name,model in models:
                    #kfold =model_selection.KFold(n_splits = 10, random_state = seed)
                    #cv_results = model_selection.cross_val_score(model,X,Y,cv=kfold,scoring = scoring)
                    #X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.20,random_state =123)
                    #cv_results = cv_results
                    #model.fit = model.fit(X,Y)
                    #y_pred = model.predict(X_test)
                    #model_names.append(name)
                   #model_mean.append(cv_results.mean())
                    #model_std.append(cv_results.std())
                    #model_mape_lin = mean_absolute_percentage_error(y_pred,y_test)
                    #model_rms = np.sqrt(mean_squared_error(y_test,y_pred))
                    #model_rmse_c = r2_score(y_test,y_pred)
                    #accuracy_results = {"model_name":name,"model_accuracy":cv_results.mean(),"Standard Deviation":cv_results.std()}
                    #all_models.append(accuracy_results)
                    #model_mape_line.append(model_mape_lin)
                    #model_rmse.append(model_rms)
                    #model_rmse_c1.append(model_rmse_c)
                    #fit.append(model.fit)
                    #model_dict = dict(zip(model_names,fit))
            
                #if st.button("Accuracy Metrics"):
                #st.dataframe(pd.DataFrame(zip(model_names,model_rmse_c1,model_rmse,model_mape_line),columns =['Model Name','R Square','RMSE ',' MAPE']))

                #st.subheader("Advanced Prediction Model Building")
                if st.button("Advanced Prediction Model"):
                    all_x = brute_force_combinations(X_imp_l)
                    #all_x = list(all_x)
                    #st.dataframe(all_x[0])
                    seed =123
                    models =[]
                    models.append(("Linear Regression",LinearRegression()))
                    models.append(("Decision Tree",DecisionTreeRegressor()))
                    models.append(("Ensemble",RandomForestRegressor()))
                    #models.append(("Support Vector Machine",SVR()))
                    models.append(("XGBoost",XGBRegressor(objective="reg:linear",random_state=123)))
                    st.write("Models Estimated",len(all_x)*len(models))
                    all_models_adv =[]
                    model_names_adv = []
                    model_r2 =[]
                    model_mape = []
                    model_rmse = []
                    #madel_mape = []
                    feat = []
                    for name,model in models:
                        for i in range(len(all_x)):
                            if i%100==0:
                                st.write(name," -",len(all_x),"+ models completed")
                            final_features = all_x[i]
                            X2 = X[final_features]
                            #X= pd.get_dummies(X, prefix=None, prefix_sep='_', dummy_na=True, drop_first=True, dtype=None)
                            X_train,X_test,y_train,y_test = train_test_split(X2,Y,test_size = 0.20,random_state =123)
                            #row = ""
                            model.fit(X_train,y_train)
                            y_pred = model.predict(X_test)
                            #row += str(model).split("(")[0]
                            #row += "," +str(final_features).replace(",",";")
                            #row += "," + str(model.intercept_)
                            r2score = r2_score(y_test,y_pred)
                            #row += "," + str(r2score)
                            mape = mean_absolute_percentage_error(y_test,y_pred)
                            #row += "," +str(mse)
                            rmse = np.sqrt(mean_squared_error(y_test,y_pred))
                            #row += "," + str(rmse)
                            #accuracy_results_adv = {"model_name":name,"model_accuracy":r2score.mean(),"MSE":mse.mean(),"RMSE":rmse.mean()}
                            #all_models_adv.append(accuracy_results_adv)
                            model_names_adv.append(name)
                            model_r2.append(r2score)
                            model_mape.append(mape)
                            model_rmse.append(rmse)
                            feat.append(final_features)
                            
                            #st.write(len(model_names_adv))
                            
                    st.table(pd.DataFrame(zip(model_names_adv,feat,model_mape,model_rmse),columns =['Model Name','Features','MAPE','RMSE']).sort_values('MAPE',ascending=True).head())
                    st.text(len(model_names_adv))
                    
                    result_df = pd.DataFrame(zip(model_names_adv,feat,model_mape,model_rmse),columns =['Model Name','Features','MAPE','RMSE']).sort_values('MAPE',ascending=True)
                    #collist = result_df.iloc[0][[1]]
                    #import pickle
                    ft = result_df.head(1)['Features']
                    em =[]
                    for row in ft:
                        for elem in row:
                           em.append(elem)
                    Xf = X[em]
                    md = result_df.head(1)['Model Name'].tolist()
                    ms = md[0]
                    mdl = 0
                    for name,model in models:
                        if name in ms:
                            X_train_f,X_test_f,y_train_f,y_test_f = train_test_split(Xf,Y,test_size = 0.20,random_state =123)
                            model.fit(X_train_f,y_train_f)
                            #filename = name + 'finalized_model.sav'
                            #pickle.dump(model, open(filename, 'wb'))

                #st.subheader("Prediction on your data")
                if st.button("Predictions on your data"):
                
                    ft = result_df.head(1)['Features']
                    em =[]
                    for row in ft:
                        for elem in row:
                           em.append(elem)
                    Xf = X[em]
                    md = result_df.head(1)['Model Name'].tolist()
                    ms = md[0]
                    mdl = 0
                    for name,model in models:
                        if name in ms:
                            X_train_f,X_test_f,y_train_f,y_test_f = train_test_split(Xf,Y,test_size = 0.20,random_state =123)
                            model.fit(X_train_f,y_train_f)
                            mdl = model
                    New_data_latest = st.file_uploader("Upload New Dataset",type=["csv","txt"])
                    if New_data_latest is not None:
                        New_df_latest =pd.read_csv(New_data)
                        st.dataframe(New_df_latest.head())
                        New_df = New_df_latest[em]
                        test_Xy = New_df.drop(columns=selected_columns)
                        ftlst = ft.tolist()
                        test_Xy
                        test_X = test_Xy[em]
                        test_X.replace(np.nan,0,inplace=True)
                        test_X.fillna(0)
                        org_col = test_X.columns
                        #X.drop(lambda x: x if (x.nunique()>15)
                        #X = X.drop(lambda x: x.nunique()>15, axis=1)
                        categorical_feature_mask = test_X.dtypes==object
                        categorical_cols = test_X.columns[categorical_feature_mask].tolist()
                        test_X.drop(test_X[list(test_X[categorical_cols].nunique().where(lambda x: x>15).dropna().index)].columns,axis=1,inplace =True)
                        categorical_feature_mask = test_X.dtypes==object
                        categorical_cols = test_X.columns[categorical_feature_mask].tolist()
                        #st.dataframe(X)
                        #st.text(categorical_cols)
                        le = LabelEncoder()
                        try:
                            test_X[categorical_cols] = test_X[categorical_cols].apply(lambda col: le.fit_transform(col.astype(str)))
                        except:
                            None
                        #st.dataframe(X[categorical_cols])
                        #X = X.drop(X[X[categorical_cols].nunique()>15].index,inplace = true)
                        treatoutliers(test_X[test_X.columns.difference(categorical_cols)])
                        #scaler = preprocessing.Standa
                        
                        #st.subheader(" Processed Data")
                        #st.dataframe(test_X.head(100))
                        #XY = X.drop(columns=selected_Target_columns)
                        #Model_select = st.multiselect("Select Model",model_names)
                        #for Model in model_names:
                            #if Model in model_dict:
                        #st.dataframe(model_dict)
                        #st.write(Model_select[0])
                        #nmod = model_dict[str(Model_select[0])]
                        
                        prediction = model.predict(test_X)
                        test_Xy['Prediction'] = prediction
                        st.subheader(" Predictions ")
                        st.dataframe(test_Xy)
                                
    
                            
                      
                
            

        
                    
                                                       
                            
                       
                
    #elif choice =='Plots':
        #st.subheader("Visualize your Data ")
        
    #elif choice =='Model Building':
        #st.subheader("Model Building")
        #if st.button("Select target columns"):
            #all_columns = new_df.columns
            #selected_Target_columns = st.multiselect("Select Columns",all_columns)
            #Y = new_df[selected_Target_columns]
            #X = new_df.drop(columns=selected_Target_columns)
            
                
        
    
    #elif choice =='Prediction':
        #st.subheader("Predict your Data")
        
if __name__ =='__main__':
    main()


# In[3]:





# In[ ]:




