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
import statsmodels.api as sm
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
from itertools import combinations
import time
#import datefinder
#import dateparser
from sklearn import preprocessing 

matplotlib.use('Agg')


# In[17]:


def main():
    
    """True ML - A Truely Automated Machine Learning """
    
    st.title("True-ML")
    st.subheader(" No Code Machine Learning Model Building")
    st.write('<style>h1{color: white; background: DarkBlue; text-align: center;}</style>', unsafe_allow_html=True)
    st.write('<style>h3{color: white;text-align: center;background: CornflowerBlue;}</style>', unsafe_allow_html=True)
    st.markdown('<style>body{color: Navy;}</style>', unsafe_allow_html=True)    
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
        df=pd.read_csv(data)
        df.fillna(df.mean(), inplace=True)
        st.dataframe(df.head(100))
        #start_row = st.slider('Start row', 0, n_rows - max_rows)
        #end_row = start_row + max_rows
        #df = df[start_row:end_row]
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
        if st.checkbox("TIme Series"):
            st.subheader("Select Date Column")
            all_columns = df.columns
            selected_columns_time = st.multiselect("Select Columns",all_columns)
            #Non_selected_columns = df.drop(columns=selected_columns_time)
            #df['Date']= df[selected_columns_time].apply(lambda x:pd.to_datetime(x,errors = 'coerce',format = '%Y'))
            #df[selected_columns_time] = df[selected_columns_time].apply(lambda x: datetime.datetime.strptime(x,"%m/%d/%Y"))
            #for i in range(len(df)):
                #df[selected_columns_time][i]=dateparser.parse(df[selected_columns_time][i],date_formats=["%m/%d/%Y"])
            st.dataframe(df[selected_columns_time])
            st.subheader("Select Data to forecast")
            selected_columns_1 = st.multiselect("Select Columns purpose",all_columns)
            
            new_df_new = df[selected_columns_time]
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
            st.subheader("Advanced Prediction Model Building")
            if st.checkbox("Advanced Prediction Model"):
                all_x = brute_force_combinations(list(X.columns))
                #all_x = list(all_x)
                #st.dataframe(all_x[0])
                seed =123
                models =[]
                models.append(("Linear Regression",LinearRegression()))
                models.append(("Decision Tree",DecisionTreeRegressor()))
                models.append(("Ensemble",RandomForestRegressor()))
                #models.append(("Support Vector Machine",SVR()))             
                all_models_adv =[]
                model_names_adv = []
                model_accuracy = []
                model_r2 =[]
                model_mse = []
                model_rmse = []
                model_mape = []
                feat = []
                for i in range(len(all_x)):
                    for name,model in models:
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
                        Accuracy = model.score(X_test,y_test)
                        #row += str(model).split("(")[0]
                        #row += "," +str(final_features).replace(",",";")
                        #row += "," + str(model.intercept_)
                        r2score = r2_score(y_test,y_pred)
                        #row += "," + str(r2score)
                        mse = mean_squared_error(y_test,y_pred)
                        #row += "," +str(mse)
                        rmse = np.sqrt(mean_squared_error(y_test,y_pred))
                        #row += "," + str(rmse)
                        MAPE = mean_absolute_percentage_error(y_test,y_pred)
                        accuracy_results_adv = {"model_name":name,"model_accuracy":r2score.mean(),"MSE":mse.mean(),"RMSE":rmse.mean(),"MAPE":MAPE.mean()}
                        all_models_adv.append(accuracy_results_adv)
                        model_names_adv.append(name)
                        model_accuracy.append(Accuracy)
                        model_r2.append(r2score)
                        model_mse.append(mse)
                        model_rmse.append(rmse)
                        model_mape.append(MAPE)
                        feat.append(final_features)
                        
                st.dataframe(pd.DataFrame(zip(model_names_adv,feat,model_accuracy,model_mse,model_rmse),columns =['Model Name','Features','Accuracy',' MSE','RMSE']).sort_values('Accuracy',ascending=False))
                st.text(len(model_names_adv))       
                #result_df = pd.DataFrame(zip(data,names=["Model","Model_intercept", "Features", "r2_score", "MSE","RMSE"]))
                #st.dataframe(all_combinations.head())

        else:
                        
            if st.checkbox("Select ID columns to remove"):
                all_columns = df.columns
                global selected_columns
                selected_columns = st.multiselect("Select Columns",all_columns)
                new_df = df.drop(columns=selected_columns)
                st.dataframe(new_df)

            if st.checkbox("Select Target columns"):
                all_columns = new_df.columns
                global selected_Target_columns
                selected_Target_columns = st.multiselect(" ",all_columns)
                X = new_df.drop(columns=selected_Target_columns)
                X.replace(np.nan,0)
                
                #X.drop(lambda x: x if (x.nunique()>10))
                
                categorical_feature_mask = X.dtypes==object
                categorical_cols = X.columns[categorical_feature_mask].tolist()
                le = LabelEncoder()
                X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col.astype(str)))
                #scaler = preprocessing.StandardScaler()
                #X = scaler.fit_transform(X)
                #X.apply(LabelEncoder().fit_transform)
                #X= pd.get_dummies(X,dummy_na=True, drop_first=True)
                st.subheader("Training Data - Input Features")
                st.dataframe(X)
                Y = new_df[selected_Target_columns]
                st.subheader("Training Data - Target Feature")
                if Y.dtypes.any()==object or Y.iloc[:,0].nunique()<5:
                    st.dataframe(Y)
                #elif Y.iloc[:,0].nunique()<5:
                                    
                    #st.dataframe(Y)
                    st.subheader("Automated Traditional Model Building")
                    seed =123
                    models =[]
                    models.append(("Logistic Regression",LogisticRegression()))
                    models.append(("Decision Tree",DecisionTreeClassifier()))
                    models.append(("Ensemble",RandomForestClassifier()))
                    #models.append(("Support Vector Machine",SVC()))
                    fit =[]
            
                    model_names_log = []
                    model_accuracy_log =[]
                    model_f1_log = []
                    all_models = []
                    scoring = 'balanced_accuracy'

                    for name,model in models:
                        #kfold =model_selection.KFold(n_splits = 10, random_state = seed)
                        #cv_results = model_selection.cross_val_score(model,X,Y,cv=kfold,scoring = scoring)
                        X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.20,random_state =123)
                        model_fit = model.fit(X,Y)
                        y_pred = model.predict(X_test)
                        model_accuracy_l = accuracy_score(y_test,y_pred)
                        model_f1_l = f1_score(y_test,y_pred,average ='weighted')
                        model_names_log.append(name)
                        model_accuracy_log.append(model_accuracy_l)
                        model_f1_log.append(model_f1_l)
                        #model_names.append(name)
                        #model_mean.append(cv_results.mean())
                        #model_std.append(cv_results.std())

                        accuracy_results = {"model_name":name,"model_accuracy":model_accuracy_l,"Standard Deviation":model_f1_l}
                        all_models.append(accuracy_results)
                        fit.append(model_fit)
                        
                        model_dict = dict(zip(model_names_log,fit))
                    
                    if st.checkbox("Accuracy Metrics"):
                        st.dataframe(pd.DataFrame(zip(model_names_log,model_accuracy_log,model_f1_log),columns =['Model Name','Model Accuracy','F1 Score']).sort_values('Model Accuracy',ascending=False))
                        #st.write(model_dict)
                    st.subheader("Advanced Prediction Model Building")
                    if st.checkbox("Advanced Prediction Model"):
                        all_x = brute_force_combinations(list(X.columns))
                        #st.dataframe(all_x)
                        seed =123
                        models =[]
                        models.append(("Logistic Regression",LogisticRegression()))
                        models.append(("Decision Tree",DecisionTreeClassifier()))
                        models.append(("Ensemble",RandomForestClassifier()))
                        #models.append(("Support Vector Machine",SVC()))      
                        all_models_adv_l =[]
                        model_names_adv_l = []
                        final_features = []
                        accu =[]
                        f1= []
                        feat = []
                        sensitivity  = []
                        result=[]
                        for i in range(len(all_x)):
                            for name,model in models:
                                final_features = all_x[i]
                                X1 = X[final_features]
                                #X= pd.get_dummies(X,dummy_na=True, drop_first=True)
                                X_train,X_test,y_train,y_test = train_test_split(X1,Y,test_size = 0.20,random_state =123)
                                #row = ""
                                model.fit(X_train,y_train)
                                y_pred = model.predict(X_test)
                                #y_pred_proba = model.predict_proba(X_test)[::,1]
                                accur = accuracy_score(y_test,y_pred)
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
                                
                        st.dataframe(pd.DataFrame(zip(model_names_adv_l,feat,accu,f1),columns =['Model Name','Features','Accuracy','F1-Score']).sort_values('Accuracy',ascending=False))
                        #st.dataframe(result_frame)
                        st.text(len(model_names_adv))
                else:
                    st.dataframe(Y)
                    st.subheader("Automated Traditional Model Building")
                    seed =123
                    models =[]
                    models.append(("Linear Regression",LinearRegression()))
                    models.append(("Decision Tree",DecisionTreeRegressor()))
                    models.append(("Ensemble",RandomForestRegressor()))
                    #models.append(("Support Vector Machine",SVR()))
                    #https://scikit-learn.org/stable/modules/model_evaluation.html
                    fit =[]
            
                    model_names = []
                    model_mean =[]
                    model_rmse = []
                    model_rmse_c1 = []
                    all_models = []
                    model_mape_line =[]
                    scoring = 'r2'

                    for name,model in models:
                        #kfold =model_selection.KFold(n_splits = 10, random_state = seed)
                        #cv_results = model_selection.cross_val_score(model,X,Y,cv=kfold,scoring = scoring)
                        X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.20,random_state =123)
                        #cv_results = cv_results
                        model.fit = model.fit(X,Y)
                        y_pred = model.predict(X_test)
                        model_names.append(name)
                       #model_mean.append(cv_results.mean())
                        #model_std.append(cv_results.std())
                        model_mape_lin = mean_absolute_percentage_error(y_pred,y_test)
                        model_rms = np.sqrt(mean_squared_error(y_test,y_pred))
                        model_rmse_c = r2_score(y_test,y_pred)
                        #accuracy_results = {"model_name":name,"model_accuracy":cv_results.mean(),"Standard Deviation":cv_results.std()}
                        #all_models.append(accuracy_results)
                        model_mape_line.append(model_mape_lin)
                        model_rmse.append(model_rms)
                        model_rmse_c1.append(model_rmse_c)
                        fit.append(model.fit)
                        model_dict = dict(zip(model_names,fit))
                
                    #if st.checkbox("Accuracy Metrics"):
                    st.dataframe(pd.DataFrame(zip(model_names,model_rmse_c1,model_rmse,model_mape_line),columns =['Model Name','R Square','RMSE ',' MAPE']))

                    st.subheader("Advanced Prediction Model Building")
                    if st.checkbox("Advanced Prediction Model"):
                        all_x = brute_force_combinations(list(X.columns))
                        #all_x = list(all_x)
                        #st.dataframe(all_x[0])
                        seed =123
                        models =[]
                        models.append(("Linear Regression",LinearRegression()))
                        models.append(("Decision Tree",DecisionTreeRegressor()))
                        models.append(("Ensemble",RandomForestRegressor()))
                        #models.append(("Support Vector Machine",SVR()))             
                        all_models_adv =[]
                        model_names_adv = []
                        model_r2 =[]
                        model_mse = []
                        model_rmse = []
                        feat = []
                        for i in range(len(all_x)):
                            for name,model in models:
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
                                mse = mean_absolute_percentage_error(y_test,y_pred)
                                #row += "," +str(mse)
                                rmse = np.sqrt(mean_squared_error(y_test,y_pred))
                                #row += "," + str(rmse)
                                accuracy_results_adv = {"model_name":name,"model_accuracy":r2score.mean(),"MSE":mse.mean(),"RMSE":rmse.mean()}
                                all_models_adv.append(accuracy_results_adv)
                                model_names_adv.append(name)
                                model_r2.append(r2score)
                                model_mse.append(mse)
                                model_rmse.append(rmse)
                                feat.append(final_features)
                                
                        st.dataframe(pd.DataFrame(zip(model_names_adv,feat,model_r2,model_mse,model_rmse),columns =['Model Name','Features','R-Square','MAPE','RMSE']).sort_values('Model R-Square',ascending=False))
                        st.text(len(model_names_adv))       
                        #result_df = pd.DataFrame(zip(data,names=["Model","Model_intercept", "Features", "r2_score", "MSE","RMSE"]))
                        #st.dataframe(all_combinations.head())

        st.subheader("Prediction on your data")    
        if st.checkbox("Predictions on your data"):
            New_data = st.file_uploader("Upload New Dataset",type=["csv","txt"])
            if New_data is not None:
                New_df=pd.read_csv(New_data)
                st.dataframe(New_df.head())
                test_X = New_df.drop(columns=selected_columns)
                test_X.replace(np.nan,0,inplace=True)
                test_X = pd.get_dummies(test_X,dummy_na=True, drop_first=True)
                
                st.subheader(" Processed Data")
                st.dataframe(test_X.head(100))
                #XY = X.drop(columns=selected_Target_columns)
                Model_select = st.multiselect("Select Model",model_names)
                #for Model in model_names:
                    #if Model in model_dict:
                #st.dataframe(model_dict)
                st.write(Model_select[0])
                nmod = model_dict[str(Model_select[0])]
                
                prediction = nmod.predict(test_X)
                test_X['Prediction'] = prediction
                st.dataframe(test_X)
                    
                                                   
                            
                       
                
    #elif choice =='Plots':
        #st.subheader("Visualize your Data ")
        
    #elif choice =='Model Building':
        #st.subheader("Model Building")
        #if st.checkbox("Select target columns"):
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




