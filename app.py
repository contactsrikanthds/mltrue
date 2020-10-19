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
from itertools import combinations
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from itertools import combinations
import time
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
        st.dataframe(df.head())
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
        if st.checkbox("TIme Series"):
            st.subheader("Select Date Column")
            all_columns = df.columns
            selected_columns_time = st.multiselect("Select Columns",all_columns)
            Non_selected_columns = df.drop(columns=selected_columns_time)
            df['Date']= df[selected_columns_time].apply(lambda x:pd.to_datetime(x,errors = 'coerce',format = '%Y'))
                         
            st.subheader("Select Data to forecast")
            selected_columns_1 = st.multiselect("Select Columns purpose",all_columns)
            new_df_new = df[selected_columns_time]
            new_df_new[selected_columns_1] = df[selected_columns_1]
            new_df_new[selected_columns_t]= df['Date']
            st.dataframe(new_df_new)
            
            st.subheader("Automated Traditional Model Building")
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
                X= pd.get_dummies(X,dummy_na=True, drop_first=True)
                st.subheader("Training Data - Input Features")
                st.dataframe(X)
                Y = new_df[selected_Target_columns]
                st.subheader("Training Data - Target Feature")
                if Y.dtypes.any()==object:
                    st.dataframe(Y)
                elif Y.iloc[:,0].nunique()<5:
                  
                    st.dataframe(Y)
                    st.subheader("Automated Traditional Model Building")
                    seed =123
                    models =[]
                    models.append(("Logistic Regression",LogisticRegression()))
                    models.append(("Decision Tree",DecisionTreeClassifier()))
                    models.append(("Ensemble",RandomForestClassifier()))
                    models.append(("Support Vector Machine",SVC()))
                    fit =[]
            
                    model_names = []
                    model_mean =[]
                    model_std = []
                    all_models = []
                    scoring = 'balanced_accuracy'

                    for name,model in models:
                        kfold =model_selection.KFold(n_splits = 10, random_state = seed)
                        cv_results = model_selection.cross_val_score(model,X,Y,cv=kfold,scoring = scoring)
                        model_fit = model.fit(X,Y)
                        model_names.append(name)
                        model_mean.append(cv_results.mean())
                        model_std.append(cv_results.std())

                        accuracy_results = {"model_name":name,"model_accuracy":cv_results.mean(),"Standard Deviation":cv_results.std()}
                        all_models.append(accuracy_results)
                        fit.append(model_fit)
                        
                        model_dict = dict(zip(model_names,fit))
                        
                    if st.checkbox("Accuracy Metrics"):
                        st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns =['Model Name','Model Accuracy','Accuracy Variation(Std)']).sort_values('Model Accuracy',ascending=False))
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
                        models.append(("Support Vector Machine",SVC()))      
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
                else:
                    st.dataframe(Y)           
                    st.subheader("Automated Traditional Model Building")
                    seed =123
                    models =[]
                    models.append(("Linear Regression",LinearRegression()))
                    models.append(("Decision Tree",DecisionTreeRegressor()))
                    models.append(("Ensemble",RandomForestRegressor()))
                    models.append(("Support Vector Machine",SVR()))
                    #https://scikit-learn.org/stable/modules/model_evaluation.html
                    fit =[]
            
                    model_names = []
                    model_mean =[]
                    model_std = []
                    all_models = []
                    scoring = 'r2'

                    for name,model in models:
                        kfold =model_selection.KFold(n_splits = 10, random_state = seed)
                        cv_results = model_selection.cross_val_score(model,X,Y,cv=kfold,scoring = scoring)
                        cv_results = cv_results
                        model.fit = model.fit(X,Y)
                        model_names.append(name)
                        model_mean.append(cv_results.mean())
                        model_std.append(cv_results.std())

                        accuracy_results = {"model_name":name,"model_accuracy":cv_results.mean(),"Standard Deviation":cv_results.std()}
                        all_models.append(accuracy_results)
                        fit.append(model.fit)
                        model_dict = dict(zip(model_names,fit))
                
                    if st.checkbox("Accuracy Metrics"):
                        st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns =['Model Name','Model R-Square',' Variatoin(Std)']))

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
                        models.append(("Support Vector Machine",SVR()))             
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
                                mse = mean_squared_error(y_test,y_pred)
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
                                
                        st.dataframe(pd.DataFrame(zip(model_names_adv,feat,model_r2,model_mse,model_rmse),columns =['Model Name','Features','Model R-Square',' MSE','RMSE']).sort_values('Model R-Square',ascending=False))
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




