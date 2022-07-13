# Airbnb-new-user-bookings
Predicting destination preference of the new user

**Business Problem**

Airbnb has consistently been one of the best site for travellers which helps them to find hotels, room stays and tourism activities that suits the travellers wish. Airbnb has lot of improvised features that help travellers filter, sort and choose the best that fits their interest. It is working on enhancing the user interface in a way that could reduce the searching time of the users by providing them with the services that they expect. Thereby they increase their quality of service.
                    
**Problem Statement**

Here we have to predict the desired destination of the first time users, thereby we can have the knowledge of the what the user expects and can show them with the more personalised content, thus decreasing the average time of the first booking. We are provided with user's basic details (like browser, devcie type, etc) and also browser sessions data.

**Evaluation Metric**

Normalised Discounted Cummulative Gain Score. Suggested by Kaggle.

**Source of Data**

The Data is taken from the Kaggle Competition page
https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data
We have 5 csv files : train_users_2.csv, test_users.csv, sessions.csv, age_gender_bkts.csv, countries.csv

train_users_2.csv: Contains basic details of users like gender,language, browser used, device used etc.

sessions.csv: Contains browser session of the user. user_id can be joined with the id of the train_users_2 dataset. This contains fields like action (action done by the user), action_type (type of the action), action_detail, device_type of the user and secs_elapsed (seconds elapsed for each action).

countries.csv: Has the information of each countries (ie latitude, longitude, country's language and the language difference between english and the native language
age_gender_bkts: This file has the information of number of people within the bucket range for each country destination and for each gender.

test_users.csv: Has the same informations as the train_users_2 file. User id in test_users file are also in sessions.csv file.

Description about the files in the repository:

'Users_Feature_Extraction,_Analysis,_Engineering.ipynb' file contains the analysis, feature engineering of train_users.csv file.

'sessions_feature_extraction,analysis,engineering_.ipynb' file contains the analysis, feature engineering of sessions.csv file.

'Airbnb_modeling.ipynb' file contains modeling of the preprocessed data.

'final.ipynb' file contains final function that can be used when deploying in the cloud.

'final_test.py' file contains the code for deployment in AWS cloud. Flask is used to create and design the webpage.

'template' folder consists of the template.

Other files are extracted during training.

Here in this github repo I haven't uploaded my final model.
