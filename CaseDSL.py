# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:14:29 2023

@author: hp
"""

from sklearn.pipeline import Pipeline
import sklearn
import pandas as pd
import numpy as np
from numpy import asarray
from datetime import datetime
import time
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import locale
from yellowbrick.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV



#Import data

Listing_Data = pd.read_csv(r'C:\Users\hp\OneDrive\Documenten\Data Science Lab\listings_summary.csv')

Listing_Data['zipcode'].value_counts()
#########################################################################################

# General data cleaning 

#1. Remove features not relevant to determination of price

Listing_Data = Listing_Data.drop(['id', 'listing_url', 'host_neighbourhood', 'calendar_updated', 'host_picture_url', 'scrape_id', 'last_scraped', 'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url', 'host_id', 'host_url', 'host_name', 'host_about', 'host_thumbnail_url','calendar_last_scraped', 'license', 'jurisdiction_names', 'weekly_price', 'monthly_price'], axis = 1)

#2. Host_listing_total_count is identical to host_listing_count apart from 26 cases which are NaN's 

Listing_Data = Listing_Data.drop(['host_total_listings_count'], axis = 1)

## print(sum((Listing_Data.host_listings_count == Listing_Data.host_total_listings_count) == False))
## df_host_listing_ident = Listing_Data.loc[((Listing_Data.host_listings_count == Listing_Data.host_total_listings_count) == False)][:5]

#3. all listings are in Berlin, so location based features can be removed under assumption that goal is to create a global prediction model

Listing_Data = Listing_Data.drop(['zipcode', 'street', 'neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'city' , 'state', 'market', 'smart_location', 'country_code', 'country', 'longitude', 'latitude'],axis = 1)

#4. Remove features with free text space, no language processing in first instance

Listing_Data = Listing_Data.drop(['name', 'summary', 'space', 'description', 'experiences_offered', 'neighborhood_overview', 'notes', 'transit', 'transit', 'access', 'interaction', 'house_rules'], axis = 1)

#5. Remove features with majority (>50%) missing (NaN) values 

Count_NaN_Values = Listing_Data.isna().sum()

Listing_Data = Listing_Data.drop(['host_response_time', 'host_response_rate', 'host_acceptance_rate', 'square_feet'], axis = 1)

#################################################################################################3
# column specific data cleaning

# transform 'host_since' to 'host_years_active', 'first_review' to 'years_since_first_review', 'last_review' to 'years_since_first_review'

Listing_Data[['host_since','first_review', 'last_review']]  = Listing_Data[['host_since', 'first_review', 'last_review']].applymap(pd.to_datetime)

Listing_Data['host_years_active'] = (datetime.now() - Listing_Data['host_since']).astype('timedelta64[Y]')
Listing_Data['years_since_first_review'] = (datetime.now() - Listing_Data['first_review']).astype('timedelta64[Y]')
Listing_Data['years_since_last_review'] = (datetime.now() - Listing_Data['last_review']).astype('timedelta64[Y]')

Listing_Data = Listing_Data.drop(['host_since', 'first_review', 'last_review'], axis = 1)

# transform features 'price, 'security_deposit', 'cleaning fee' ,'extra people' to float
Listing_Data[['price', 'security_deposit', 'cleaning_fee','extra_people']] = Listing_Data[['price', 'security_deposit', 'cleaning_fee','extra_people']].apply(lambda x:x.str[1:-3])
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
Listing_Data[['price', 'security_deposit', 'cleaning_fee','extra_people']]= Listing_Data[['price', 'security_deposit', 'cleaning_fee','extra_people']].astype(str).applymap(locale.atof)

# host_location transform into boolean feature indicating whether host lives in same city as property is located in

Listing_Data['host_in_city'] = ['t' if x == 'Berlin, Berlin, Germany' else 'f' for x in Listing_Data['host_location']]

Listing_Data = Listing_Data.drop(['host_location'], axis = 1)

# Due to time reasons features 'amenities' and 'host_verifcations' are removed, still this feature is expected to have predictive value

Listing_Data = Listing_Data.drop(['host_verifications', 'amenities'], axis = 1)

# Property_type has many categories, therefore we diminish nr of categories
Listing_Data['property_type'].value_counts()
Listing_Data['property_type'].replace({
    'Townhouse': 'House',
    'Serviced apartment': 'Apartment',
    'Villa': 'House',
    'Tiny house': 'House',
    'Earth house': 'House',
     }, inplace=True)

Listing_Data.loc[~Listing_Data['property_type'].isin(['House', 'Apartment']), 'property_type'] = 'Other'
########################################################################################################

X = Listing_Data.loc[:,Listing_Data.columns != 'price']
y = Listing_Data['price']


#preprocessing 

#train-validation-test split (80%-10%-10%) before preprocessing to prevent data leakage
test_size = 0.2

# train_val-test split (90%-10%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

# train-val split (11.11%-88.89%) -> results in train-validation-test split (80%-10%-10%) 
#X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=(val_size*len(X)/len(X_train_val)), random_state=1)

#Distinguish numerical, nominal, ordinal and binary features
num_features = ['host_listings_count', 'host_years_active', 'years_since_first_review', 'years_since_last_review', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'security_deposit', 'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights','availability_30', 'availability_60', 'availability_90', 'availability_365', 'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'calculated_host_listings_count', 'reviews_per_month'] 
bin_features = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'is_location_exact', 'has_availability', 'requires_license','instant_bookable', 'is_business_travel_ready', 'require_guest_profile_picture', 'require_guest_phone_verification', 'host_in_city']
nom_features = ['property_type', 'room_type', 'bed_type','cancellation_policy']

#num_features = ['accommodates', 'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights','availability_30', 'availability_60', 'availability_90', 'availability_365', 'number_of_reviews', 'calculated_host_listings_count'] 
#bin_features = ['is_location_exact', 'has_availability', 'requires_license','instant_bookable', 'is_business_travel_ready', 'require_guest_profile_picture', 'require_guest_phone_verification', 'host_in_city']
#nom_features = ['room_type', 'bed_type','cancellation_policy']


all_features = num_features + bin_features + nom_features
completeness_check = len(X_train_val.drop(all_features, axis = 1).columns)

#Create customized encoder that replaces 't' and 'f' into binary variables 1 respectively 0 
def binary_replacer(df):
    return df.replace({'t': 1, 'f': 0})

#First output feature price is supposed to be numerical but y_train has dtype string
#y_train_val = convert_df_to_float(pd.DataFrame(y_train_val))

from sklearn.impute import SimpleImputer

#Create preprocessing pipelines for each dtype
num_pipeline = Pipeline([('scaler', StandardScaler()),
                         ('imputer', SimpleImputer(strategy='mean'))])
bin_pipeline = Pipeline([('encoder', FunctionTransformer(binary_replacer)),
                ('imputer', SimpleImputer(strategy = 'most_frequent'))])
nom_pipeline = Pipeline([('encoder', OneHotEncoder(sparse = False, drop = 'first'))])
                        
combined_transformer = ColumnTransformer([
        ("numerical",num_pipeline, num_features),
        ("binary", bin_pipeline, bin_features),
        ("nominal",nom_pipeline, nom_features)])

#Baseline model: Linear Regression
pipe_lin_regr = Pipeline([('preprocessing', combined_transformer), 
                     ('model', LinearRegression())])

lin_regr_fit = pipe_lin_regr.fit(X_train_val, y_train_val)

from sklearn.metrics import mean_squared_error, r2_score

prediction = lin_regr_fit.predict(X_test)
RMSE = mean_squared_error(y_test, prediction, squared = False)
R2 = r2_score(y_test, prediction)


#Random forest regressor model
pipe_rand_for = Pipeline([('preprocessing', combined_transformer), 
                     ('model', RandomForestRegressor())])


rand_for_fit = pipe_rand_for.fit(X_train_val, y_train_val)
prediction = rand_for_fit.predict(X_train_val)
RMSE = mean_squared_error(y_train_val, prediction, squared = False)
R2 = r2_score(y_train_val, prediction)


#Neural Network model



#Hyperparameter optimization 

#Model evaluation 

#Mean Squared Error (MSE)
                        

                



#Hyperparameter optimization 


#Encode categorical features

#Boolean features -> transform 't' to 1 and 'f' to 0 

#Boolean_Features = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'is_location_exact', 'has_availability', 'requires_license','instant_bookable', 'is_business_travel_ready', 'require_guest_profile_picture', 'require_guest_phone_verification', 'host_in_city']
#Listing_Data[Boolean_Features] = Listing_Data[Boolean_Features].apply([lambda x: 1 if x == 't'else 0])

# Ordinal features (Ordinal encoding)

#Arr_Canc_Pol = asarray(Listing_Data['host_in_city']).reshape(-1,1)
#Encoder = OrdinalEncoder()
#Encoded_Ord_Data = Encoder.fit_transform(Arr_Canc_Pol)
#test = Encoded_Ord_Data.astype(int)

# Nominal features (One-hot-encoding)
#Nominal_Features = ['property_type', 'room_type']
#Encoded_Nom_Data = pd.get_dummies(Listing_Data[Nominal_Features], drop_first = true) #drop_first because we only need k-1 dummy variables for each feature
#Listing_Data = pd.concat([Listing_Data, Encoded_Nom_Data], axis = 1)
#Listing_Data = Listing_Data.drop(Nominal_Features, axis = 1)

#################################################################################

# Create data pipeline including scaling-step, imputing-step and prediction-step

