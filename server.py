# import Flask class from the flask module
from flask import Flask, request
import pickle

# Libraries
import numpy as np
import pandas as pd
from pandas import DataFrame,Series
from flask import jsonify
from keras.models import load_model
import tensorflow as tf

# Enabling CORS
from flask_cors import CORS
app = Flask(__name__)
CORS(app)



@app.route('/test',methods=['GET','POST'])
def test():
	return("test")
# 	# Get values
# 	df=get_data(request.form)
# 	# for categorical variable
# 	df = pd.get_dummies(data=df, columns=['smoke','hormonal_contra','iud','std',
#                                       'dx_cancer','dx_cin','dx_hpv','dx','hin','cit','sch'])
# 	# clean_df=clean_data(df)
# 	print(df)
# 	return(str(df) )

# 	return("test"+str(new_df))
# @app.route('/transform',methods=['GET','POST'])
# def transform():
# 	# Get values
# 	df=get_data(request.form)
# 	clean_df=clean_data(df)
# 	my_json = {}
# 	i=0
# 	for col in ['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane']:
# 		my_json[col] = clean_df[0][i]
# 		i=i+1
# 	return(str(my_json) )


@app.route('/rf',methods=['GET','POST'])
def rf():
	# Get values
	df=get_data(request.form)
	# clean_df_rf=clean_data(df_rf)
	class_prediced = rf.predict(df)
	print(class_prediced)
	return (str(class_prediced))

@app.route('/svm',methods=['GET','POST'])
def svm():
	df=get_data(request.form)
	# clean_df_rf=clean_data(df_rf)
	class_prediced = svm.predict(df)
	print(class_prediced)
	return (str(class_prediced))

@app.route('/nn',methods=['GET','POST'])
def nn():
	df=get_data(request.form)
	global graph
	with graph.as_default():
		class_prediced = nn.predict(df)
		print(class_prediced)
		return (str(class_prediced))

# Routes for models Ends

def get_data(dict):

	age = dict.get('age')
	sex = dict.get('sex')
	chest_pain = dict.get('chest_pain')
	bp = dict.get('bp')
	serum_cholestoral = dict.get('serum_cholestoral')
	fasting_blood_sugar = dict.get('fasting_blood_sugar')
	electrocardiographic = dict.get('electrocardiographic')
	max_heart_rate = dict.get('max_heart_rate')
	induced_angina = dict.get('induced_angina')
	ST_depression = dict.get('ST_depression')
	slope = dict.get('slope')
	vessels = dict.get('vessels')
	thal = dict.get('thal')

	data = np.array([age,sex,chest_pain,bp,serum_cholestoral,fasting_blood_sugar,electrocardiographic,
		max_heart_rate,induced_angina,ST_depression,slope,vessels,thal]).reshape(1,13)
	df=pd.DataFrame(data)
	df.columns = ['age','sex','chest_pain','bp','serum_cholestoral','fasting_blood_sugar','electrocardiographic',
	'max_heart_rate','induced_angina','ST_depression','slope','vessels','thal']
	return (df)


# def clean_data(df):
# 	df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})
# 	df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})
# 	df[['pcc','ba']] = df[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})
# 	df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})
# 	df_new=(df.iloc[0]).reshape(1,-1)
# 	return(df_new)

def get_model():
	global rf
	global svm
	global nn

	rf_file = open('models/rf.pckl', 'rb')
	rf = pickle.load(rf_file)
	rf_file.close()

	svm_file = open('models/svm.pckl', 'rb')
	svm = pickle.load(svm_file)
	svm_file.close()

	nn = load_model('models/model4.h5')
	global graph
	graph = tf.get_default_graph()


if __name__ == "__main__":
	print("**Starting Server...")

	# Call function that loads Model
	get_model()

# Run Server
app.run()
