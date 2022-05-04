
from re import M
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import json
from firebase_admin import firestore
from flask import Flask
from flask import request
import threading
import time
import pandas as pd
import random
import pickle

app = Flask(__name__)


databaseURL = "https://panda-121720-default-rtdb.asia-southeast1.firebasedatabase.app/"
cred = credentials.Certificate("panda-121720-firebase-adminsdk-hd9m8-3870d01444.json")
dataset=  pd.read_csv("Crop_recommendation.csv")
c = dataset.label.astype('category')

default_app = firebase_admin.initialize_app(cred, {
	'databaseURL':databaseURL
	})

fin = firebase_admin.initialize_app(cred,name="FireStore")
dbF = firestore.client(fin)



ref = db.reference("/")
with open('CropPredictor final .pkl' , 'rb') as f:
    lr = pickle.load(f)

def thread_function(userID):
	time.sleep(5)
	#print the size of the dataframe
	
	while True:
		# print(random.randint(0,dataset.size))
	
		# print(lr.predict([[obj.N,obj.P,obj.K,obj.temperature,obj.humidity,obj.ph,obj.rainfall]])[0])
		obj = dataset.loc[random.randint(0,dataset.shape[0])]
		dic = {
			"N":str(obj.N),
			"P":str(obj.P),
			"K":str(obj.K),
			"temperature":str(obj.temperature),
			"humidity":str(obj.humidity),
			"ph":str(obj.ph),
			"rainfall":str(obj.rainfall),
			"output":c.cat.categories[lr.predict([[obj.N,obj.P,obj.K,obj.temperature,obj.humidity,obj.ph,obj.rainfall]])[0]]
		}
		ref.child(userID).update(
			dic
		)
		doc_ref = dbF.collection(u'data').document(userID)
		doc_ref.update({u'N': firestore.ArrayUnion([u'{}'.format(str(obj.N))])})
		doc_ref.update({u'P': firestore.ArrayUnion([u'{}'.format(str(obj.P))])})
		doc_ref.update({u'K': firestore.ArrayUnion([u'{}'.format(str(obj.K))])})
		doc_ref.update({u'temperature': firestore.ArrayUnion([u'{}'.format(str(obj.temperature))])})
		doc_ref.update({u'humidity': firestore.ArrayUnion([u'{}'.format(str(obj.humidity))])})
		doc_ref.update({u'ph': firestore.ArrayUnion([u'{}'.format(str(obj.ph))])})
		doc_ref.update({u'rainfall': firestore.ArrayUnion([u'{}'.format(str(obj.rainfall))])})
		doc_ref.update({u'output': firestore.ArrayUnion([u'{}'.format(c.cat.categories[lr.predict([[dataset.iloc[0].N,dataset.iloc[0].P,dataset.iloc[0].K,dataset.iloc[0].temperature,dataset.iloc[0].humidity,dataset.iloc[0].ph,dataset.iloc[0].rainfall]])[0]])])})

		time.sleep(5)

@app.route('/')
def index():
   return '<html><body><h1>Hello World</h1></body></html>'

@app.route('/start', methods = ['POST'])
def user():
	data = request.json # a multidict containing POST data
	dic = {
		str(data['userID']):{
			"N":str(dataset.loc[0].N),
			"P":str(dataset.loc[0].P),
			"K":str(dataset.loc[0].K),
			"temperature":str(dataset.loc[0].temperature),
			"humidity":str(dataset.loc[0].humidity),
			"ph":str(dataset.loc[0].ph),
			"rainfall":str(dataset.loc[0].rainfall),
			"output":c.cat.categories[lr.predict([[dataset.iloc[0].N,dataset.iloc[0].P,dataset.iloc[0].K,dataset.iloc[0].temperature,dataset.iloc[0].humidity,dataset.iloc[0].ph,dataset.iloc[0].rainfall]])[0]]
		}
	}
	ref.set(dic)
	doc_ref = dbF.collection(u'data').document(str(data['userID']))
	doc_ref.set({u'N': firestore.ArrayUnion([u'{}'.format(str(dataset.loc[0].N))])})
	doc_ref.set({u'P': firestore.ArrayUnion([u'{}'.format(str(dataset.loc[0].P))])})
	doc_ref.set({u'K': firestore.ArrayUnion([u'{}'.format(str(dataset.loc[0].K))])})
	doc_ref.set({u'temperature': firestore.ArrayUnion([u'{}'.format(str(dataset.loc[0].temperature))])})
	doc_ref.set({u'humidity': firestore.ArrayUnion([u'{}'.format(str(dataset.loc[0].humidity))])})
	doc_ref.set({u'ph': firestore.ArrayUnion([u'{}'.format(str(dataset.loc[0].ph))])})
	doc_ref.set({u'rainfall': firestore.ArrayUnion([u'{}'.format(str(dataset.loc[0].rainfall))])})
	doc_ref.set({u'output': firestore.ArrayUnion([u'{}'.format(c.cat.categories[lr.predict([[dataset.iloc[0].N,dataset.iloc[0].P,dataset.iloc[0].K,dataset.iloc[0].temperature,dataset.iloc[0].humidity,dataset.iloc[0].ph,dataset.iloc[0].rainfall]])[0]])])})

	x = threading.Thread(target=thread_function, args=(str(data['userID']),))
	x.start()
	return '<html><body><h1>Thread Started</h1></body></html>'



if __name__ == '__main__':
	app.run(debug=True)
      
  