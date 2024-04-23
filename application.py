from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as pd
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
app = Flask(__name__)

# scaler=pickle.load(open('Model/standardScalar.pkl','rb'))
# model=pickle.load(open('Model/modelForPrediction.pkl','rb'))

Item_Type_encoder=pickle.load(open('models/Item_Type_encoder.pkl','rb'))        
Item_Fat_Content_encoder=pickle.load(open("models/Item_Fat_Content_encoder.pkl","rb"))
Outlet_Size_encoder=pickle.load(open("models/Outlet_Size_encoder.pkl","rb"))
Outlet_Location_Type_encoder=pickle.load(open("models/Outlet_Location_Type_encoder.pkl","rb"))
Outlet_Type_encoder=pickle.load(open("models/Outlet_Type_encoder.pkl","rb"))
Outlet_Identifier_encoder=pickle.load(open("models/Outlet_Identifier_encoder.pkl","rb"))
model = pickle.load(open('models/DTR.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# route for homepage
@app.route("/")
def hello_world():
    return render_template('index.html')

## Route for single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=0

    if request.method=='POST':
        Item_Weight=float(request.form.get('Item_Weight'))
        print(Item_Weight)
        Item_Fat_Content=str(request.form.get('Item_Fat_Content'))
        print(Item_Fat_Content)
        Item_Visibility=float(request.form.get('Item_Visibility'))
        print(Item_Visibility)
        Item_Type=str(request.form.get('Item_Type'))
        print(Item_Type)
        Item_MRP=float(request.form.get('Item_MRP'))
        print(Item_MRP)
        Outlet_Identifier=str(request.form.get('Outlet_Identifier'))
        print(Outlet_Identifier)
        Outlet_Establishment_Year=int(request.form.get('Outlet_Establishment_Year'))
        print(Outlet_Establishment_Year)
        Outlet_Size=str(request.form.get('Outlet_Size'))
        print(Outlet_Size)
        Outlet_Location_Type=str(request.form.get('Outlet_Location_Type'))
        print(Outlet_Location_Type)
        Outlet_Type=str(request.form.get('Outlet_Type'))
        print(Outlet_Type)
        # Item_Outlet_Sales=float(request.form.get('Item_Outlet_Sales'))

        Item_Type=int(Item_Type_encoder.transform([[Item_Type]])[0])
        Item_Fat_Content=int(Item_Fat_Content_encoder.transform([[Item_Fat_Content]])[0][0])
        Outlet_Size=int(Outlet_Size_encoder.transform([[Outlet_Size]])[0][0])
        Outlet_Location_Type=int(Outlet_Location_Type_encoder.transform([[Outlet_Location_Type]])[0][0])
        Outlet_Type=int(Outlet_Type_encoder.transform([[Outlet_Type]])[0][0])
        # Outlet_Identifier=int(Outlet_Identifier_encoder.transform([[Outlet_Identifier]])[0][0])
        
        


        

        

        new_data=scaler.transform([[Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type,Item_MRP, Outlet_Identifier, Outlet_Establishment_Year,Outlet_Size, Outlet_Location_Type, Outlet_Type]])

        result=model.predict(new_data)
       
        
            
        return render_template('single_prediction.html',result=result)

    else:
        return render_template('home.html') 



if __name__=="__main__":
    app.run(host="0.0.0.0",port=8000)