#str
import streamlit as st
import pickle as pkl
import numpy as np
import pandas as pd

st.title('Car Price Prediction Website')
pipe=pkl.load(open('CPP.pkl','rb'))
df=pd.read_csv('final_data.csv')
companies=sorted(df['company'].unique())
years=range(2000,2027)
company=st.sidebar.selectbox('Select company:',companies)
names=sorted(df[df['company']==company]['name'].unique())
name=st.sidebar.selectbox('Select name:',names)
year=st.sidebar.selectbox('Select year:',years)
kms_driven=st.sidebar.number_input('Enter kms_driven:',value=5000,min_value=1000,max_value=200000,step=1000)
fuel_type=st.sidebar.selectbox('Select Fuel Type:',['Petrol','Diesel'])

if st.sidebar.button('Predict Price:'):
    st.write('You have selected:')
    st.write(f'company:{company}')
    st.write(f'name:{name}')
    st.write(f'year:{year}')
    st.write(f'kms_driven:{kms_driven}')
    st.write(f'Fuel Type:{fuel_type}')
    
    myinput=[[company,name,year,kms_driven,fuel_type]]
    columns=['company','name','year','kms_driven','fuel_type']
    myinput=pd.DataFrame(data=myinput,columns=columns)
    
    result=pipe.predict(myinput)
    if result[0]<0:
        st.write('Sorry Predicted Price is negative.........Please check your input values.')
    else:
        st.write('Predicted Price:',str(round(result[0,0])))