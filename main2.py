#python -m uvicorn main:app --reload

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from typing import Optional
import numpy as np


from pymongo import MongoClient

# Conectar a MongoDB en localhost
client = MongoClient("mongodb+srv://brysagix1234:admin1234@cluster0.o9ii7lx.mongodb.net/?appName=Cluster0")

# Seleccionar base de datos
db = client["uvProyectoOrquideas"]

# Seleccionar colección
coleccion = db["nuevosClasificados"]
coleccion2 = db["nuevasPredicciones"]

app = FastAPI() 


clasificador_modelo5 = joblib.load("forest.joblib")
clasificador_modelo6 = joblib.load("forest.joblib")
clasificador_modelo9 = joblib.load("random_forest_model_clasif_30test_id9.joblib")
clasificador_modelo12 = joblib.load("random_forest_model_clasif_30test_id12.joblib")
clasificador_modelo15 = joblib.load("mlp_clasif_30test_id15.joblib")


predictor_modelo5 = joblib.load("forest.joblib")
predictor_modelo6 = joblib.load("forest.joblib")
predictor_modelo9 = joblib.load("boosting_30_predict_id_9.joblib")
predictor_modelo12 = joblib.load("boosting_30_predict_id_12.joblib")
predictor_modelo15 = joblib.load("boosting_30_predic_id_15.joblib")




# ==========================================
# FUNCIÓN PARA PREPARAR LOS DATOS
# ==========================================

def preparar_datos(
    sexo=None,
    edad_continua=None,
    comuna_raw=None
):


    datos = {}

    if sexo is not None:
        datos["sexo"] = sexo

    if edad_continua is not None:
        datos["edad_continua"] = edad_continua

        if edad_continua <= 12:
            edad_grupo = 1
        elif edad_continua <= 26:
            edad_grupo = 2
        elif edad_continua <= 93:
            edad_grupo = 3
        else:
            edad_grupo = None

        if edad_grupo is not None:
            datos["edad_grupo"] = edad_grupo
            datos["edad_g1"] = int(edad_grupo == 1)
            datos["edad_g2"] = int(edad_grupo == 2)
            datos["edad_g3"] = int(edad_grupo == 3)

    if comuna_raw is not None:
        datos["comuna_raw"] = comuna_raw

        for i in range(1, 13):
            datos[f"comuna_{i}"] = int(comuna_raw == i)

    return datos




def construir_raw(*bits):
    binario = "".join(str(int(bit)) for bit in bits)
    return int(binario, 2)







# ==========================================
# ENDPOINT PARA PREPARAR DATOS
# ==========================================



# GET
@app.get("/")
def inicio():
    return {
        "mensaje": "Hola, esta es mi API funcionando"
    }


@app.post("/clasificador_prueba_cinco")
def inicio(sexo: int, edad_continua: int, comuna_raw:int, hb_aa:int, hb_cc:int, hb_ss:int, hb_ac:int, hb_as:int, hb_sc:int):

    resultado = preparar_datos(
        sexo=sexo,
        edad_continua=edad_continua,
        comuna_raw=comuna_raw,
    )

    hb_raw = construir_raw(hb_aa, hb_cc, hb_ss, hb_ac, hb_as, hb_sc)

    new_data_point_np = np.array([[resultado["sexo"], resultado["edad_continua"], resultado["edad_g1"], resultado["edad_g2"], resultado["edad_g3"], resultado["comuna_raw"], hb_raw]])
    prediction_np = clasificador_modelo5.predict(new_data_point_np)

    
    nuevo_dato = {
        "tipo_prueba": 5,
        "sexo": sexo,
        "edad_continua": edad_continua,
        "comuna_raw": comuna_raw,
        "hb_raw": hb_raw,
        "hb_aa": hb_aa,
        "hb_cc": hb_cc,
        "hb_ss": hb_ss,
        "hb_ac": hb_ac,
        "hb_as": hb_as,
        "hb_sc": hb_sc,
        "clasificacion":  prediction_np[0]
    }

    resultado = coleccion.insert_one(nuevo_dato)

    return {
        #"mensaje": resultado
        "mensaje": prediction_np[0]
    }



@app.post("/clasificador_prueba_seis")
def inicio(sexo: int, edad_continua: int, comuna_raw:int, hb_aa:int, hb_cc:int, hb_ss:int, hb_ac:int, hb_as:int, hb_sc:int):

    resultado = preparar_datos(
        sexo=sexo,
        edad_continua=edad_continua,
        comuna_raw=comuna_raw,
    )

    hb_raw = construir_raw(hb_aa, hb_cc, hb_ss, hb_ac, hb_as, hb_sc)

    new_data_point_np = np.array([[resultado["sexo"], resultado["edad_continua"], resultado["edad_g1"], resultado["edad_g2"], resultado["edad_g3"], resultado["comuna_raw"], hb_raw]])
    prediction_np = clasificador_modelo6.predict(new_data_point_np)

    nuevo_dato = {
        "tipo_prueba":6,
        "sexo": sexo,
        "edad_continua": edad_continua,
        "comuna_raw": comuna_raw,
        "hb_raw": hb_raw,
        "hb_aa": hb_aa,
        "hb_cc": hb_cc,
        "hb_ss": hb_ss,
        "hb_ac": hb_ac,
        "hb_as": hb_as,
        "hb_sc": hb_sc,
        "clasificacion":  prediction_np[0]
    }

    resultado = coleccion.insert_one(nuevo_dato)

    return {
        #"mensaje": resultado
        "mensaje": prediction_np[0]
    }



@app.post("/clasificador_prueba_nueve")
def inicio(sexo: int, edad_continua: int, comuna_raw:int, hb_raw:int, g6pd_raw:int ):
    
    resultado = preparar_datos(
        sexo=sexo,
        edad_continua=edad_continua,
        comuna_raw=comuna_raw,
        hb_raw=hb_raw,
        g6pd_raw=g6pd_raw
    )

    new_data_point_np = np.array([[resultado["sexo"],
                                   resultado["edad_continua"], resultado["edad_g1"], resultado["edad_g2"], resultado["edad_g3"],
                                   resultado["comuna_raw"],
                                   resultado["hb_aa"], resultado["hb_cc"], resultado["hb_ss"], resultado["hb_ac"], resultado["hb_as"] , resultado["hb_sc"],
                                   resultado["g6pd_bb"], resultado["g6pd_ap_ap"], resultado["g6pd_am_am"], resultado["g6pd_b_ap"], resultado["g6pd_b_am"], resultado["g6pd_ap_am"], resultado["g6pd_b"], resultado["g6pd_ap"], resultado["g6pd_am"] ]])
    prediction_np = clasificador_modelo9.predict(new_data_point_np)


    nuevo_dato = {
        "tipo_prueba":9,
        "sexo": sexo,
        "edad_continua": edad_continua,
        "comuna_raw": comuna_raw,
        "hb_raw": hb_raw,
        "g6pd_raw": g6pd_raw,
        "clasificacion":  prediction_np[0]
    }

    resultado = coleccion.insert_one(nuevo_dato)

    return {
        #"mensaje": resultado
        "mensaje": prediction_np[0]
    }



@app.post("/clasificador_prueba_doce")
def inicio(sexo: int, edad_continua: int, comuna_raw:int, hb_raw:int, duffy_raw:int, g6pd_raw:int ):

    resultado = preparar_datos(
        sexo=sexo,
        edad_continua=edad_continua,
        comuna_raw=comuna_raw,
        hb_raw=hb_raw,
        g6pd_raw=g6pd_raw,
        duffy_raw=duffy_raw
    )

    new_data_point_np = np.array([[resultado["sexo"],
                                   resultado["edad_continua"], resultado["edad_g1"], resultado["edad_g2"], resultado["edad_g3"],
                                   resultado["comuna_raw"],
                                   resultado["hb_aa"], resultado["hb_cc"], resultado["hb_ss"], resultado["hb_ac"], resultado["hb_as"] , resultado["hb_sc"],
                                   resultado["g6pd_bb"], resultado["g6pd_ap_ap"], resultado["g6pd_am_am"], resultado["g6pd_b_ap"], resultado["g6pd_b_am"], resultado["g6pd_ap_am"], resultado["g6pd_b"], resultado["g6pd_ap"], resultado["g6pd_am"],
                                   resultado["duffy_fybes_fybes"], resultado["duffy_fya_fya"], resultado["duffy_fyb_fyb"], resultado["duffy_fya_fybes"], resultado["duffy_fyb_fybes"], resultado["duffy_fya_fyb"] ]])
    prediction_np = clasificador_modelo12.predict(new_data_point_np)


    
    nuevo_dato = {
        "tipo_prueba":12,
        "sexo": sexo,
        "edad_continua": edad_continua,
        "comuna_raw": comuna_raw,
        "hb_raw": hb_raw,
        "g6pd_raw": g6pd_raw,
        "duffy_raw": duffy_raw,
        "clasificacion":  prediction_np[0]
    }

    resultado = coleccion.insert_one(nuevo_dato)


    return {
        #"mensaje": resultado
        "mensaje": prediction_np[0]
    }



@app.post("/clasificador_prueba_quince")
def inicio(sexo: int, edad_continua: int, comuna_raw:int, hb_raw:int, duffy_raw:int, btal29_raw:int, btal88_raw:int, g6pd_raw:int ):

    resultado = preparar_datos(
        sexo=sexo,
        edad_continua=edad_continua,
        comuna_raw=comuna_raw,
        hb_raw=hb_raw,
        g6pd_raw=g6pd_raw,
        duffy_raw=duffy_raw,
        btal29_raw=btal29_raw,
        btal88_raw=btal88_raw
    )

    new_data_point_np = np.array([[resultado["sexo"],
                                   resultado["edad_continua"], resultado["edad_g1"], resultado["edad_g2"], resultado["edad_g3"],
                                   resultado["comuna_raw"],
                                   resultado["hb_aa"], resultado["hb_cc"], resultado["hb_ss"], resultado["hb_ac"], resultado["hb_as"] , resultado["hb_sc"],
                                   resultado["g6pd_bb"], resultado["g6pd_ap_ap"], resultado["g6pd_am_am"], resultado["g6pd_b_ap"], resultado["g6pd_b_am"], resultado["g6pd_ap_am"], resultado["g6pd_b"], resultado["g6pd_ap"], resultado["g6pd_am"],
                                   resultado["duffy_fybes_fybes"], resultado["duffy_fya_fya"], resultado["duffy_fyb_fyb"], resultado["duffy_fya_fybes"], resultado["duffy_fyb_fybes"], resultado["duffy_fya_fyb"],
                                   resultado["btal29_aa"], resultado["btal29_ag"], resultado["btal29_gg"],
                                   resultado["btal88_cc"], resultado["btal88_ct"], resultado["btal88_tt"] ]])
    prediction_np = clasificador_modelo15.predict(new_data_point_np)

    nuevo_dato = {
        "tipo_prueba":15,
        "sexo": sexo,
        "edad_continua": edad_continua,
        "comuna_raw": comuna_raw,
        "hb_raw": hb_raw,
        "g6pd_raw": g6pd_raw,
        "duffy_raw": duffy_raw,
        "btal29_raw": btal29_raw,
        "btal88_raw": btal88_raw,
        "clasificacion": prediction_np
    }

    resultado = coleccion.insert_one(nuevo_dato)


    return {
        #"mensaje": resultado
        "mensaje": prediction_np[0]
    }




@app.post("/predictor_prueba_cinco")
def inicio(sexo: int, edad_continua: int, comuna_raw:int, hb_raw:int):

    resultado = preparar_datos(
        sexo=sexo,
        edad_continua=edad_continua,
        comuna_raw=comuna_raw,
        hb_raw=hb_raw,
    )

    new_data_point_np = np.array([[resultado["sexo"], resultado["edad_continua"], resultado["edad_g1"], resultado["edad_g2"], resultado["edad_g3"], resultado["comuna_raw"], resultado["hb_raw"]]])
    prediction_np = predictor_modelo5.predict(new_data_point_np)

    return {
        #"mensaje": resultado
        "mensaje": prediction_np[0]
    }



@app.post("/predictor_prueba_seis")
def inicio(sexo: int, edad_continua: int, comuna_raw:int, hb_raw:int):

    resultado = preparar_datos(
        sexo=sexo,
        edad_continua=edad_continua,
        comuna_raw=comuna_raw,
        hb_raw=hb_raw,
    )

    new_data_point_np = np.array([[resultado["sexo"], resultado["edad_continua"], resultado["edad_g1"], resultado["edad_g2"], resultado["edad_g3"], resultado["comuna_raw"], resultado["hb_aa"], resultado["hb_cc"], resultado["hb_ss"], resultado["hb_ac"], resultado["hb_as"] , resultado["hb_sc"] ]])
    prediction_np = predictor_modelo6.predict(new_data_point_np)

    return {
        #"mensaje": resultado
        "mensaje": prediction_np[0]
    }



@app.post("/predictor_prueba_nueve")
def inicio(sexo: int, edad_continua: int, comuna_raw:int, hb_raw:int, g6pd_raw:int ):
    
    resultado = preparar_datos(
        sexo=sexo,
        edad_continua=edad_continua,
        comuna_raw=comuna_raw,
        hb_raw=hb_raw,
        g6pd_raw=g6pd_raw
    )

    new_data_point_np = np.array([[resultado["sexo"],
                                   resultado["edad_continua"], resultado["edad_g1"], resultado["edad_g2"], resultado["edad_g3"],
                                   resultado["comuna_raw"],
                                   resultado["hb_aa"], resultado["hb_cc"], resultado["hb_ss"], resultado["hb_ac"], resultado["hb_as"] , resultado["hb_sc"],
                                   resultado["g6pd_bb"], resultado["g6pd_ap_ap"], resultado["g6pd_am_am"], resultado["g6pd_b_ap"], resultado["g6pd_b_am"], resultado["g6pd_ap_am"], resultado["g6pd_b"], resultado["g6pd_ap"], resultado["g6pd_am"] ]])
    prediction_np = predictor_modelo9.predict(new_data_point_np)

    nuevo_dato = {
        "tipo_prueba":9,
        "sexo": sexo,
        "edad_continua": edad_continua,
        "comuna_raw": comuna_raw,
        "hb_raw": hb_raw,
        "g6pd_raw": g6pd_raw,
        "prediccion":  prediction_np[0]
    }

    resultado = coleccion2.insert_one(nuevo_dato)

    return {
        #"mensaje": resultado
        "mensaje": prediction_np[0]
    }



@app.post("/predictor_prueba_doce")
def inicio(sexo: int, edad_continua: int, comuna_raw:int, hb_raw:int, duffy_raw:int, g6pd_raw:int ):

    resultado = preparar_datos(
        sexo=sexo,
        edad_continua=edad_continua,
        comuna_raw=comuna_raw,
        hb_raw=hb_raw,
        g6pd_raw=g6pd_raw,
        duffy_raw=duffy_raw
    )

    new_data_point_np = np.array([[resultado["sexo"],
                                   resultado["edad_continua"], resultado["edad_g1"], resultado["edad_g2"], resultado["edad_g3"],
                                   resultado["comuna_raw"],
                                   resultado["hb_aa"], resultado["hb_cc"], resultado["hb_ss"], resultado["hb_ac"], resultado["hb_as"] , resultado["hb_sc"],
                                   resultado["g6pd_bb"], resultado["g6pd_ap_ap"], resultado["g6pd_am_am"], resultado["g6pd_b_ap"], resultado["g6pd_b_am"], resultado["g6pd_ap_am"], resultado["g6pd_b"], resultado["g6pd_ap"], resultado["g6pd_am"],
                                   resultado["duffy_fybes_fybes"], resultado["duffy_fya_fya"], resultado["duffy_fyb_fyb"], resultado["duffy_fya_fybes"], resultado["duffy_fyb_fybes"], resultado["duffy_fya_fyb"] ]])
    prediction_np = predictor_modelo12.predict(new_data_point_np)

    nuevo_dato = {
        "tipo_prueba":12,
        "sexo": sexo,
        "edad_continua": edad_continua,
        "comuna_raw": comuna_raw,
        "hb_raw": hb_raw,
        "g6pd_raw": g6pd_raw,
        "duffy_raw": duffy_raw,
        "prediccion":  prediction_np[0]
    }

    resultado = coleccion2.insert_one(nuevo_dato)

    return {
        #"mensaje": resultado
        "mensaje": prediction_np[0]
    }



@app.post("/predictor_prueba_quince")
def inicio(sexo: int, edad_continua: int, comuna_raw:int, hb_raw:int, duffy_raw:int, btal29_raw:int, btal88_raw:int, g6pd_raw:int ):

    resultado = preparar_datos(
        sexo=sexo,
        edad_continua=edad_continua,
        comuna_raw=comuna_raw,
        hb_raw=hb_raw,
        g6pd_raw=g6pd_raw,
        duffy_raw=duffy_raw,
        btal29_raw=btal29_raw,
        btal88_raw=btal88_raw
    )

    new_data_point_np = np.array([[resultado["sexo"],
                                   resultado["edad_continua"], resultado["edad_g1"], resultado["edad_g2"], resultado["edad_g3"],
                                   resultado["comuna_raw"],
                                   resultado["hb_aa"], resultado["hb_cc"], resultado["hb_ss"], resultado["hb_ac"], resultado["hb_as"] , resultado["hb_sc"],
                                   resultado["g6pd_bb"], resultado["g6pd_ap_ap"], resultado["g6pd_am_am"], resultado["g6pd_b_ap"], resultado["g6pd_b_am"], resultado["g6pd_ap_am"], resultado["g6pd_b"], resultado["g6pd_ap"], resultado["g6pd_am"],
                                   resultado["duffy_fybes_fybes"], resultado["duffy_fya_fya"], resultado["duffy_fyb_fyb"], resultado["duffy_fya_fybes"], resultado["duffy_fyb_fybes"], resultado["duffy_fya_fyb"],
                                   resultado["btal29_aa"], resultado["btal29_ag"], resultado["btal29_gg"],
                                   resultado["btal88_cc"], resultado["btal88_ct"], resultado["btal88_tt"] ]])
    prediction_np = predictor_modelo15.predict(new_data_point_np)


    nuevo_dato = {
        "tipo_prueba":15,
        "sexo": sexo,
        "edad_continua": edad_continua,
        "comuna_raw": comuna_raw,
        "hb_raw": hb_raw,
        "g6pd_raw": g6pd_raw,
        "duffy_raw": duffy_raw,
        "btal29_raw": btal29_raw,
        "btal88_raw": btal88_raw,
        "prediccion": prediction_np
    }

    resultado = coleccion2.insert_one(nuevo_dato)


    

    return {
        #"mensaje": resultado
        "mensaje": prediction_np[0]
    }

