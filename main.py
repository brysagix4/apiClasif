from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from typing import Optional
import numpy as np

app = FastAPI()

clasificador_modelo5 = joblib.load("forest.joblib")
clasificador_modelo6 = joblib.load("forest.joblib")
clasificador_modelo9 = joblib.load("forest.joblib")
clasificador_modelo12 = joblib.load("forest.joblib")
clasificador_modelo15 = joblib.load("forest.joblib")


predictor_modelo5 = joblib.load("forest.joblib")
predictor_modelo6 = joblib.load("forest.joblib")
predictor_modelo9 = joblib.load("forest.joblib")
predictor_modelo12 = joblib.load("forest.joblib")
predictor_modelo15 = joblib.load("forest.joblib")



# ==========================================
# DATOS DE ENTRADA PARA PREPARAR
# ==========================================

class DatosPreparacion(BaseModel):
    sexo: Optional[int] = None
    edad_continua: Optional[int] = None
    comuna_raw: Optional[int] = None
    hb_raw: Optional[int] = None
    duffy_raw: Optional[int] = None
    btal29_raw: Optional[int] = None
    btal88_raw: Optional[int] = None
    g6pd_raw: Optional[int] = None


# ==========================================
# FUNCIÓN PARA PREPARAR LOS DATOS
# ==========================================

def preparar_datos(
    sexo=None,
    edad_continua=None,
    comuna_raw=None,
    hb_raw=None,
    duffy_raw=None,
    btal29_raw=None,
    btal88_raw=None,
    g6pd_raw=None
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

    if hb_raw is not None:
        datos["hb_raw"] = hb_raw

        nombres_hb = {
            1: "hb_aa",
            2: "hb_cc",
            3: "hb_ss",
            4: "hb_ac",
            5: "hb_as",
            6: "hb_sc"
        }

        if hb_raw in nombres_hb:
            for i, nombre in nombres_hb.items():
                datos[nombre] = int(hb_raw == i)

    if duffy_raw is not None:
        datos["duffy_raw"] = duffy_raw

        nombres_duffy = {
            1: "duffy_fybes_fybes",
            2: "duffy_fya_fya",
            3: "duffy_fyb_fyb",
            4: "duffy_fya_fybes",
            5: "duffy_fyb_fybes",
            6: "duffy_fya_fyb"
        }

        if duffy_raw in nombres_duffy:
            for i, nombre in nombres_duffy.items():
                datos[nombre] = int(duffy_raw == i)

    if btal29_raw is not None:
        datos["btal29_raw"] = btal29_raw

        nombres_btal29 = {
            1: "btal29_aa",
            2: "btal29_ag",
            3: "btal29_gg"
        }

        if btal29_raw in nombres_btal29:
            for i, nombre in nombres_btal29.items():
                datos[nombre] = int(btal29_raw == i)

    if btal88_raw is not None:
        datos["btal88_raw"] = btal88_raw

        nombres_btal88 = {
            1: "btal88_cc",
            2: "btal88_ct",
            3: "btal88_tt"
        }

        if btal88_raw in nombres_btal88:
            for i, nombre in nombres_btal88.items():
                datos[nombre] = int(btal88_raw == i)

    if g6pd_raw is not None:
        datos["g6pd_raw"] = g6pd_raw

        nombres_g6pd = {
            1: "g6pd_bb",
            2: "g6pd_ap_ap",
            3: "g6pd_am_am",
            4: "g6pd_b_ap",
            5: "g6pd_b_am",
            6: "g6pd_ap_am",
            7: "g6pd_b",
            8: "g6pd_ap",
            9: "g6pd_am"
        }

        if g6pd_raw in nombres_g6pd:
            for i, nombre in nombres_g6pd.items():
                datos[nombre] = int(g6pd_raw == i)

    return datos





# ==========================================
# ENDPOINT PARA PREPARAR DATOS
# ==========================================



# GET
@app.get("/")
def inicio():
    return {
        "mensaje": "Hola, esta es mi API funcionando"
    }



# POST
@app.post("/pruebaJsonEnviado")
def preparar(datos: DatosPreparacion):

    resultado = preparar_datos(
        sexo=datos.sexo,
        edad_continua=datos.edad_continua,
        comuna_raw=datos.comuna_raw,
        hb_raw=datos.hb_raw,
        duffy_raw=datos.duffy_raw,
        btal29_raw=datos.btal29_raw,
        btal88_raw=datos.btal88_raw,
        g6pd_raw=datos.g6pd_raw
    )

    return {
        "datos_preparados": resultado
    }




@app.post("/pruebaCombinada")
def inicio(sexo: int, edad_continua: float,comuna_raw:int):

    new_data_point_np = np.array([[sexo, edad_continua, comuna_raw, 0, 0, 9, 1.0]])
    prediction_np = clasificador_modelo5.predict(new_data_point_np)

    return {
        "mensaje": prediction_np[0]
    }


@app.post("/clasificador_prueba_cinco")
def inicio(sexo: int, edad_continua: int, comuna_raw:int, hb_raw:int):

    resultado = preparar_datos(
        sexo=sexo,
        edad_continua=edad_continua,
        comuna_raw=comuna_raw,
        hb_raw=hb_raw,
    )

    new_data_point_np = np.array([[resultado["sexo"], resultado["edad_continua"], resultado["edad_g1"], resultado["edad_g2"], resultado["edad_g3"], resultado["comuna_raw"], resultado["hb_raw"]]])
    prediction_np = clasificador_modelo5.predict(new_data_point_np)

    return {
        #"mensaje": resultado
        "mensaje": prediction_np[0]
    }



@app.post("/clasificador_prueba_seis")
def inicio(sexo: int, edad_continua: int, comuna_raw:int, hb_raw:int):

    resultado = preparar_datos(
        sexo=sexo,
        edad_continua=edad_continua,
        comuna_raw=comuna_raw,
        hb_raw=hb_raw,
    )

    new_data_point_np = np.array([[resultado["sexo"], resultado["edad_continua"], resultado["edad_g1"], resultado["edad_g2"], resultado["edad_g3"], resultado["comuna_raw"], resultado["hb_aa"], resultado["hb_cc"], resultado["hb_ss"], resultado["hb_ac"], resultado["hb_as"] , resultado["hb_sc"] ]])
    prediction_np = clasificador_modelo6.predict(new_data_point_np)

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
                                   resultado["btal29_raw"],
                                   resultado["btal88_raw"],
                                   resultado["btal29_aa"], resultado["btal29_ag"], resultado["btal29_gg"],
                                   resultado["btal88_cc"], resultado["btal88_ct"], resultado["btal88_tt"] ]])
    prediction_np = clasificador_modelo15.predict(new_data_point_np)

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
                                   resultado["btal29_raw"],
                                   resultado["btal88_raw"],
                                   resultado["btal29_aa"], resultado["btal29_ag"], resultado["btal29_gg"],
                                   resultado["btal88_cc"], resultado["btal88_ct"], resultado["btal88_tt"] ]])
    prediction_np = predictor_modelo15.predict(new_data_point_np)

    return {
        #"mensaje": resultado
        "mensaje": prediction_np[0]
    }

