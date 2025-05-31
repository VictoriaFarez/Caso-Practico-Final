#Librerias utilizadas
import os
import torch
import tempfile
import numpy as np
import pytesseract
import base64
import pandas as pd
import matplotlib.pyplot as plt
import pydotplus
from app import app
from flask import render_template, redirect, url_for, request, Flask, Blueprint, send_file
from prophet import Prophet
from io import BytesIO, StringIO
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama 
from PIL import Image
from scripts.ocr_recetas import procesar_receta
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import Image, SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')

bp = Blueprint('main', __name__)

@app.route('/')
# Página principal que muestra el nombre del proyecto
def index():
    nombre_farmacia = "PROYECTO PLATAFORMA INTELIGENTE PARA LA INTERPRETACIÓN DE RECETAS Y PREDICCIÓN DE VENTAS"
    return render_template('index.html', nombre_farmacia=nombre_farmacia)


@app.route('/ventas', methods=['GET'])
def ventas():
    # Cargar datos de ventas desde un CSV
    df = pd.read_csv('data/ventas.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])

    # Obtener mes seleccionado del formulario
    anio = request.args.get('anio')
    mes = request.args.get('mes')
    mes_filtrado_texto = None

    if anio and mes:
        try:
            anio = int(anio)
            mes = int(mes)
            # Filtrar dataframe por año y mes seleccionados
            df = df[(df['fecha'].dt.year == anio) & (df['fecha'].dt.month == mes)]

            # Lista de meses en español para mostrar en la interfaz
            meses_es = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
                        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
            mes_filtrado_texto = f"{meses_es[mes - 1]} {anio}"
        except:
            pass  # Evita errores por entradas inválidas
    
    # Si no hay datos, inicializar valores vacíos para la plantilla
    if df.empty:
        total_ventas = 0
        total_medicamentos = 0
        csv_data = []
    else:
        # Agrupar ventas diarias para modelado y gráfico
        df_agrupado = df.groupby('fecha')['cantidad_vendida'].sum().reset_index()
        df_agrupado.rename(columns={'fecha': 'ds', 'cantidad_vendida': 'y'}, inplace=True)

        # Crear y entrenar modelo Prophet para predicción de ventas futuras
        model = Prophet()
        model.fit(df_agrupado)

        # Generar dataframe para 30 días futuros y hacer predicción
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Calcular totales para mostrar
        total_ventas = (df['cantidad_vendida'] * df['precio']).sum()
        total_medicamentos = df['cantidad_vendida'].sum()
        csv_data = df.to_dict(orient='records')
        
        if not df.empty:
            # Graficar ventas diarias reales
            df_agrupado = df.groupby('fecha')['cantidad_vendida'].sum().reset_index()

            # Crear gráfico
            plt.figure(figsize=(10, 5))
            plt.plot(df_agrupado['fecha'], df_agrupado['cantidad_vendida'], marker='o')
            plt.title(f"Ventas reales por día - {mes_filtrado_texto}")
            plt.xlabel("Fecha")
            plt.ylabel("Cantidad Vendida")
            plt.grid(True)
            plt.tight_layout()

            # Guardar imagen en memoria
            img = BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            grafico_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
            plt.close()
        else:
            grafico_base64 = None

    return render_template(
        'ventas.html',
        total_ventas=total_ventas,
        total_medicamentos=total_medicamentos,
        csv_data=csv_data,
        mes_filtrado_texto=mes_filtrado_texto, # 🔹 Lo agregas al contexto
        grafico_base64=grafico_base64
    )



@app.route('/prediccion', methods=['GET'])
def prediccion():
    # Parámetro umbral para alerta de stock bajo, recibido vía query string
    umbral_stock = request.args.get('umbral', default=100, type=float)

    # Carga del archivo CSV con datos de ventas
    df = pd.read_csv('data/ventas.csv')
    df_original = df.copy()  # Para recuperar nombres reales

    if 'fecha_vencimiento' not in df.columns:
        raise KeyError("La columna 'fecha_vencimiento' no existe en el DataFrame")
    # Conversión a tipo datetime y creación de variables derivadas
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['dias_vencer'] = (pd.to_datetime(df['fecha_vencimiento']) - df['fecha']).dt.days
    df['mes'] = df['fecha'].dt.month
    df['dia'] = df['fecha'].dt.day

    # Codificación de variables categóricas a valores numéricos para ML
    le_codigo = LabelEncoder()
    le_nombre = LabelEncoder()
    le_dosis = LabelEncoder()

    df['codigo'] = le_codigo.fit_transform(df['codigo'].astype(str))
    df['nombre'] = le_nombre.fit_transform(df['nombre'].astype(str))
    df['dosis'] = le_dosis.fit_transform(df['dosis'].astype(str))

    # Definición de características (features) y variable objetivo (target)
    features = ['codigo', 'nombre', 'precio', 'cantidad_disponible', 'dias_vencer', 'dosis', 'mes', 'dia']
    X = df[features]
    y = df['cantidad_vendida']

    # División de datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     # --- Modelo Random Forest ---
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    # Evaluación del modelo con métricas comunes
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # --- Modelo Árbol de Decisión (para visualización e interpretación) ---
    dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)

    # Exportar árbol para graficar (visualización de decisiones del modelo)
    dot_data = StringIO()
    export_graphviz(
        dt_model, out_file=dot_data, feature_names=features,
        filled=True, rounded=True, special_characters=True
    )
    dot_data.seek(0)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    if graph is None:
        raise ValueError("Error al generar el gráfico del árbol de decisión.")

    # Convertir gráfico a imagen codificada para mostrar en HTML
    tree_img = BytesIO(graph.create_png())
    tree_img.seek(0)
    arbol_decision_img = base64.b64encode(tree_img.getvalue()).decode('utf-8')

    # --- Predicción actual para stock y generación de alertas ---
    stock_df = df[['codigo', 'nombre', 'precio', 'cantidad_disponible', 'fecha_vencimiento', 'dosis']].drop_duplicates()
    stock_df['dias_vencer'] = (pd.to_datetime(stock_df['fecha_vencimiento']) - pd.Timestamp.today()).dt.days
    stock_df['mes'] = pd.Timestamp.today().month
    stock_df['dia'] = pd.Timestamp.today().day

     # Función para codificar etiquetas nuevas sin error (manejo de datos desconocidos)
    def safe_transform(le, values):
        known_labels = set(le.classes_)
        values_str = values.astype(str)
        transformed = []
        for v in values_str:
            if v in known_labels:
                transformed.append(le.transform([v])[0])
            else:
                transformed.append(-1)
        return np.array(transformed)

    # Codificar características categóricas para el stock actual
    stock_df_encoded = stock_df.copy()
    stock_df_encoded['codigo'] = safe_transform(le_codigo, stock_df_encoded['codigo'])
    stock_df_encoded['nombre'] = safe_transform(le_nombre, stock_df_encoded['nombre'])
    stock_df_encoded['dosis'] = safe_transform(le_dosis, stock_df_encoded['dosis'])

    # Predicción de demanda con Random Forest para stock actual
    predicciones = rf_model.predict(stock_df_encoded[features])
    stock_df['demanda_predicha'] = np.round(predicciones)

    # Recuperar nombre real para mejor visualización
    stock_df = stock_df.merge(df_original[['codigo', 'nombre']], on='codigo', how='left', suffixes=('', '_real'))
    stock_df['nombre'] = stock_df['nombre_real']

    # Generar alertas donde la demanda predicha supere el stock + umbral
    stock_df_alertas = stock_df[stock_df['demanda_predicha'] > stock_df['cantidad_disponible'] + umbral_stock]
    alertas = stock_df_alertas[['nombre', 'demanda_predicha', 'cantidad_disponible']].to_dict(orient='records')

    # --- Visualización de resultados: gráfico Real vs Predicción ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(y_test.values, label='Real', alpha=0.7)
    ax.plot(y_pred, label='Predicho', alpha=0.7)

    # Añadir métricas al gráfico
    textstr = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}'
    props = dict(boxstyle='round', facecolor='lightgray', alpha=0.5)
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # Títulos y etiquetas
    ax.set_title('Random Forest - Real vs Predicción')
    ax.set_xlabel('Índice de muestra')
    ax.set_ylabel('Cantidad Vendida')
    ax.grid(True)
    ax.legend()
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    grafico = base64.b64encode(img.getvalue()).decode('utf-8')

    # --- Clasificación BCG para categorización de productos según ventas y rotación ---
    df_bcg = df_original.copy()
    df_bcg['fecha'] = pd.to_datetime(df_bcg['fecha'])

    bcg = df_bcg.groupby('nombre').agg({
        'cantidad_vendida': 'sum',
        'fecha': ['nunique', 'min', 'max']
    })
    bcg.columns = ['ventas_totales', 'dias_vendidos', 'fecha_min', 'fecha_max']
    bcg['dias_totales'] = (bcg['fecha_max'] - bcg['fecha_min']).dt.days + 1
    bcg['rotacion'] = bcg['dias_vendidos'] / bcg['dias_totales']

    # Variables para clasificar la demanda y rotación como altas o bajas
    bcg['demanda_alta'] = bcg['ventas_totales'] > bcg['ventas_totales'].median()
    bcg['rotacion_alta'] = bcg['rotacion'] > bcg['rotacion'].median()

    # Clasificación de cada producto en categorías BCG (Estrella, Dilema, Vaca Lechera, Perro)
    def clasificar(row):
        if row['demanda_alta'] and row['rotacion_alta']:
            return 'Estrella'
        elif row['demanda_alta']:
            return 'Dilema'
        elif row['rotacion_alta']:
            return 'Vaca Lechera'
        else:
            return 'Perro'

    bcg['categoria'] = bcg.apply(clasificar, axis=1)
    bcg_resultado = bcg.reset_index()[['nombre', 'ventas_totales', 'rotacion', 'categoria']]

    # --- Proyección de demanda a 30 días para simulación futura ---
    stock_futuro = df[['codigo', 'nombre', 'precio', 'cantidad_disponible', 'fecha_vencimiento', 'dosis']].drop_duplicates()
    stock_futuro['dias_vencer'] = (pd.to_datetime(stock_futuro['fecha_vencimiento']) - pd.Timestamp.today()).dt.days
    stock_futuro['mes'] = (pd.Timestamp.today() + pd.Timedelta(days=30)).month
    stock_futuro['dia'] = (pd.Timestamp.today() + pd.Timedelta(days=30)).day

    stock_futuro_codificado = stock_futuro.copy()
    stock_futuro_codificado['codigo'] = safe_transform(le_codigo, stock_futuro_codificado['codigo'])
    stock_futuro_codificado['nombre'] = safe_transform(le_nombre, stock_futuro_codificado['nombre'])
    stock_futuro_codificado['dosis'] = safe_transform(le_dosis, stock_futuro_codificado['dosis'])

# Recuperar nombre real desde df_original
    stock_futuro = stock_futuro.merge(df_original[['codigo', 'nombre']], on='codigo', how='left', suffixes=('', '_real'))
    stock_futuro['nombre'] = stock_futuro['nombre_real']

    stock_futuro['demanda_30d'] = np.round(rf_model.predict(stock_futuro_codificado[features]))
    simulacion_futura = stock_futuro[['nombre','dosis', 'cantidad_disponible', 'demanda_30d']].drop_duplicates()

# Filtro por categoría en Clasificación BCG
    categoria_filtrada = request.args.get('categoria', default='Todas')

    # Verificación
    print("Categoría recibida:", categoria_filtrada)
    print("Categorías disponibles:", bcg_resultado['categoria'].unique())

    if categoria_filtrada and categoria_filtrada.strip() != '' and categoria_filtrada != 'Todas':
        bcg_resultado = bcg_resultado[bcg_resultado['categoria'] == categoria_filtrada]

    return render_template(
        'prediccion.html',
        mae=round(mae, 2),
        rmse=round(rmse, 2),
        r2=round(r2, 2),
        alertas=alertas,
        grafico=grafico,
        arbol_decision_img=arbol_decision_img,
        umbral_stock=umbral_stock,
        bcg_resultado=bcg_resultado.to_dict(orient='records'),
        simulacion_futura=simulacion_futura.to_dict(orient='records'),
        categoria_seleccionada=categoria_filtrada
    )


@app.route('/agrupamiento')
def agrupamiento():
    # Cargar los datos de ventas
    df = pd.read_csv('data/ventas.csv')

    # Limpieza de datos
    df["cantidad_vendida"] = pd.to_numeric(df["cantidad_vendida"], errors='coerce')
    df = df.dropna(subset=["nombre", "cantidad_vendida"])

    # Agrupamiento por medicamento
    agrupado = df.groupby("nombre")["cantidad_vendida"].agg(["mean", "std"]).dropna()

    # El modelo KMeans necesita un número definido de clústeres
    n_medicamentos = len(agrupado)
    n_clusters = min(3, n_medicamentos)

    if n_clusters < 1:
        return "No hay suficientes medicamentos para agrupar."

    # Modelo de IA: KMeans clustering
    # Se utiliza KMeans para agrupar los medicamentos según su media y desviación estándar de ventas
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    agrupado["cluster"] = kmeans.fit_predict(agrupado)


    # Generar resumen de cada clúster
    # Se interpretan los grupos encontrados por el modelo con estadísticas promedio
    resumen_clusters = []

    for c in range(n_clusters):
        cluster_df = agrupado[agrupado["cluster"] == c]
        resumen_clusters.append({
            "cluster": c,
            "num_medicamentos": len(cluster_df),
            "promedio_ventas": round(cluster_df["mean"].mean(), 2),
            "promedio_std": round(cluster_df["std"].mean(), 2),
            "interpretacion": ""
        })

    # Interpretación de resultados del modelo
    # Clasifica los clústeres en categorías interpretables por humanos
    for r in resumen_clusters:
        if r["promedio_ventas"] > 100 and r["promedio_std"] < 20:
            r["interpretacion"] = "Ventas altas y estables"
        elif r["promedio_ventas"] < 50 and r["promedio_std"] > 30:
            r["interpretacion"] = "Ventas bajas y muy variables"
        else:
            r["interpretacion"] = "Ventas moderadas o variables"


    resultado = agrupado.reset_index().to_dict(orient='records')

    #Visualización de los resultados del clustering
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(agrupado["mean"], agrupado["std"], c=agrupado["cluster"], cmap='viridis', s=100)

    for idx, label in enumerate(agrupado.index):
        ax.annotate(label, (agrupado["mean"].iloc[idx], agrupado["std"].iloc[idx]), fontsize=9)

    ax.set_xlabel("Promedio de Ventas")
    ax.set_ylabel("Desviación Estándar")
    ax.set_title("Clustering de Medicamentos por Patrón de Ventas")

    img = BytesIO()
    plt.tight_layout()
    fig.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    return render_template(
        'agrupamiento.html',
        nombre=resultado,
        grafico_base64=img_base64,
        resumen_clusters=resumen_clusters
    )


# Ruta al binario de Tesseract si es necesario (ajústalo según tu instalación)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@app.route('/subir_receta', methods=['GET', 'POST'])
def subir_receta():
    if request.method == 'POST':
        if 'archivo_receta' not in request.files:
            return "No se subió ningún archivo"

        imagen = request.files['archivo_receta']
        if imagen.filename == '':
            return "Nombre de archivo vacío"

        # Crear carpeta si no existe
        carpeta_destino = os.path.join('static', 'recetas')
        os.makedirs(carpeta_destino, exist_ok=True)

        # Guardar imagen
        ruta_guardado = os.path.join(carpeta_destino, imagen.filename)
        imagen.save(ruta_guardado)

# --------------- MODELO DE IA: PROCESAMIENTO OCR ---------------
        # Se llama a la función 'procesar_receta', que aplica el modelo OCR
        # para extraer texto de la imagen y analizar la disponibilidad de medicamentos.
        texto, disponibilidad = procesar_receta(ruta_guardado)

        # Mostrar resultados
        return render_template('resultado_ocr.html', texto=texto, disponibilidad=disponibilidad)

    return render_template('subir_receta.html')


@app.route('/descargar_prediccion')
def descargar_prediccion():
    umbral_stock = request.args.get('umbral', default=100, type=float)

    # Carga del archivo CSV con las ventas históricas
    df = pd.read_csv('data/ventas.csv')
    df_original = df.copy()  # Para recuperar nombres reales

    # Quita filas con nombre vacío
    df = df.dropna(subset=['nombre'])

    # Elimina duplicados dejando la primera ocurrencia por código
    df = df.drop_duplicates(subset=['codigo'], keep='first')

    # Verificación de que no haya códigos repetidos con nombres distintos
    duplicados = df[['codigo', 'nombre']].drop_duplicates()
    if duplicados['codigo'].duplicated().any():
        conflictos = duplicados[duplicados['codigo'].duplicated(keep=False)]
        raise ValueError(f"Códigos con múltiples nombres:\n{conflictos}")

    # Procesamiento de fechas
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['dias_vencer'] = (pd.to_datetime(df['fecha_vencimiento']) - df['fecha']).dt.days
    df['mes'] = df['fecha'].dt.month
    df['dia'] = df['fecha'].dt.day

    # Codificación de variables categóricas con LabelEncoder
    le_codigo = LabelEncoder()
    le_nombre = LabelEncoder()
    le_dosis = LabelEncoder()

    df['codigo'] = le_codigo.fit_transform(df['codigo'].astype(str))
    df['nombre'] = le_nombre.fit_transform(df['nombre'].astype(str))
    df['dosis'] = le_dosis.fit_transform(df['dosis'].astype(str))

    # Selección de características (X) y variable objetivo (y)
    features = ['codigo', 'nombre', 'precio', 'cantidad_disponible', 'dias_vencer', 'dosis', 'mes', 'dia']
    X = df[features]
    y = df['cantidad_vendida']

    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenamiento del modelo Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predicción y evaluación
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # Generar gráfico Real vs Predicho
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(y_test.values, label='Real')
    ax.plot(y_pred, label='Predicho')
    ax.set_title("Real vs Predicción")
    ax.legend()
    img_bytes = BytesIO()
    fig.savefig(img_bytes, format='png')
    img_bytes.seek(0)

    # Generar árbol de decisión
    dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    dot_data = StringIO()
    export_graphviz(dt_model, out_file=dot_data, feature_names=features, filled=True, rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    arbol_img_bytes = BytesIO(graph.create_png())
    arbol_img_bytes.seek(0)

    # Preparar alertas
    stock_df = df[['codigo', 'nombre', 'precio', 'cantidad_disponible', 'fecha_vencimiento', 'dosis']].drop_duplicates()
    stock_df['dias_vencer'] = (pd.to_datetime(stock_df['fecha_vencimiento']) - pd.Timestamp.today()).dt.days
    stock_df['mes'] = pd.Timestamp.today().month
    stock_df['dia'] = pd.Timestamp.today().day

    def safe_transform(le, values):
        known_labels = set(le.classes_)
        values_str = values.astype(str)
        transformed = []
        for v in values_str:
            if v in known_labels:
                transformed.append(le.transform([v])[0])
            else:
                transformed.append(-1)
        return np.array(transformed)

    stock_df_encoded = stock_df.copy()
    stock_df_encoded['codigo'] = safe_transform(le_codigo, stock_df_encoded['codigo'])
    stock_df_encoded['nombre'] = safe_transform(le_nombre, stock_df_encoded['nombre'])
    stock_df_encoded['dosis'] = safe_transform(le_dosis, stock_df_encoded['dosis'])

    predicciones = rf_model.predict(stock_df_encoded[features])
    stock_df['demanda_predicha'] = np.round(predicciones)
    stock_df = stock_df.merge(df_original[['codigo', 'nombre']], on='codigo', how='left', suffixes=('', '_real'))
    stock_df['nombre'] = stock_df['nombre_real']
    alertas_df = stock_df[stock_df['demanda_predicha'] > stock_df['cantidad_disponible'] + umbral_stock]

    # === Generar PDF ===
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elementos = []

    estilos = getSampleStyleSheet()
    elementos.append(Paragraph("Informe de Predicción de Demanda", estilos['Title']))
    elementos.append(Spacer(1, 12))

    elementos.append(Paragraph(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}", estilos['Normal']))
    elementos.append(Spacer(1, 12))

    # Agregar gráfico de predicción
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(img_bytes.read())
        tmp.flush()
        elementos.append(Paragraph("Gráfico Real vs Predicción", estilos['Heading3']))
        elementos.append(Image(tmp.name, width=400, height=200))

    elementos.append(Spacer(1, 12))

    # Agregar imagen del árbol
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp2:
        tmp2.write(arbol_img_bytes.read())
        tmp2.flush()
        elementos.append(Paragraph("Árbol de Decisión", estilos['Heading3']))
        elementos.append(Image(tmp2.name, width=400, height=300))

    elementos.append(Spacer(1, 24))

    # Tabla de alertas
    if not alertas_df.empty:
        elementos.append(Paragraph("Alertas de Stock", estilos['Heading3']))
        data = [["Medicamento", "Demanda Predicha", "Stock"]]
        for _, row in alertas_df.iterrows():
            data.append([
                str(row['nombre']),
                str(int(row['demanda_predicha'])),
                str(int(row['cantidad_disponible']))
            ])

        t = Table(data, colWidths=[200, 150, 100])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 9)
        ]))
        elementos.append(t)
        elementos.append(Spacer(1, 12))
    else:
        elementos.append(Paragraph("No se generaron alertas de stock.", estilos['Normal']))

    # === Agregar tabla BCG ===
    # Crear tabla BCG: clasificar productos según demanda_predicha y cantidad_disponible
    bcg_data = [["Medicamento", "Demanda Predicha", "Stock", "Clasificación BCG"]]
    for _, row in stock_df.iterrows():
        demanda = row['demanda_predicha']
        stock = row['cantidad_disponible']
        # Regla simple de clasificación BCG:
        if demanda > stock and demanda > umbral_stock:
            categoria = "Estrella"
        elif demanda > stock:
            categoria = "Interrogante"
        elif stock > demanda:
            categoria = "Vaca Lechera"
        else:
            categoria = "Perro"
        bcg_data.append([str(row['nombre']), str(int(demanda)), str(int(stock)), categoria])

    elementos.append(Paragraph("Matriz BCG de Productos", estilos['Heading3']))
    t_bcg = Table(bcg_data, colWidths=[180, 120, 100, 120])
    t_bcg.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 9)
    ]))
    elementos.append(t_bcg)
    elementos.append(Spacer(1, 12))

    # === Agregar proyección futura ===
    # Por ejemplo, predecir demanda mensual para los próximos 3 meses sumando medias móviles o extrapolación simple

    futuro_meses = 3
    ult_fecha = df['fecha'].max()
    fechas_futuras = [ult_fecha + pd.DateOffset(months=i) for i in range(1, futuro_meses+1)]

    elementos.append(Paragraph("Proyección Futura de Demanda (Próximos 3 meses)", estilos['Heading3']))

    # Agregamos una columna extra para dosis
    proy_data = [["Mes", "Demanda Estimada Total", "Dosis Más Frecuente"]]

    demanda_mensual_prom = df.groupby(df['fecha'].dt.month)['cantidad_vendida'].mean().mean()
    dosis_frecuente = df['dosis'].mode()[0]  # ejemplo de dosis más frecuente

    for i, fecha in enumerate(fechas_futuras, start=1):
        mes_nombre = fecha.strftime("%B %Y")
        proy_data.append([mes_nombre, f"{demanda_mensual_prom:.0f}", dosis_frecuente])

    t_proy = Table(proy_data, colWidths=[200, 150, 150])
    t_proy.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 10)
    ]))
    elementos.append(t_proy)
    elementos.append(Spacer(1, 12))

    doc.build(elementos)
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name='informe_prediccion.pdf', mimetype='application/pdf')
