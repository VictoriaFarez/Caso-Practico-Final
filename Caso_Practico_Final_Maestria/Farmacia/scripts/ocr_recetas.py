#Librerias utilizadas
from PIL import Image
import pytesseract
import re
import os
import pandas as pd
import datetime
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# Función para traducir nombres de medicamentos del español al inglés usando GoogleTranslator.
# Se captura la excepción para evitar que una falla detenga el programa y se retorna el nombre original en minúsculas.
def traducir_al_ingles(nombre_espanol):
    try:
        traduccion = GoogleTranslator(source='es', target='en').translate(nombre_espanol)
        return traduccion.lower()
    except Exception as e:
        print(f"Error al traducir '{nombre_espanol}': {e}")
        return nombre_espanol.lower()

# Normaliza la dosis eliminando espacios y pasando a minúsculas para evitar diferencias en la comparación.
# Maneja el caso en que dosis pueda ser None o vacío.
def normalizar_dosis(dosis):
    if not dosis:
        return ""
    return re.sub(r'\s+', '', dosis).lower()

# Función para cargar el archivo CSV con las equivalencias de medicamentos traducidas al inglés.
# Se limpian los nombres de columnas para evitar errores por espacios o mayúsculas.

def cargar_equivalencias_traducidas(ruta_csv):
    # Cargar CSV y limpiar nombres de columnas
    df = pd.read_csv(ruta_csv, encoding='utf-8')
    df.columns = df.columns.str.strip().str.lower()  # Normaliza: quita espacios y pone en minúsculas
    print("Columnas cargadas:", df.columns.tolist())  # Diagnóstico

    # Asegúrate de que 'name' esté presente
    if 'name' in df.columns:
        df['name'] = df['name'].fillna('')
    else:
        raise KeyError("La columna 'name' no fue encontrada en el archivo.")

    return df

# Carga global de equivalencias y vectorizador TF-IDF para consulta rápida y eficiente.
df_equivalencias = cargar_equivalencias_traducidas("data/med_equivalencias.csv")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_equivalencias['name'])

# Busca el equivalente más parecido en inglés para un medicamento dado, usando similitud de coseno sobre vectores TF-IDF.
def buscar_equivalente(nombre_medicamento, dosis=""):
    nombre_traducido = traducir_al_ingles(nombre_medicamento)
    consulta = f"{nombre_traducido} {dosis}".strip().lower()
    query_vec = vectorizer.transform([consulta])
    similitudes = cosine_similarity(query_vec, tfidf_matrix).flatten()
    idx_max = similitudes.argmax()
    score_max = similitudes[idx_max]

    print(f"[DEBUG] Consulta: '{consulta}' - Score: {score_max}")

    if score_max < 0.3:  # Umbral de confianza, puedes ajustar
        return None

    mejor_equivalente = df_equivalencias.iloc[idx_max]
    # Extrae sustitutos, efectos secundarios y usos de las columnas correspondientes (en rangos predefinidos).
    sustitutos = [mejor_equivalente.get(f"substitute{i}") for i in range(5) if pd.notna(mejor_equivalente.get(f"substitute{i}"))]
    efectos_sec = [mejor_equivalente.get(f"sideEffect{i}") for i in range(42) if pd.notna(mejor_equivalente.get(f"sideEffect{i}"))]
    usos = [mejor_equivalente.get(f"use{i}") for i in range(5) if pd.notna(mejor_equivalente.get(f"use{i}"))]

    return {
        "nombre_encontrado": mejor_equivalente["name"],
        "sustitutos": sustitutos,
        "efectos_secundarios": efectos_sec,
        "usos": usos
    }

# Función alternativa para buscar sustitutos basándose solo en coincidencias cercanas con difflib.
# Útil cuando no se quiere traducir o hacer vectorización.
def buscar_sustitutos_por_nombre(nombre_medicamento):
    nombre_medicamento = nombre_medicamento.lower().strip()

    # Cargar archivo de equivalencias
    df = pd.read_csv("data/med_equivalencias.csv", encoding='utf-8')
    df.columns = df.columns.str.strip().str.lower()

    if 'name' not in df.columns:
        raise ValueError("El archivo no contiene la columna 'name'.")

    # Lista de nombres del archivo
    nombres_archivo = df['name'].astype(str).str.lower().str.strip().tolist()

    # Buscar coincidencia más cercana
    match = difflib.get_close_matches(nombre_medicamento, nombres_archivo, n=1, cutoff=0.8)

    if not match:
        return f"No se encontró un equivalente para '{nombre_medicamento}'."

    nombre_encontrado = match[0]

    fila = df[df['name'].str.lower().str.strip() == nombre_encontrado].iloc[0]


    sustitutos = [fila[f"substitute{i}"] for i in range(5) if f"substitute{i}" in fila and pd.notna(fila[f"substitute{i}"])]

    return {
        "nombre_encontrado": nombre_encontrado,
        "sustitutos": sustitutos
    }

# Función principal que procesa una imagen de receta médica:
# - Aplica OCR para extraer texto
# - Detecta medicamentos con posibles dosis mediante regex
# - Normaliza y filtra los datos
# - Verifica disponibilidad y obtiene información complementaria de sustitutos y efectos
def procesar_receta(ruta_imagen):
    texto = pytesseract.image_to_string(Image.open(ruta_imagen), lang='spa')
    # Expresión regular para detectar medicamentos con dosis (ej. Paracetamol 500mg)
    medicamentos_raw = re.findall(r'\b([A-Z][a-zA-ZáéíóúñüÁÉÍÓÚÑÜ]{3,})\s*(\d+\.?\d*\s?(mg|g|ml|mcg))?', texto)
    print("Texto extraído del OCR:", texto)
    # Palabras comunes irrelevantes para filtrar falsos positivos en nombres detectados
    stopwords = {"Tomar", "cada", "día", "mañana", "noche", "ml", "mg", "Nombre", "Edad", "Receta",
                 "Fecha", "Vencimiento", "Diagnóstico", "Teléfonos", "CLÍNICA"}
    # Carga stock actual de medicamentos
    ruta_stock = os.path.join('data', 'medicamentos_stock.csv')
    df_stock = pd.read_csv(ruta_stock, encoding='utf-8-sig')
    df_stock['nombre'] = df_stock['nombre'].astype(str).str.lower().str.strip()
    if 'dosis' in df_stock.columns:
        df_stock['dosis'] = df_stock['dosis'].astype(str).apply(normalizar_dosis)
    else:
        df_stock['dosis'] = ""
    # Lista combinada nombre + dosis para facilitar búsqueda con difflib
    lista_stock = df_stock[['nombre', 'dosis']].apply(lambda x: f"{x['nombre']} {x['dosis']}".strip(), axis=1).tolist()
    # Solo medicamentos únicos para no repetir procesamiento
    medicamentos_unicos = list(set(medicamentos_raw))
    disponibilidad = []

    for nombre, dosis, _ in medicamentos_unicos:
        if nombre in stopwords:
            continue

        nombre_lower = nombre.lower().strip()
        dosis_lower = normalizar_dosis(dosis)
        # Busca coincidencia en stock, usando dosis si está disponible para mayor precisión.
        if dosis_lower:
            consulta_completa = f"{nombre_lower} {dosis_lower}"
            match_cercano = difflib.get_close_matches(consulta_completa, lista_stock, n=1, cutoff=0.85)
        else:
            match_cercano = difflib.get_close_matches(nombre_lower, df_stock['nombre'].tolist(), n=1, cutoff=0.85)

        if match_cercano:
            if dosis_lower:
                nombre_encontrado = match_cercano[0]
                nombre_match, dosis_match = nombre_encontrado.rsplit(' ', 1) if ' ' in nombre_encontrado else (nombre_encontrado, "")
                match_df = df_stock[(df_stock['nombre'] == nombre_match) & (df_stock['dosis'] == dosis_match)]
            else:
                nombre_match = match_cercano[0]
                match_df = df_stock[df_stock['nombre'] == nombre_match]
                dosis_match = ""
            # Si hay stock disponible, se recupera info complementaria con búsqueda avanzada
            if not match_df.empty and match_df.iloc[0]['cantidad_disponible'] > 0:
                equivalencia = buscar_equivalente(nombre_lower, dosis_lower)
                if equivalencia:
                    sustitutos = ", ".join(equivalencia["sustitutos"])
                    efectos_secundarios = ", ".join(equivalencia["efectos_secundarios"])
                    usos = ", ".join(equivalencia["usos"])
                else:
                    sustitutos = ""
                    efectos_secundarios = ""
                    usos = ""

                info = {
                    'medicamento': nombre.capitalize(),
                    'dosis': dosis_lower,
                    'disponible': True,
                    'cantidad_disponible': int(match_df.iloc[0]['cantidad_disponible']),
                    'precio': float(match_df.iloc[0]['precio']),
                    'sustitutos': sustitutos,
                    'efectos_secundarios': efectos_secundarios,
                    'usos': usos
                }

            else:
                equivalencia = buscar_equivalente(nombre_lower, dosis_lower)
                if equivalencia:
                    info = {
                        'medicamento': nombre.capitalize(),
                        'dosis': dosis_lower,
                        'disponible': False,
                        'cantidad_disponible': 0,
                        'precio': 0.0,
                        'sustitutos': ", ".join(equivalencia["sustitutos"]),
                        'efectos_secundarios': ", ".join(equivalencia["efectos_secundarios"]),
                        'usos': ", ".join(equivalencia["usos"])
                    }
                else:
                    info = {
                        'medicamento': nombre.capitalize(),
                        'dosis': dosis_lower,
                        'disponible': False,
                        'cantidad_disponible': 0,
                        'precio': 0.0,
                        'sustitutos': "",
                        'efectos_secundarios': "",
                        'usos': ""
                    }

            disponibilidad.append(info)

     # Guardar la extracción OCR y el resumen en un archivo CSV con timestamp para histórico
    ruta_ocr = os.path.join('data', 'ocr_recetas.csv')
    os.makedirs(os.path.dirname(ruta_ocr), exist_ok=True)

    nueva_fila = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'texto_ocr': texto,
        'medicamentos_detectados': ', '.join([f"{d['medicamento']} {d['dosis']}".strip() if d['dosis'] else d['medicamento'] for d in disponibilidad])
    }
    # Si el archivo existe, añade sin cabecera, si no, crea con cabecera
    if os.path.exists(ruta_ocr):
        pd.DataFrame([nueva_fila]).to_csv(ruta_ocr, mode='a', header=False, index=False)
    else:
        pd.DataFrame([nueva_fila]).to_csv(ruta_ocr, mode='w', header=True, index=False)

    return texto, disponibilidad
