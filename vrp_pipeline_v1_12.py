import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import distance_matrix
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import traceback # Added for more detailed error logging
from sklearn.cluster import KMeans
from itertools import combinations
import math # Needed for ceil

# PASO 0: Constantes y funciones de ayuda
# ==============================================================================
# Latitud base para división Norte/Sur (Quito)
base_lat = -0.22985
# Umbral para división de grupos por diámetro en km
UMBRAL_DIAMETRO_KM = 30
# Coordenada base para cálculos de rumbo/cuadrante (Ej: Centro de Quito)
coord_base = (-0.186449, -78.502346) # Reemplazar con la coordenada base real si es diferente

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en kilómetros
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# PASO 1: Carga y preparación de datos
# ==============================================================================
def cargar_datos(archivo_cargado):
    try:
        df = pd.read_excel(archivo_cargado)
        # Convertir todas las columnas relevantes a tipos numéricos, manejando errores
        cols_to_numeric = ['latitude', 'longitude', 'demand'] # Asumiendo que 'demand' es el nombre de la columna
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Eliminar filas con NaN en columnas críticas después de la conversión
        df.dropna(subset=['latitude', 'longitude', 'demand'], inplace=True)
        if df.empty:
            st.warning("El archivo no contiene datos válidos o las columnas 'latitude', 'longitude', 'demand' tienen problemas.")
            return pd.DataFrame() # Devolver DataFrame vacío si no hay datos válidos
        return df
    except Exception as e:
        st.error(f"Error al cargar o procesar el archivo: {e}")
        return pd.DataFrame()

# PASO 2: Creación de la matriz de distancias
# ==============================================================================
def crear_matriz_distancias(coordenadas_df, tipo_distancia="EUCLIDEAN"):
    coords = coordenadas_df[['latitude', 'longitude']].values
    if tipo_distancia == "HAVERSINE":
        num_locations = coords.shape[0]
        dist_matrix = np.zeros((num_locations, num_locations))
        for i in range(num_locations):
            for j in range(num_locations):
                if i == j:
                    dist_matrix[i, j] = 0
                else:
                    dist_matrix[i, j] = haversine(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
        return dist_matrix.astype(int) # OR-Tools prefiere enteros
    else: # EUCLIDEAN (o cualquier otro soportado por distance_matrix)
        dist_matrix = distance_matrix(coords, coords, p=2) # p=2 para Euclidiana
        # Escalar si las coordenadas son grados lat/lon para evitar números muy pequeños si no se usa Haversine
        # Esto es una simplificación; Haversine es mejor para distancias geográficas reales.
        return (dist_matrix * 1000).astype(int) # Multiplicar para tener enteros más manejables

# PASO 3: Creación del modelo de datos para OR-Tools
# ==============================================================================
def crear_modelo_datos(matriz_distancias, demandas, capacidades_vehiculos, num_vehiculos, indice_deposito=0):
    data = {}
    data['distance_matrix'] = matriz_distancias
    data['demands'] = demandas.tolist() # Asegurar que es una lista
    data['vehicle_capacities'] = capacidades_vehiculos
    data['num_vehicles'] = num_vehiculos
    data['depot'] = indice_deposito
    return data

# PASO 4: Resolución del VRP con OR-Tools
# ==============================================================================
def resolver_vrp(data_model, tiempo_limite_seg=30):
    try:
        manager = pywrapcp.RoutingIndexManager(len(data_model['distance_matrix']),
                                               data_model['num_vehicles'], data_model['depot'])
        routing = pywrapcp.RoutingModel(manager)

        # Callback de distancia
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data_model['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Callback de demanda y dimensión de capacidad
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data_model['demands'][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data_model['vehicle_capacities'],
            True,  # start cumul to zero
            'Capacity')

        # Parámetros de búsqueda
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(tiempo_limite_seg)
        # search_parameters.log_search = True # Para depuración más detallada si es necesario

        solution = routing.SolveWithParameters(search_parameters)
        return manager, routing, solution
    except Exception as e:
        st.error(f"Error en la resolución del VRP con OR-Tools: {e}")
        st.text(traceback.format_exc())
        return None, None, None

# PASO 5: Extracción y formateo de rutas de la solución
# ==============================================================================
def extraer_rutas(manager, routing, solution, num_vehiculos, df_nodos_original, nombre_col_id='name'):
    rutas_formateadas = []
    distancia_total = 0
    carga_total_transportada = 0 # Para verificar la carga total

    if not solution:
        st.warning("No se encontró solución.")
        return [], 0, 0

    # Asegurarse de que el DataFrame de nodos tenga un índice reseteado si se va a usar iloc
    # O mejor, usar el índice original del DataFrame si es el que corresponde a los nodos de OR-Tools

    for vehicle_id in range(num_vehiculos):
        index = routing.Start(vehicle_id)
        ruta_nodos_indices = [] # Índices de nodos de OR-Tools
        ruta_distancia_vehiculo = 0
        ruta_carga_vehiculo = 0

        while not routing.IsEnd(index):
            node_index_manager = index # Este es el índice del Routing Manager
            node_index_original = manager.IndexToNode(node_index_manager) # Este es el índice original (0 para depot, 1+ para clientes)

            ruta_nodos_indices.append(node_index_original)

            # Sumar demanda del nodo actual (si no es el depósito inicial en la ruta)
            # La demanda del depósito es 0 y no debería sumarse a la carga del vehículo.
            if node_index_original != routing.Start(vehicle_id): # No sumar demanda del depósito al inicio
                 # Corrección: Usar la demanda del df_nodos_original que corresponde al node_index_original
                 # Asumiendo que df_nodos_original incluye el depósito en el índice 0 y luego los clientes
                 if node_index_original < len(df_nodos_original): # Chequeo de seguridad
                    ruta_carga_vehiculo += df_nodos_original.iloc[node_index_original]['demand']


            previous_index = index
            index = solution.Value(routing.NextVar(index))

            if index < routing.Size(): # Comprobar que el índice es válido antes de calcular el costo
                ruta_distancia_vehiculo += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            else: # Podría ser el final del índice si algo va mal o es un nodo final artificial
                pass

        # Añadir el nodo final (depósito) a la ruta de índices
        node_index_original_final = manager.IndexToNode(index)
        ruta_nodos_indices.append(node_index_original_final)

        # Mapear índices de nodos a nombres/IDs de clientes/depósito
        # df_nodos_original debe tener el depósito en la primera fila (índice 0)
        # y los clientes en las filas subsiguientes, coincidiendo con cómo se creó la matriz de distancias.
        if not ruta_nodos_indices or len(ruta_nodos_indices) <= 2: # Si la ruta está vacía o solo tiene depósito->depósito
            if len(ruta_nodos_indices) == 2 and ruta_nodos_indices[0] == ruta_nodos_indices[1] and ruta_nodos_indices[0] == manager.IndexToNode(routing.Start(vehicle_id)):
                 # Esto es una ruta vacía válida (depósito a depósito), no la mostramos usualmente.
                 continue # Saltar a la siguiente iteración del vehículo

        # Convertir índices de nodos a nombres usando df_nodos_original
        # Se asume que df_nodos_original tiene el depósito en la fila 0 y los clientes después
        nombres_ruta = []
        for node_idx_orig in ruta_nodos_indices:
            if node_idx_orig < len(df_nodos_original): # Chequeo de límites
                # Usar la columna correcta para el nombre/ID, por ejemplo 'name' o el índice
                nombre_nodo = df_nodos_original.iloc[node_idx_orig].get(nombre_col_id, f"Nodo {node_idx_orig}")
                nombres_ruta.append(nombre_nodo)
            else:
                nombres_ruta.append(f"Índice fuera de rango: {node_idx_orig}")

        if len(nombres_ruta) > 2 or (len(nombres_ruta) == 2 and nombres_ruta[0] != nombres_ruta[1]): # Solo si hay al menos un cliente
            rutas_formateadas.append({
                "Vehículo": vehicle_id + 1,
                "Ruta": " -> ".join(nombres_ruta),
                "Distancia": ruta_distancia_vehiculo,
                "Carga": ruta_carga_vehiculo,
                "Capacidad": manager.model().GetDimensionOrDie('Capacity').VehicleCapacity(vehicle_id) # Capacidad del vehículo
            })
            distancia_total += ruta_distancia_vehiculo
            carga_total_transportada += ruta_carga_vehiculo
        elif len(nombres_ruta) <= 2 : # Rutas con solo depósito o depósito -> cliente -> depósito mal formadas
            # Podría ser útil loggear esto para depuración si las rutas no son como se esperan
            # st.sidebar.write(f"Vehículo {vehicle_id+1}: Ruta vacía o solo depósito. Nodos: {nombres_ruta}")
            pass


    return rutas_formateadas, distancia_total, carga_total_transportada

# PASO 6: Visualización de resultados (tabla y mapa)
# ==============================================================================
def mostrar_resultados(df_rutas, df_nodos_vis, deposito_info, distancia_total_rutas, carga_total_final):
    st.subheader("Resultados de la Optimización")

    if df_rutas.empty:
        st.info("No se generaron rutas válidas o no hay clientes que atender.")
        return

    st.dataframe(df_rutas)
    st.metric(label="Distancia Total Recorrida (unidades de matriz)", value=f"{distancia_total_rutas:.2f}")
    st.metric(label="Carga Total Transportada", value=f"{carga_total_final:.2f}")

    # Preparación de datos para el mapa
    # df_nodos_vis debe contener todos los puntos a graficar (depósito y clientes)
    # con columnas 'latitude', 'longitude', 'name', y una columna 'type' para diferenciar

    # Crear el DataFrame para el mapa si no existe o está mal formado
    puntos_mapa = []
    # Añadir depósito
    puntos_mapa.append({
        'name': deposito_info.get('name', 'Depósito'),
        'latitude': deposito_info['latitude'],
        'longitude': deposito_info['longitude'],
        'type': 'Depósito',
        'size': 15, # Tamaño para el marcador del depósito
        'demand': 0 # Demanda del depósito es 0
    })

    # Añadir clientes desde df_nodos_vis (que debería ser df_clientes_filtrados)
    for _, row in df_nodos_vis.iterrows():
        puntos_mapa.append({
            'name': row['name'], # Asegúrate que 'name' exista
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'type': 'Cliente',
            'size': 8, # Tamaño para marcadores de clientes
            'demand': row['demand']
        })

    df_map_points = pd.DataFrame(puntos_mapa)

    # Validar que df_map_points no esté vacío y tenga las columnas necesarias
    if df_map_points.empty or not all(col in df_map_points.columns for col in ['latitude', 'longitude', 'name', 'type', 'size']):
        st.error("No se pudieron preparar los datos para el mapa. Verifique los datos de entrada.")
        return

    # Mapa base con todos los puntos (depósito y clientes)
    fig = px.scatter_mapbox(df_map_points,
                            lat="latitude",
                            lon="longitude",
                            color="type",
                            size="size",
                            text="name",
                            hover_name="name",
                            hover_data={"latitude":False, "longitude":False, "type":True, "size":False, "demand": True},
                            zoom=10, # Ajustar zoom inicial según sea necesario
                            height=600,
                            color_discrete_map={'Depósito': 'red', 'Cliente': 'blue'})
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    # Añadir líneas de ruta al mapa
    # Necesitamos mapear los nombres de la ruta de vuelta a coordenadas
    # df_nodos_original_con_deposito es el DataFrame que tiene todos los nodos (depósito y clientes)
    # y que fue usado para generar los índices de la matriz de distancias.

    # Crear un diccionario para buscar coordenadas por nombre
    # Se asume que 'df_nodos_original_con_deposito' está disponible y tiene 'name', 'latitude', 'longitude'
    # Este df debe ser el mismo que se usó para crear la matriz de distancias, con el depósito en el índice 0.
    # Aquí usamos df_map_points que ya tiene esta información consolidada.

    coords_map = {row['name']: (row['latitude'], row['longitude']) for _, row in df_map_points.iterrows()}

    for _, fila_ruta in df_rutas.iterrows():
        nombres_en_ruta = [s.strip() for s in fila_ruta["Ruta"].split("->")]
        lats_ruta = []
        lons_ruta = []
        for nombre_nodo in nombres_en_ruta:
            if nombre_nodo in coords_map:
                lat, lon = coords_map[nombre_nodo]
                lats_ruta.append(lat)
                lons_ruta.append(lon)
            else:
                st.warning(f"Nombre de nodo '{nombre_nodo}' en la ruta no encontrado en los datos del mapa.")

        if lats_ruta and lons_ruta:
            fig.add_trace(go.Scattermapbox(
                mode="lines+markers", # Mostrar marcadores en la línea de ruta también
                lon=lons_ruta,
                lat=lats_ruta,
                name=f"Vehículo {fila_ruta['Vehículo']}",
                line=dict(width=2),
                marker=dict(size=5) # Tamaño de los marcadores en la línea de ruta
            ))

    st.plotly_chart(fig, use_container_width=True)


# PASO 7: Interfaz de Streamlit y flujo principal
# ==============================================================================
def main():
    st.set_page_config(layout="wide", page_title="Optimizador VRP")
    st.title("🚚 Optimizador de Rutas de Vehículos (VRP)")

    # --- Carga de Archivo ---
    st.sidebar.header("1. Cargar Datos")
    archivo_cargado = st.sidebar.file_uploader("Cargar archivo Excel (.xlsx)", type=["xlsx"])

    df_clientes_original = pd.DataFrame() # Inicializar DataFrame vacío
    deposito_coords_default = {'latitude': 0.0, 'longitude': 0.0} # Para evitar errores si no hay datos

    if archivo_cargado:
        df_clientes_original = cargar_datos(archivo_cargado)
        if not df_clientes_original.empty:
            st.subheader("Vista Previa de Datos Cargados (Clientes)")
            st.dataframe(df_clientes_original.head())
            # Usar la media de las coordenadas de los clientes como default para el depósito si se ingresa manualmente
            deposito_coords_default['latitude'] = df_clientes_original['latitude'].mean()
            deposito_coords_default['longitude'] = df_clientes_original['longitude'].mean()
        else:
            st.info("El archivo está vacío o no tiene el formato esperado después del procesamiento inicial.")
            return # Detener si el archivo está vacío o es inválido
    else:
        st.info("Por favor, cargue un archivo Excel con los datos de los clientes.")
        return # Detener si no hay archivo

    # --- Configuración del Depósito ---
    st.sidebar.header("2. Configurar Depósito")
    opcion_deposito = st.sidebar.selectbox("Origen del depósito:",
                                           ["Ingresar manualmente", "Seleccionar de datos cargados"])
    deposito_info = {'name': 'Depósito Central'} # Nombre por defecto

    if opcion_deposito == "Ingresar manualmente":
        deposito_info['latitude'] = st.sidebar.number_input("Latitud del Depósito:",
                                                            format="%.6f",
                                                            value=deposito_coords_default['latitude'])
        deposito_info['longitude'] = st.sidebar.number_input("Longitud del Depósito:",
                                                             format="%.6f",
                                                             value=deposito_coords_default['longitude'])
    else: # Seleccionar de datos cargados
        if not df_clientes_original.empty and 'name' in df_clientes_original.columns:
            # Permitir seleccionar un cliente como depósito y luego eliminarlo de la lista de clientes
            nombre_deposito_seleccionado = st.sidebar.selectbox("Seleccionar un cliente como Depósito:",
                                                                df_clientes_original['name'].unique())
            deposito_seleccionado_df = df_clientes_original[df_clientes_original['name'] == nombre_deposito_seleccionado].iloc[0]
            deposito_info['latitude'] = deposito_seleccionado_df['latitude']
            deposito_info['longitude'] = deposito_seleccionado_df['longitude']
            deposito_info['name'] = deposito_seleccionado_df['name']
            # Excluir el depósito de la lista de clientes
            df_clientes = df_clientes_original[df_clientes_original['name'] != nombre_deposito_seleccionado].copy()
        else:
            st.sidebar.warning("No hay datos de clientes cargados o falta la columna 'name' para seleccionar un depósito.")
            st.sidebar.info("Por favor, ingrese el depósito manualmente o cargue datos válidos.")
            # Forzar entrada manual si no se puede seleccionar
            deposito_info['latitude'] = st.sidebar.number_input("Latitud del Depósito (manual):", format="%.6f", value=deposito_coords_default['latitude'])
            deposito_info['longitude'] = st.sidebar.number_input("Longitud del Depósito (manual):", format="%.6f", value=deposito_coords_default['longitude'])
            df_clientes = df_clientes_original.copy() # Usar todos los clientes si el depósito es manual y no se pudo seleccionar

    if 'df_clientes' not in locals(): # Asegurar que df_clientes exista
        df_clientes = df_clientes_original.copy()


    # --- Configuración de Vehículos y Capacidades ---
    st.sidebar.header("3. Configurar Vehículos")
    num_vehiculos = st.sidebar.number_input("Número de Vehículos:", min_value=1, value=2)

    # Capacidades de los vehículos: opción para capacidades uniformes o individuales
    tipo_capacidad = st.sidebar.radio("Tipo de Capacidad de Vehículos:", ["Uniforme", "Individual"])
    capacidades_vehiculos = []
    if tipo_capacidad == "Uniforme":
        capacidad_uniforme = st.sidebar.number_input("Capacidad por Vehículo:", min_value=1, value=100)
        capacidades_vehiculos = [capacidad_uniforme] * num_vehiculos
    else: # Individual
        for i in range(num_vehiculos):
            cap_v = st.sidebar.number_input(f"Capacidad Vehículo {i+1}:", min_value=1, value=100, key=f"cap_v_{i}")
            capacidades_vehiculos.append(cap_v)

    # --- Columna de Demanda ---
    st.sidebar.header("4. Configurar Demanda")
    if not df_clientes.empty:
        # Intentar encontrar 'demand' o la primera columna numérica como default
        default_demand_col_index = 0
        if 'demand' in df_clientes.columns:
            default_demand_col_index = df_clientes.columns.get_loc('demand')
        elif len(df_clientes.select_dtypes(include=np.number).columns) > 0:
            # Si 'demand' no existe, tomar la primera columna numérica como opción (excluyendo lat/lon si es posible)
            numeric_cols = df_clientes.select_dtypes(include=np.number).columns
            potential_demand_cols = [col for col in numeric_cols if col not in ['latitude', 'longitude']]
            if potential_demand_cols:
                 default_demand_col_index = df_clientes.columns.get_loc(potential_demand_cols[0])
            else: # Si solo hay lat/lon numéricos, usar la primera columna general
                 default_demand_col_index = 0


        columna_demanda_nombre = st.sidebar.selectbox("Seleccionar columna de Demanda:",
                                                      df_clientes.columns,
                                                      index=default_demand_col_index)
        # Asegurar que la columna de demanda seleccionada sea numérica y no tenga NaNs
        if pd.api.types.is_numeric_dtype(df_clientes[columna_demanda_nombre]):
            if df_clientes[columna_demanda_nombre].isnull().any():
                st.sidebar.warning(f"La columna de demanda '{columna_demanda_nombre}' tiene valores faltantes. Se reemplazarán con 0.")
                df_clientes[columna_demanda_nombre] = df_clientes[columna_demanda_nombre].fillna(0)
            df_clientes['demand'] = df_clientes[columna_demanda_nombre] # Usar 'demand' internamente
        else:
            st.sidebar.error(f"La columna '{columna_demanda_nombre}' no es numérica. Por favor, seleccione una columna numérica para la demanda.")
            return
    else:
        st.sidebar.warning("Cargue datos de clientes para configurar la demanda.")
        return

    # --- Opciones Adicionales (Ej: tipo de distancia) ---
    st.sidebar.header("5. Opciones Adicionales")
    tipo_distancia = st.sidebar.selectbox("Tipo de Cálculo de Distancia:", ["EUCLIDEAN", "HAVERSINE"], index=1)
    tiempo_limite_opt = st.sidebar.slider("Tiempo Límite para Optimización (segundos):", 1, 60, 10)


    # --- Botón de Ejecución ---
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Ejecutar Optimización VRP", type="primary"):
        if df_clientes.empty:
            st.error("No hay datos de clientes para procesar. Verifique el archivo cargado y la configuración del depósito.")
            return
        if 'demand' not in df_clientes.columns or df_clientes['demand'].isnull().any():
            st.error("La columna de demanda no está configurada correctamente o tiene valores nulos. Verifique la configuración.")
            return

        # PASO A: Preparar DataFrame final para la matriz de distancias (depósito + clientes)
        # ------------------------------------------------------------------------------------
        df_deposito = pd.DataFrame([deposito_info]) # Convertir info del depósito a DataFrame
        if 'demand' not in df_deposito.columns: # Asegurar que el depósito tenga columna 'demand' (con valor 0)
            df_deposito['demand'] = 0

        # Concatenar depósito al principio, luego los clientes. Resetear índice es crucial.
        df_nodos_final = pd.concat([df_deposito, df_clientes[['name', 'latitude', 'longitude', 'demand']]], ignore_index=True)

        # Verificar que 'name' exista en df_nodos_final, si no, crearla a partir del índice
        if 'name' not in df_nodos_final.columns:
            df_nodos_final['name'] = [f"Nodo_{i}" for i in df_nodos_final.index]
        df_nodos_final['name'] = df_nodos_final['name'].astype(str) # Asegurar que nombres sean strings
        df_nodos_final.fillna({'name': 'NombreDesconocido'}, inplace=True) # Rellenar nombres NaN


        # PASO B: Crear Matriz de Distancias
        # ----------------------------------
        matriz_dist = crear_matriz_distancias(df_nodos_final[['latitude', 'longitude']], tipo_distancia)

        # PASO C: Crear Modelo de Datos OR-Tools
        # --------------------------------------
        # Las demandas deben ser una lista donde el primer elemento es la demanda del depósito (0)
        demandas_lista = [0] + df_clientes['demand'].astype(int).tolist() # Depósito (0) + demandas de clientes

        # Asegurar que la longitud de demandas_lista coincida con el número de nodos en df_nodos_final
        if len(demandas_lista) != len(df_nodos_final):
            st.error(f"Error en la consistencia de datos: {len(demandas_lista)} demandas vs {len(df_nodos_final)} nodos.")
            # st.write("Demandas:", demandas_lista)
            # st.write("Nodos Finales:", df_nodos_final)
            # Intentar reconstruir demandas_lista directamente desde df_nodos_final para asegurar consistencia
            demandas_lista_corregida = df_nodos_final['demand'].astype(int).tolist()
            if len(demandas_lista_corregida) != len(df_nodos_final):
                 st.error("La corrección de demandas falló. No se puede continuar.")
                 return # No continuar si las demandas no son consistentes
            demandas_lista = demandas_lista_corregida # Usar la lista corregida

        modelo_datos = crear_modelo_datos(matriz_dist, np.array(demandas_lista), capacidades_vehiculos, num_vehiculos, indice_deposito=0)

        # PASO D: Resolver VRP
        # --------------------
        st.info(f"Iniciando optimización con {num_vehiculos} vehículos y tiempo límite de {tiempo_limite_opt}s...")
        manager, routing, solution = resolver_vrp(modelo_datos, tiempo_limite_seg=tiempo_limite_opt)

        # PASO E: Extraer y Mostrar Rutas
        # -------------------------------
        if solution:
            st.success("¡Optimización completada con éxito!")
            # Pasar df_nodos_final a extraer_rutas, ya que este contiene el depósito y los clientes en el orden correcto
            # y tiene la columna 'name' que se espera.
            rutas_df, dist_total, carga_total = extraer_rutas(manager, routing, solution, num_vehiculos, df_nodos_final, nombre_col_id='name')

            # df_clientes (sin el depósito) se usa para la visualización de puntos de clientes en el mapa.
            # deposito_info se usa para el punto del depósito en el mapa.
            mostrar_resultados(pd.DataFrame(rutas_df), df_clientes, deposito_info, dist_total, carga_total)
        else:
            st.error("No se pudo encontrar una solución para el VRP con los parámetros dados.")
            # Mostrar información de depuración si es útil
            # st.subheader("Información de Depuración del Modelo:")
            # st.json(modelo_datos, expanded=False) # Cuidado con matrices grandes

    else:
        st.sidebar.info("Configure los parámetros y haga clic en 'Ejecutar Optimización'.")

# PASO 6.1: Lógica de clusterización avanzada (Norte/Sur para Quito, Límite de 13 clientes con KMeans)
# ==============================================================================
def paso_6_1_logica_clusterizacion_avanzada(clientes_df, base_lat_param):
    """
    Procesa los grupos de clientes:
    1. Grupos <= 13 clientes se mantienen.
    2. Grupos 'QUITO' > 13 clientes se dividen en Norte/Sur.
       Estos subgrupos Norte/Sur luego se verifican contra el límite de 13 clientes.
    3. Cualquier grupo o subgrupo > 13 clientes se subdivide usando KMeans.
    Espera columnas: 'GID', 'Lat', 'Lon', 'CiudadCorregida'.
    """
    if not all(col in clientes_df.columns for col in ['GID', 'Lat', 'Lon', 'Demanda']): # 'CiudadCorregida' es opcional
        st.error("Paso 6.1: DataFrame debe contener 'GID', 'Lat', 'Lon', 'Demanda'. 'CiudadCorregida' es opcional pero útil.")
        return clientes_df

    processed_groups_dfs = [] # Lista para guardar los DataFrames de los grupos procesados

    for gid, group_df in clientes_df.groupby('GID'):
        # st.write(f"P6.1 Procesando GID: {gid}, Tamaño: {len(group_df)}")

        # Lista para los subgrupos que resultan de este GID original (puede ser el mismo grupo o varios)
        sub_groups_to_process_further = []

        # Condición 1: Quito North/South Division
        # Asumimos que 'CiudadCorregida' existe y está poblada. Si no, esta lógica no se activa para Quito.
        is_quito_group = False
        if 'CiudadCorregida' in group_df.columns:
            try:
                # Asegurarse que CiudadCorregida no tenga NaNs para .mode()
                ciudad_mode = group_df['CiudadCorregida'].dropna().mode()
                if not ciudad_mode.empty and ciudad_mode[0].upper() == 'QUITO':
                    is_quito_group = True
            except Exception as e: # Por si .mode() falla o algo inesperado
                # st.warning(f"No se pudo determinar la ciudad para GID {gid}: {e}")
                pass # Continuar como si no fuera Quito


        if is_quito_group and len(group_df) > 13: # Solo dividir Quito si es grande
            # st.write(f"GID {gid} es Quito y >13. Dividiendo N/S.")
            quito_norte = group_df[group_df['Lat'] > base_lat_param].copy()
            quito_sur = group_df[group_df['Lat'] <= base_lat_param].copy()

            if not quito_norte.empty:
                quito_norte['GID'] = f"{gid}_QN"
                sub_groups_to_process_further.append(quito_norte)
            if not quito_sur.empty:
                quito_sur['GID'] = f"{gid}_QS"
                sub_groups_to_process_further.append(quito_sur)

            if quito_norte.empty and quito_sur.empty: # Si el grupo original de Quito quedó vacío
                # st.warning(f"GID {gid} (Quito) resultó vacío después de la división N/S. Se mantiene el original.")
                sub_groups_to_process_further.append(group_df.copy())

        else: # No es Quito, o es Quito pero <= 13 (o no se pudo determinar ciudad)
            sub_groups_to_process_further.append(group_df.copy())

        # Condición 2: 13-Client Limit (KMeans Subdivision)
        # Aplicar a cada subgrupo resultante (original, o _QN, _QS)
        for sub_group_df in sub_groups_to_process_further:
            current_gid = sub_group_df['GID'].iloc[0] # GID del subgrupo actual
            # st.write(f"  P6.1 Subgrupo: {current_gid}, Tamaño: {len(sub_group_df)}")
            if len(sub_group_df) <= 13:
                # st.write(f"    Subgrupo {current_gid} tiene <= 13 clientes. Se mantiene.")
                processed_groups_dfs.append(sub_group_df)
            else:
                # st.write(f"    Subgrupo {current_gid} tiene > 13 clientes. Aplicando KMeans.")
                num_clientes = len(sub_group_df)
                k = math.ceil(num_clientes / 13.0)

                coords = sub_group_df[['Lat', 'Lon']].values
                # Manejar el caso donde k=1 aunque len > 13 (e.g., 14 clientes, k=ceil(14/13)=2, pero si fueran 13.1 k=2)
                # Si k=1 después de ceil, significa que el grupo es <=13, lo cual ya está manejado.
                # Pero si k=1 y len(sub_group_df) > 13, algo está mal. Kmeans n_clusters debe ser >= 2 en la práctica si se divide.
                # Sin embargo, KMeans puede tener n_clusters=1, simplemente asigna todos a un cluster.
                # Para nuestro propósito, si k=1, no deberíamos hacer KMeans.
                if k <= 1 : # Si k es 1, no necesitamos KMeans, el grupo ya cumple (o es un error de lógica)
                    # st.warning(f"Subgrupo {current_gid} resultó en k={k} para KMeans. No se dividirá por KMeans.")
                    processed_groups_dfs.append(sub_group_df)
                    continue

                kmeans = KMeans(n_clusters=int(k), random_state=0, n_init=10)
                sub_group_df['kmeans_cluster'] = kmeans.fit_predict(coords)

                # Crear nuevos GIDs para los clusters de KMeans
                for cluster_id in range(int(k)):
                    cluster_data = sub_group_df[sub_group_df['kmeans_cluster'] == cluster_id].copy()
                    if not cluster_data.empty:
                        cluster_data['GID'] = f"{current_gid}_K{cluster_id+1}"
                        cluster_data.drop(columns=['kmeans_cluster'], inplace=True)
                        processed_groups_dfs.append(cluster_data)
                # st.write(f"    Subgrupo {current_gid} dividido en {k} subgrupos por KMeans.")

    if not processed_groups_dfs:
        st.warning("Paso 6.1: No se procesaron grupos o el DataFrame resultante está vacío.")
        return clientes_df # Devolver original si no hay nada

    final_df = pd.concat(processed_groups_dfs, ignore_index=True)
    # st.success("Paso 6.1: Lógica de clusterización avanzada completada.")
    # st.write("DataFrame después de PASO 6.1 (primeras filas):", final_df.head())
    return final_df

# PASO 6.2: Main - Procesamiento de diámetro y llamada a PASO 6.1
# ==============================================================================
def paso_6_2_main_con_logica_diametro(clientes_df, umbral_diametro_km, base_lat_param_for_6_1):
    """
    1. Divide grupos si su diámetro máximo excede `umbral_diametro_km`.
    2. Llama a `paso_6_1_logica_clusterizacion_avanzada` sobre el resultado.
    Espera columnas: 'GID', 'Lat', 'Lon', 'CiudadCorregida' (opcional para paso 6.1).
    """
    if not all(col in clientes_df.columns for col in ['GID', 'Lat', 'Lon', 'Demanda']):
        st.error("Paso 6.2: DataFrame debe contener 'GID', 'Lat', 'Lon', 'Demanda'.")
        return clientes_df

    df_copy_6_2 = clientes_df.copy()
    grupos_post_diametro_split = []

    gids_unicos_6_2 = df_copy_6_2['GID'].unique()

    for gid_original_6_2 in gids_unicos_6_2:
        group_df_6_2 = df_copy_6_2[df_copy_6_2['GID'] == gid_original_6_2].copy()

        if len(group_df_6_2) <= 2:
            grupos_post_diametro_split.append(group_df_6_2)
            continue

        max_dist_km = 0
        # Filtrar clientes sin coordenadas válidas para el cálculo de distancia
        valid_coords_df = group_df_6_2.dropna(subset=['Lat', 'Lon'])
        if len(valid_coords_df) > 1:
            coords_6_2 = valid_coords_df[['Lat', 'Lon']].values
            for punto1, punto2 in combinations(coords_6_2, 2):
                dist = haversine(punto1[0], punto1[1], punto2[0], punto2[1])
                if dist > max_dist_km:
                    max_dist_km = dist

        # st.write(f"P6.2 GID: {gid_original_6_2}, Num Clientes: {len(group_df_6_2)}, Max Dist: {max_dist_km:.2f} km")

        if max_dist_km > umbral_diametro_km and len(group_df_6_2) > 2: # len > 2 es redundante si combinations funcionó
            # st.write(f"  P6.2 Dividiendo GID: {gid_original_6_2} (Max Dist: {max_dist_km:.2f} km)")
            coords_para_kmeans_6_2 = valid_coords_df[['Lat', 'Lon']].values # Usar solo clientes con coordenadas válidas

            # Aplicar KMeans solo al DataFrame con coordenadas válidas
            kmeans_6_2 = KMeans(n_clusters=2, random_state=0, n_init=10)
            valid_coords_df['sub_cluster_id_6_2'] = kmeans_6_2.fit_predict(coords_para_kmeans_6_2)

            # Mapear los resultados de vuelta al group_df_6_2 original si había filas con NaN coords
            # Esto es importante si queremos mantener todos los clientes, incluso los que no participaron en KMeans
            group_df_6_2 = pd.merge(group_df_6_2, valid_coords_df[['sub_cluster_id_6_2']], left_index=True, right_index=True, how='left')

            # Para los que no participaron en KMeans (NaN en sub_cluster_id_6_2), asignar a un cluster (e.g., 0 o -1)
            # o decidir cómo manejarlos. Por ahora, se les podría asignar un GID que indique esto.
            # O, más simple, solo se procesan los que tienen coordenadas. El prompt no especifica.
            # Asumamos que solo los clientes con coordenadas válidas son considerados para la división.
            # Los que no tienen coordenadas válidas se mantendrán en sus GID originales o se filtrarán antes.
            # El código actual con valid_coords_df y luego merge maneja esto bien si se quiere preservar todos.
            # Si no, el KMeans se haría sobre group_df_6_2.dropna(subset=['Lat','Lon']) y luego se concatenan.

            group_df_6_2.loc[group_df_6_2['sub_cluster_id_6_2'] == 0, 'GID'] = f"{gid_original_6_2}_D1"
            group_df_6_2.loc[group_df_6_2['sub_cluster_id_6_2'] == 1, 'GID'] = f"{gid_original_6_2}_D2"
            # Los que quedaron con GID original (si sub_cluster_id_6_2 es NaN) deben ser manejados
            # Si hay filas con NaN en 'sub_cluster_id_6_2', su GID no cambiará.
            # Esto es razonable: si un cliente no tiene coords, no participa en el split por diámetro.

            group_df_6_2.drop(columns=['sub_cluster_id_6_2'], inplace=True, errors='ignore')
            grupos_post_diametro_split.append(group_df_6_2)
        else:
            grupos_post_diametro_split.append(group_df_6_2)

    if not grupos_post_diametro_split:
        st.warning("Paso 6.2: No se procesaron grupos o el DataFrame intermedio está vacío.")
        # Si no hay nada, llamar a paso 6.1 con el df original de todas formas
        # para mantener el flujo esperado, aunque 6.1 también podría devolverlo vacío.
        return paso_6_1_logica_clusterizacion_avanzada(clientes_df, base_lat_param_for_6_1)

    result_after_6_2_splits = pd.concat(grupos_post_diametro_split, ignore_index=True)
    # st.success("Paso 6.2: División por diámetro completada.")
    # st.write("DataFrame después de PASO 6.2 (splits por diámetro, antes de llamar a 6.1):", result_after_6_2_splits.head())

    # Llamada crucial a PASO 6.1
    final_df = paso_6_1_logica_clusterizacion_avanzada(result_after_6_2_splits, base_lat_param_for_6_1)

    # st.success("Paso 6.2 (main): Ejecución completa, incluyendo llamada a Paso 6.1.")
    # st.write("DataFrame final después de PASO 6.2 y PASO 6.1 (primeras filas):", final_df.head())
    return final_df

# PASO 9: Funciones de ayuda para disolución y reasignación de grupos pequeños
# ==============================================================================
def bearing_from_base(base_coords, point_coords):
    """Calcula el bearing (rumbo) desde una coordenada base a un punto."""
    lat1, lon1 = np.radians(base_coords[0]), np.radians(base_coords[1])
    lat2, lon2 = np.radians(point_coords[0]), np.radians(point_coords[1])

    dLon = lon2 - lon1
    x = np.sin(dLon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    initial_bearing = np.arctan2(x, y)
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

def calcular_cuadrante(bearing_val):
    """Calcula el cuadrante geográfico a partir del bearing."""
    if bearing_val is None:
        return None
    if 0 <= bearing_val < 90: return 'NE'
    if 90 <= bearing_val < 180: return 'SE'
    if 180 <= bearing_val < 270: return 'SO' # SW en inglés
    if 270 <= bearing_val <= 360: return 'NO' # NW en inglés
    return None

def calcular_centroides_y_modo_ciudad_cuadrante(df_para_centroides):
    """
    Calcula información agregada por grupo: centroide, suma de demanda,
    ciudad modo y cuadrante modo.
    Espera columnas 'GID', 'Lat', 'Lon', 'Demanda'.
    'CiudadCorregida' y 'cuadrante' son opcionales pero usadas si existen.
    """
    if df_para_centroides.empty or 'GID' not in df_para_centroides.columns:
        return pd.DataFrame()

    aggregations = {
        'Lat': 'mean',
        'Lon': 'mean',
        'Demanda': 'sum'
    }
    if 'CiudadCorregida' in df_para_centroides.columns:
        aggregations['CiudadCorregida'] = lambda x: x.mode()[0] if not x.mode().empty else None
    if 'cuadrante' in df_para_centroides.columns:
        aggregations['cuadrante'] = lambda x: x.mode()[0] if not x.mode().empty else None

    centroides_df = df_para_centroides.groupby('GID').agg(aggregations).reset_index()

    rename_cols = {'Lat': 'centroid_lat', 'Lon': 'centroid_lon', 'Demanda': 'TotalDemanda'}
    if 'CiudadCorregida' in centroides_df.columns: # Solo renombrar si la agregación se hizo
        rename_cols['CiudadCorregida'] = 'CiudadModo'
    if 'cuadrante' in centroides_df.columns: # Solo renombrar si la agregación se hizo
        rename_cols['cuadrante'] = 'CuadranteModo'

    centroides_df.rename(columns=rename_cols, inplace=True)
    return centroides_df

# PASO 9: Lógica principal de disolución y reasignación
# ==============================================================================
def paso_9_dissolve_and_reassign(clientes_df_input, coord_base_param, umbral_demanda_minima=150):
    """
    Disuelve grupos con TotalDemanda < umbral_demanda_minima y reasigna sus clientes.
    """
    required_cols = ['GID', 'Lat', 'Lon', 'Demanda', 'Nombre'] # 'CiudadCorregida' es opcional
    if clientes_df_input.empty or not all(col in clientes_df_input.columns for col in required_cols if col != 'CiudadCorregida'):
        st.warning("PASO 9: DataFrame de entrada vacío o faltan columnas esenciales. Se omite el paso.")
        return clientes_df_input

    clientes_df_work = clientes_df_input.copy()

    # Calcular 'bearing' y 'cuadrante' para cada cliente
    if 'Lat' in clientes_df_work.columns and 'Lon' in clientes_df_work.columns:
        clientes_df_work['bearing'] = clientes_df_work.apply(
            lambda row: bearing_from_base(coord_base_param, (row['Lat'], row['Lon'])) if pd.notnull(row['Lat']) and pd.notnull(row['Lon']) else None, axis=1
        )
        clientes_df_work['cuadrante'] = clientes_df_work['bearing'].apply(calcular_cuadrante)
    else: # Si no hay Lat/Lon, no se pueden calcular bearing/cuadrante
        clientes_df_work['bearing'] = None
        clientes_df_work['cuadrante'] = None


    st.write("Iniciando PASO 9: Disolución y Reasignación de Grupos Pequeños...")
    iteracion_count = 0
    max_iteraciones = 5 # Para evitar bucles infinitos si algo va mal

    while iteracion_count < max_iteraciones:
        iteracion_count += 1
        st.write(f"  PASO 9 - Iteración {iteracion_count}:")

        centroides_actuales = calcular_centroides_y_modo_ciudad_cuadrante(clientes_df_work)
        if centroides_actuales.empty:
            st.warning("  PASO 9: No se pudieron calcular centroides. Terminando disolución.")
            break

        grupos_pequenos = centroides_actuales[centroides_actuales['TotalDemanda'] < umbral_demanda_minima]

        if grupos_pequenos.empty:
            st.success(f"  PASO 9: No hay más grupos pequeños (< {umbral_demanda_minima} demanda) para disolver. Proceso completado.")
            break

        st.write(f"    Encontrados {len(grupos_pequenos)} grupos pequeños para disolver en esta iteración.")
        # st.write("Grupos pequeños a disolver:", grupos_pequenos[['GID', 'TotalDemanda']])

        # Obtener lista de GIDs a disolver EN ESTA ITERACIÓN
        gids_a_disolver_esta_iteracion = grupos_pequenos['GID'].tolist()

        # Filtrar centroides destino: deben ser > umbral Y NO estar en la lista de disolución de ESTA iteración
        # Esto es crucial: no reasignar a un grupo que también se va a disolver en la misma tanda.
        centroides_destino_validos_global = centroides_actuales[
            (centroides_actuales['TotalDemanda'] >= umbral_demanda_minima) &
            (~centroides_actuales['GID'].isin(gids_a_disolver_esta_iteracion))
        ].copy() # .copy() para evitar SettingWithCopyWarning al añadir 'dist_a_cliente'

        if centroides_destino_validos_global.empty:
            st.warning("  PASO 9: No hay grupos destino válidos (> umbral y no pequeños). No se pueden reasignar clientes de grupos pequeños.")
            break # Salir del while si no hay a dónde reasignar

        clientes_reaisgnados_count_total_iteracion = 0
        for gid_disolver in gids_a_disolver_esta_iteracion:
            # st.write(f"      Procesando disolución de GID: {gid_disolver}")
            indices_clientes_a_reasignar = clientes_df_work[clientes_df_work['GID'] == gid_disolver].index

            if indices_clientes_a_reasignar.empty:
                # st.write(f"        GID {gid_disolver} no tiene clientes para reasignar (ya fueron movidos o es un GID fantasma en centroides).")
                continue

            for idx_cliente in indices_clientes_a_reasignar:
                cliente = clientes_df_work.loc[idx_cliente]
                lat_cliente, lon_cliente = cliente.get('Lat'), cliente.get('Lon')
                ciudad_cliente = cliente.get('CiudadCorregida')
                cuadrante_cliente = cliente.get('cuadrante')
                demanda_cliente = cliente.get('Demanda', 0) # Default a 0 si no hay demanda

                if pd.isnull(lat_cliente) or pd.isnull(lon_cliente):
                    # st.warning(f"Cliente {cliente.get('Nombre', idx_cliente)} en GID {gid_disolver} no tiene Lat/Lon. No se puede reasignar por distancia.")
                    continue # O asignar a un GID residual, o el más cercano por otro criterio si es posible

                # Calcular distancias a los centroides destino válidos
                # No es necesario recalcular centroides_destino_validos_global para cada cliente, solo distancias
                # y luego filtrar por ciudad/cuadrante.

                # Copiar para no modificar el DataFrame global de centroides válidos al añadir dist_a_cliente
                centroides_para_este_cliente = centroides_destino_validos_global.copy()

                # Si centroides_para_este_cliente está vacío (porque todos los grupos grandes fueron eliminados o algo así)
                if centroides_para_este_cliente.empty:
                    # st.warning(f"No hay centroides destino válidos para el cliente {cliente.get('Nombre', idx_cliente)} del GID {gid_disolver}.")
                    continue

                centroides_para_este_cliente['dist_a_cliente'] = centroides_para_este_cliente.apply(
                    lambda row: haversine(lat_cliente, lon_cliente, row['centroid_lat'], row['centroid_lon']), axis=1
                )

                grupo_destino_elegido = None
                # Lógica de reasignación por prioridad
                # 1. Misma Ciudad
                if ciudad_cliente and 'CiudadModo' in centroides_para_este_cliente.columns:
                    candidatos_ciudad = centroides_para_este_cliente[centroides_para_este_cliente['CiudadModo'] == ciudad_cliente]
                    if not candidatos_ciudad.empty:
                        grupo_destino_elegido = candidatos_ciudad.loc[candidatos_ciudad['dist_a_cliente'].idxmin()]['GID']

                # 2. Mismo Cuadrante (si no hay match por ciudad)
                if not grupo_destino_elegido and cuadrante_cliente and 'CuadranteModo' in centroides_para_este_cliente.columns:
                    candidatos_cuadrante = centroides_para_este_cliente[centroides_para_este_cliente['CuadranteModo'] == cuadrante_cliente]
                    if not candidatos_cuadrante.empty:
                        grupo_destino_elegido = candidatos_cuadrante.loc[candidatos_cuadrante['dist_a_cliente'].idxmin()]['GID']

                # 3. Global más cercano (si no hay match por ciudad ni cuadrante)
                if not grupo_destino_elegido:
                    if not centroides_para_este_cliente.empty: # Asegurar que aún haya candidatos
                         grupo_destino_elegido = centroides_para_este_cliente.loc[centroides_para_este_cliente['dist_a_cliente'].idxmin()]['GID']

                if grupo_destino_elegido:
                    clientes_df_work.loc[idx_cliente, 'GID_original'] = clientes_df_work.loc[idx_cliente, 'GID'] # Guardar GID original
                    clientes_df_work.loc[idx_cliente, 'GID'] = grupo_destino_elegido
                    clientes_reaisgnados_count_total_iteracion +=1
                    # Actualizar TotalDemanda del grupo destino en centroides_actuales para que
                    # las siguientes reasignaciones DENTRO DE ESTA MISMA ITERACIÓN lo vean.
                    # Esto es una aproximación, ya que el centroide no se mueve dinámicamente aquí.
                    # El recálculo completo de centroides_actuales al final de la iteración de gids_a_disolver_esta_iteracion es más robusto.
                    idx_centroide_destino = centroides_actuales[centroides_actuales['GID'] == grupo_destino_elegido].index
                    if not idx_centroide_destino.empty:
                         centroides_actuales.loc[idx_centroide_destino, 'TotalDemanda'] += demanda_cliente
                else:
                    # st.warning(f"No se encontró grupo destino para cliente {cliente.get('Nombre', idx_cliente)} de GID {gid_disolver}.")
                    # El cliente permanece en su GID original (que se supone será eliminado o marcado)
                    # O se podría asignar a un GID especial 'NO_REASIGNADO'
                    pass
            # st.write(f"      GID {gid_disolver} disuelto. Clientes reasignados (si es posible).")

        if clientes_reaisgnados_count_total_iteracion == 0 and not grupos_pequenos.empty :
            st.warning(f"  PASO 9: En la iteración {iteracion_count}, se encontraron {len(grupos_pequenos)} grupos pequeños, pero no se pudo reasignar ningún cliente. "
                       "Esto puede deberse a falta de grupos destino válidos o clientes sin coordenadas. El bucle terminará para evitar ciclos.")
            break

    if iteracion_count == max_iteraciones and not grupos_pequenos.empty:
         st.warning(f"  PASO 9: Se alcanzó el máximo de {max_iteraciones} iteraciones. Aún podrían existir grupos pequeños.")

    # Finalizar: Eliminar los GIDs disueltos (aquellos que no tienen clientes)
    # Opcionalmente, se puede filtrar clientes_df_work para quitar GIDs que ya no existen en centroides_actuales post-disolución
    # Por ahora, se asume que si un GID no tiene clientes, efectivamente está "disuelto".
    # El DataFrame clientes_df_work ya tiene los GIDs actualizados.

    st.success("PASO 9: Disolución y Reasignación de Grupos Pequeños completada.")
    # st.write("DataFrame después de PASO 9 (primeras filas):", clientes_df_work.head())
    # st.write("Últimos centroides calculados en PASO 9:", calcular_centroides_y_modo_ciudad_cuadrante(clientes_df_work).head())
    return clientes_df_work


# Modificación en la función main() para integrar los nuevos pasos
# ==============================================================================
def main():
    st.set_page_config(layout="wide", page_title="Optimizador VRP")
    st.title("🚚 Optimizador de Rutas de Vehículos (VRP)")

    # --- Carga de Archivo ---
    st.sidebar.header("1. Cargar Datos")
    archivo_cargado = st.sidebar.file_uploader("Cargar archivo Excel (.xlsx)", type=["xlsx"])

    df_clientes_original_raw = pd.DataFrame()
    deposito_coords_default = {'latitude': 0.0, 'longitude': 0.0}

    if archivo_cargado:
        df_clientes_original_raw = cargar_datos(archivo_cargado) # Esta función ya hace un pd.read_excel

        # Columnas esperadas por las funciones de clustering/división:
        # 'GID', 'Lat', 'Lon', 'Nombre', 'Demanda', 'CiudadCorregida' (opcional)
        # Columnas esperadas por el VRP solver (después de clustering):
        # 'name', 'latitude', 'longitude', 'demand' (y 'GID' se mantiene si existe)

        # Mapeo inicial de nombres de columna del archivo Excel a nombres internos estándar
        # El usuario debe asegurarse que su Excel tenga estas columnas o ajustar el mapeo
        rename_map_inicial = {
            'latitude': 'Lat',          # Nombre común en archivos de entrada
            'longitude': 'Lon',         # Nombre común en archivos de entrada
            'ID GRUPO': 'GID',          # Ejemplo de nombre de columna de grupo
            'GRUPO': 'GID',             # Otro ejemplo
            'Grupo': 'GID',             # Otro ejemplo
            'name': 'Nombre',           # Nombre común para identificador de cliente
            'NOMBRE': 'Nombre',         # Otro ejemplo
            'CLIENTE': 'Nombre',        # Otro ejemplo
            'demand': 'Demanda',        # Nombre común para demanda
            'DEMANDA': 'Demanda',       # Otro ejemplo
            'CIUDAD': 'CiudadCorregida', # Ejemplo de columna de ciudad
            'Ciudad': 'CiudadCorregida'
        }
        df_clientes_original = df_clientes_original_raw.rename(columns={k: v for k, v in rename_map_inicial.items() if k in df_clientes_original_raw.columns})

        if not df_clientes_original.empty:
            # Verificar y convertir columnas numéricas esenciales
            cols_numericas_esenciales = ['Lat', 'Lon', 'Demanda']
            for col_num in cols_numericas_esenciales:
                if col_num in df_clientes_original.columns:
                    df_clientes_original[col_num] = pd.to_numeric(df_clientes_original[col_num], errors='coerce')
                else:
                    st.error(f"Falta la columna numérica esencial '{col_num}' (o su mapeo) en el archivo.")
                    return
            df_clientes_original.dropna(subset=cols_numericas_esenciales, inplace=True)

            # Asegurar existencia de 'GID' y 'Nombre'
            if 'GID' not in df_clientes_original.columns:
                st.warning("Columna 'GID' no encontrada. Se asignará un GID por defecto ('GID_DEFAULT').")
                df_clientes_original['GID'] = 'GID_DEFAULT'
            df_clientes_original['GID'] = df_clientes_original['GID'].astype(str)

            if 'Nombre' not in df_clientes_original.columns:
                st.warning("Columna 'Nombre' (identificador de cliente) no encontrada. Se creará una a partir del índice.")
                df_clientes_original['Nombre'] = ['Cliente_' + str(i) for i in df_clientes_original.index]
            df_clientes_original['Nombre'] = df_clientes_original['Nombre'].astype(str)

            # 'CiudadCorregida' es opcional para paso_6_1, pero si existe, asegurar que sea string.
            if 'CiudadCorregida' in df_clientes_original.columns:
                df_clientes_original['CiudadCorregida'] = df_clientes_original['CiudadCorregida'].astype(str).str.upper()


            st.subheader("Vista Previa de Datos de Clientes (Nombres Internos)")
            st.dataframe(df_clientes_original.head())

            if not df_clientes_original.empty:
                deposito_coords_default['latitude'] = df_clientes_original['Lat'].mean()
                deposito_coords_default['longitude'] = df_clientes_original['Lon'].mean()
            else: # Si después de limpiar queda vacío
                 st.info("El archivo no contiene datos válidos después de la limpieza inicial.")
                 return
        else:
            st.info("El archivo está vacío o no tiene el formato esperado.")
            return
    else:
        st.info("Por favor, cargue un archivo Excel con los datos de los clientes.")
        return

    # --- Configuración del Depósito ---
    st.sidebar.header("2. Configurar Depósito")
    opcion_deposito = st.sidebar.selectbox("Origen del depósito:",
                                           ["Ingresar manualmente", "Seleccionar de datos cargados (por Nombre)"])

    deposito_info = {'name': 'Depósito Central', 'latitude': 0.0, 'longitude': 0.0, 'demand': 0} # VRP espera estos nombres

    df_clientes_para_procesar = df_clientes_original.copy() # Copia para no alterar el original mostrado

    if opcion_deposito == "Ingresar manualmente":
        deposito_info['latitude'] = st.sidebar.number_input("Latitud del Depósito:", format="%.6f", value=deposito_coords_default['latitude'])
        deposito_info['longitude'] = st.sidebar.number_input("Longitud del Depósito:", format="%.6f", value=deposito_coords_default['longitude'])
        # df_clientes_para_procesar ya tiene todos los clientes
    else: # Seleccionar de datos cargados
        if not df_clientes_original.empty and 'Nombre' in df_clientes_original.columns:
            nombre_deposito_seleccionado = st.sidebar.selectbox("Seleccionar un cliente como Depósito:",
                                                                df_clientes_original['Nombre'].unique())

            deposito_seleccionado_df = df_clientes_original[df_clientes_original['Nombre'] == nombre_deposito_seleccionado]
            if not deposito_seleccionado_df.empty:
                deposito_row = deposito_seleccionado_df.iloc[0]
                deposito_info['latitude'] = deposito_row['Lat']
                deposito_info['longitude'] = deposito_row['Lon']
                deposito_info['name'] = deposito_row['Nombre']
                # Excluir el depósito de la lista de clientes a procesar
                df_clientes_para_procesar = df_clientes_original[df_clientes_original['Nombre'] != nombre_deposito_seleccionado].copy()
            else:
                st.sidebar.error(f"No se encontró el cliente '{nombre_deposito_seleccionado}' para usar como depósito. Usando entrada manual.")
                opcion_deposito = "Ingresar manualmente" # Revertir a manual
                deposito_info['latitude'] = st.sidebar.number_input("Latitud del Depósito:", format="%.6f", value=deposito_coords_default['latitude'])
                deposito_info['longitude'] = st.sidebar.number_input("Longitud del Depósito:", format="%.6f", value=deposito_coords_default['longitude'])
        else:
            st.sidebar.warning("No hay datos de clientes o falta 'Nombre' para seleccionar depósito. Usando entrada manual.")
            opcion_deposito = "Ingresar manualmente" # Revertir a manual
            deposito_info['latitude'] = st.sidebar.number_input("Latitud del Depósito:", format="%.6f", value=deposito_coords_default['latitude'])
            deposito_info['longitude'] = st.sidebar.number_input("Longitud del Depósito:", format="%.6f", value=deposito_coords_default['longitude'])

    # --- PASO 6.2 y PASO 6.1 (Nueva Lógica) ---
    st.sidebar.header("Pre-procesamiento Avanzado de Grupos")
    aplicar_logica_clusters_avanzada = st.sidebar.checkbox("Aplicar Lógica de Clusterización Avanzada (Pasos 6.2 y 6.1)", value=True)

    df_clientes_listos_para_vrp_columnas_std = pd.DataFrame()

    if aplicar_logica_clusters_avanzada:
        if not df_clientes_para_procesar.empty:
            # df_clientes_para_procesar tiene columnas 'GID', 'Lat', 'Lon', 'Nombre', 'Demanda', 'CiudadCorregida' (opcional)
            # La función paso_6_2_main_con_logica_diametro espera estas y devuelve GIDs actualizados.
            df_clientes_tras_paso_6 = paso_6_2_main_con_logica_diametro(
                df_clientes_para_procesar,
                UMBRAL_DIAMETRO_KM,
                base_lat # base_lat global para el paso 6.1 interno
            )
            st.write("DataFrame después de Clusterización Avanzada (Pasos 6.2 -> 6.1):", df_clientes_tras_paso_6.head())
        else:
            st.warning("No hay clientes para el pre-procesamiento avanzado de grupos (PASO 6.x).")
            df_clientes_tras_paso_6 = df_clientes_para_procesar # Usar datos como estaban si no hay nada que procesar
    else: # No aplicar lógica avanzada PASO 6.x
        st.info("Se omite la lógica de clusterización avanzada (PASO 6.x).")
        df_clientes_tras_paso_6 = df_clientes_para_procesar.copy() # Usar datos como estaban

    # --- PASO 9: Disolución y Reasignación de Grupos Pequeños ---
    st.sidebar.header("Disolución de Grupos Pequeños (PASO 9)")
    aplicar_paso_9 = st.sidebar.checkbox("Aplicar PASO 9: Disolver grupos pequeños y reasignar clientes", value=True)

    df_clientes_tras_paso_9 = pd.DataFrame() # Inicializar

    if aplicar_paso_9:
        if not df_clientes_tras_paso_6.empty:
            # PASO 9 opera sobre el resultado de PASO 6.x.
            # Espera columnas: 'GID', 'Lat', 'Lon', 'Demanda', 'Nombre', 'CiudadCorregida' (opcional), 'cuadrante' (calculado dentro)
            # coord_base se define globalmente
            df_clientes_tras_paso_9 = paso_9_dissolve_and_reassign(df_clientes_tras_paso_6, coord_base)
            st.write("DataFrame después de PASO 9 (Disolución y Reasignación):", df_clientes_tras_paso_9.head())
        else:
            st.warning("No hay clientes para procesar en PASO 9 (entrada de PASO 6.x estaba vacía).")
            df_clientes_tras_paso_9 = df_clientes_tras_paso_6 # Usar datos como estaban
    else:
        st.info("Se omite PASO 9: Disolución de grupos pequeños.")
        df_clientes_tras_paso_9 = df_clientes_tras_paso_6.copy() # Usar datos como estaban


    # Renombrar columnas finales para el VRP Solver
    # Columnas esperadas por el VRP: 'name', 'latitude', 'longitude', 'demand'
    df_clientes_final_para_vrp = df_clientes_tras_paso_9.rename(columns={
        'Lat':'latitude', 'Lon':'longitude', 'Nombre':'name', 'Demanda':'demand'
    })

    # Mantener 'GID' y otras columnas si existen y son útiles para análisis posterior,
    # pero asegurarse que las columnas principales para el VRP estén con los nombres correctos.
    if 'GID' in df_clientes_final_para_vrp.columns:
        st.write("Columna GID se mantiene para referencia en datos finales:", df_clientes_final_para_vrp[['name', 'GID']].head())

    st.subheader("Clientes Listos para Optimización VRP (Después de todos los pre-procesamientos)")
    st.dataframe(df_clientes_final_para_vrp.head())


    # --- Configuración de Vehículos y Capacidades ---
    st.sidebar.header("3. Configurar Vehículos")
    num_vehiculos = st.sidebar.number_input("Número de Vehículos:", min_value=1, value=2)

    tipo_capacidad = st.sidebar.radio("Tipo de Capacidad de Vehículos:", ["Uniforme", "Individual"])
    capacidades_vehiculos = []
    if tipo_capacidad == "Uniforme":
        capacidad_uniforme = st.sidebar.number_input("Capacidad por Vehículo:", min_value=1, value=100)
        capacidades_vehiculos = [capacidad_uniforme] * num_vehiculos
    else:
        for i in range(num_vehiculos):
            cap_v = st.sidebar.number_input(f"Capacidad Vehículo {i+1}:", min_value=1, value=100, key=f"cap_v_{i}")
            capacidades_vehiculos.append(cap_v)

    # --- Columna de Demanda (ya manejada como 'demand' internamente) ---
    st.sidebar.header("4. Configurar Demanda")
    # La columna de demanda ya fue estandarizada a 'demand' y verificada.
    # Solo necesitamos asegurar que exista en el df final para el VRP.
    if 'demand' not in df_clientes_final_para_vrp.columns:
        st.sidebar.error("La columna 'demand' no está presente en los datos finales de clientes. Verifique los pasos anteriores.")
        return
    st.sidebar.info(f"Se utilizará la columna 'demand' para las demandas de los clientes.")


    # --- Opciones Adicionales (Ej: tipo de distancia) ---
    st.sidebar.header("5. Opciones Adicionales")
    tipo_distancia = st.sidebar.selectbox("Tipo de Cálculo de Distancia:", ["EUCLIDEAN", "HAVERSINE"], index=1)
    tiempo_limite_opt = st.sidebar.slider("Tiempo Límite para Optimización (segundos):", 1, 60, 10)


    # --- Botón de Ejecución ---
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Ejecutar Optimización VRP", type="primary"):
        # df_clientes_final_para_vrp es el que se usará aquí
        if df_clientes_final_para_vrp.empty:
            st.error("No hay datos de clientes para procesar después del filtrado/preprocesamiento. Verifique el archivo cargado y la configuración.")
            return
        if 'demand' not in df_clientes_final_para_vrp.columns or df_clientes_final_para_vrp['demand'].isnull().any():
            st.error("La columna de demanda no está configurada correctamente o tiene valores nulos en los datos finales. Verifique la configuración y los datos.")
            return

        # PASO A: Preparar DataFrame final para la matriz de distancias (depósito + clientes)
        df_deposito_vrp = pd.DataFrame([deposito_info])
        # asegurar que deposito_info tenga 'name', 'latitude', 'longitude', 'demand'
        # df_deposito_vrp = df_deposito_vrp[['name', 'latitude', 'longitude', 'demand']] # Reordenar/seleccionar por si acaso

        # Columnas esperadas por el VRP: 'name', 'latitude', 'longitude', 'demand'
        # df_clientes_final_para_vrp ya debería tener estos nombres
        df_nodos_final = pd.concat([df_deposito_vrp, df_clientes_final_para_vrp[['name', 'latitude', 'longitude', 'demand']]], ignore_index=True)

        if 'name' not in df_nodos_final.columns:
            df_nodos_final['name'] = [f"Nodo_{i}" for i in df_nodos_final.index]
        df_nodos_final['name'] = df_nodos_final['name'].astype(str)
        df_nodos_final.fillna({'name': 'NombreDesconocido'}, inplace=True)


        # PASO B: Crear Matriz de Distancias
        matriz_dist = crear_matriz_distancias(df_nodos_final[['latitude', 'longitude']], tipo_distancia)

        # PASO C: Crear Modelo de Datos OR-Tools
        # Demandas: la primera es 0 (depósito), luego las de los clientes.
        demandas_lista = df_nodos_final['demand'].astype(int).tolist()

        if len(demandas_lista) != len(df_nodos_final):
            st.error(f"Error en la consistencia de datos de demanda: {len(demandas_lista)} demandas vs {len(df_nodos_final)} nodos.")
            return

        modelo_datos = crear_modelo_datos(matriz_dist, np.array(demandas_lista), capacidades_vehiculos, num_vehiculos, indice_deposito=0)

        # PASO D: Resolver VRP
        st.info(f"Iniciando optimización con {num_vehiculos} vehículos y tiempo límite de {tiempo_limite_opt}s...")
        manager, routing, solution = resolver_vrp(modelo_datos, tiempo_limite_seg=tiempo_limite_opt)

        # PASO E: Extraer y Mostrar Rutas
        if solution:
            st.success("¡Optimización completada con éxito!")
            rutas_df, dist_total, carga_total = extraer_rutas(manager, routing, solution, num_vehiculos, df_nodos_final, nombre_col_id='name')

            # Para mostrar_resultados, df_clientes_final_para_vrp (que no incluye el depósito) se usa para los puntos de clientes.
            mostrar_resultados(pd.DataFrame(rutas_df), df_clientes_final_para_vrp, deposito_info, dist_total, carga_total)
        else:
            st.error("No se pudo encontrar una solución para el VRP con los parámetros dados.")
    else:
        st.sidebar.info("Configure los parámetros y haga clic en 'Ejecutar Optimización'.")

if __name__ == "__main__":
    main()
