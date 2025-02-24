import os
import json
import numpy as np
from ovito.io import import_file, export_file
from ovito.modifiers import ExpressionSelectionModifier, DeleteSelectedModifier, ClusterAnalysisModifier, ConstructSurfaceModifier, InvertSelectionModifier
from input_params import CONFIG,PREDICTOR_COLUMNS
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
import math
class SurfaceProcessor:
    def __init__(self, config=CONFIG[0], json_path="outputs.json/key_archivos.json", radi=None, threshold_file="outputs.vfinder/key_single_vacancy.json"):
        self.config = config
        self.smoothing_level = config["smoothing level"]
        self.radi = radi if radi is not None else [2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.json_path = json_path
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.clusters_final = self.data.get("clusters_final", [])
        self.results_matrix = None
        with open(threshold_file, "r", encoding="utf-8") as f:
            threshold_data = json.load(f)
        self.min_area_threshold = threshold_data["surface_area"][0] / 2
        self.min_filled_volume_threshold = threshold_data["filled_volume"][0] / 2

    def process_surface_for_file(self, archivo):
        best_area = 0
        best_filled_volume = 0
        best_radius = None
        best_pipeline = None
        best_avg_distance = 0
        for r in self.radi:
            pipeline = import_file(archivo)
            pipeline.modifiers.append(ConstructSurfaceModifier(
                radius=r,
                smoothing_level=self.smoothing_level,
                identify_regions=True,
                select_surface_particles=True
            ))
            data = pipeline.compute()
            cluster_size = data.particles.count
            try:
                area = data.attributes['ConstructSurfaceMesh.surface_area']
            except Exception as e:
                area = 0
            try:
                filled_volume = data.attributes['ConstructSurfaceMesh.filled_volume']
            except Exception as e:
                filled_volume = 0
            # Calcular la distancia promedio al centro de masa del clúster
            positions = data.particles.positions
            if positions.shape[0] > 0:
                center = np.mean(positions, axis=0)
                avg_distance = np.mean(np.linalg.norm(positions - center, axis=1))
            else:
                avg_distance = 0
            if area > best_area:
                best_area = area
                best_filled_volume = filled_volume
                best_radius = r
                best_pipeline = pipeline
                best_avg_distance = avg_distance
        if best_area < self.min_area_threshold or best_filled_volume < self.min_filled_volume_threshold:
            return None, None, None, None, None, None
        return best_pipeline, best_radius, best_area, best_filled_volume, cluster_size, best_avg_distance

    def process_all_files(self):
        results = []
        for archivo in self.clusters_final:
            bp, br, ba, fv, num_atm, avg_dist = self.process_surface_for_file(archivo)
            if bp is not None:
                results.append([archivo, br, ba, fv, num_atm, avg_dist])
        self.results_matrix = np.array(results)
        return self.results_matrix

    def export_results(self, output_csv="outputs.json/resultados_procesados.csv"):
        if self.results_matrix is None:
            self.process_all_files()
        np.savetxt(output_csv, self.results_matrix, delimiter=",", fmt="%s",
                   header="archivo,mejor_radio,area,filled_volume,num_atm,mean_distance", comments="")


class ClusterDumpProcessor:
    def __init__(self, file_path: str, decimals: int = 5):
        self.file_path = file_path
        self.matriz_total = None
        self.header = None
        self.subset = None
        self.divisions_of_cluster = CONFIG[0]['divisions_of_cluster']

    def load_data(self):
        self.matriz_total = self.extraer_datos_completos(self.file_path)
        self.header = self.extraer_encabezado(self.file_path)
        if self.matriz_total.size == 0:
            raise ValueError(f"No se pudieron extraer datos de {self.file_path}")
        self.subset = self.matriz_total[:, 2:5]

    def calcular_dispersion(self, points: np.ndarray) -> float:
        if points.shape[0] == 0:
            return 0.0
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        return np.mean(distances)

    def process_clusters(self):
        centro_masa_global = np.mean(self.subset, axis=0)
        p1, p2, distancia_maxima = self.find_farthest_points(self.subset)
        dispersion = self.calcular_dispersion(self.subset)
        threshold = self.divisions_of_cluster
        etiquetas = self.aplicar_kmeans(self.subset, p1, p2, centro_masa_global, n_clusters=3)
        if etiquetas.shape[0] != self.matriz_total.shape[0]:
            raise ValueError("El número de etiquetas no coincide con la matriz total.")
        self.matriz_total[:, 5] = etiquetas

    def ejecutar_silhotte(self):
        lista_criticos = UtilidadesClustering.cargar_lista_archivos_criticos("outputs.json/key_archivos.json")
        processor = ClusterProcessor(self.archivo, decimals=4, threshold=1.2, max_iterations=10)
        processor.process_clusters()
        processor.export_updated_file()

    def separar_coordenadas_por_cluster(self) -> dict:
        if self.matriz_total is None:
            raise ValueError("Los datos no han sido cargados. Ejecuta load_data() primero.")
        clusters_dict = {0, 1, 2}
        etiquetas_unicas = np.unique(self.matriz_total[:, 5])
        for etiqueta in etiquetas_unicas:
            coords = self.matriz_total[self.matriz_total[:, 5] == etiqueta][:, 2:5]
            clusters_dict[int(etiqueta)] = coords
        return clusters_dict

    def export_updated_file(self, output_file: str = None):
        if output_file is None:
            output_file = f"{self.file_path}_actualizado.txt"
        fmt = ("%d %d %.5f %.5f %.5f %d")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(self.header)
                np.savetxt(f, self.matriz_total, fmt=fmt, delimiter=" ")
        except Exception as e:
            pass

    @staticmethod
    def cargar_lista_archivos_criticos(json_path: str) -> list:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                datos = json.load(f)
            return datos.get("clusters_criticos", [])
        except FileNotFoundError:
            return []
        except json.JSONDecodeError as e:
            return []

    @staticmethod
    def extraer_datos_completos(file_path: str) -> np.ndarray:
        datos = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            return np.array([])
        start_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("ITEM: ATOMS"):
                start_index = i + 1
                break
        if start_index is None:
            return np.array([])
        for line in lines[start_index:]:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                id_val = int(parts[0])
                type_val = int(parts[1])
                x = round(float(parts[2]), 5)
                y = round(float(parts[3]), 5)
                z = round(float(parts[4]), 5)
                cluster_val = int(parts[5])
                datos.append([id_val, type_val, x, y, z, cluster_val])
            except ValueError:
                continue
        return np.array(datos)

    @staticmethod
    def extraer_encabezado(file_path: str) -> list:
        encabezado = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    encabezado.append(line)
                    if line.strip().startswith("ITEM: ATOMS"):
                        break
        except Exception as e:
            pass
        return encabezado

    @staticmethod
    def aplicar_kmeans(coordenadas: np.ndarray, p1, p2, centro_masa_global, n_clusters: int) -> np.ndarray:
        from sklearn.cluster import KMeans
        if n_clusters == 2:
            init_centers = np.array([p1, p2])
        elif n_clusters == 3:
            init_centers = np.array([p1, p2, centro_masa_global])
        else:
            raise ValueError("Solo se admite n_clusters igual a 2 o 3.")
        kmeans = KMeans(n_clusters=n_clusters,
                        init=init_centers,
                        n_init=1,
                        max_iter=300,
                        tol=1,
                        random_state=42)
        etiquetas = kmeans.fit_predict(coordenadas)
        return etiquetas

    @staticmethod
    def find_farthest_points(coordenadas: np.ndarray):
        pts = np.array(coordenadas)
        n = pts.shape[0]
        if n < 2:
            return None, None, 0
        diffs = pts[:, None, :] - pts[None, :, :]
        distancias = np.sqrt(np.sum(diffs**2, axis=-1))
        idx = np.unravel_index(np.argmax(distancias), distancias.shape)
        distancia_maxima = distancias[idx]
        punto1 = pts[idx[0]]
        punto2 = pts[idx[1]]
        return punto1, punto2, distancia_maxima

def merge_clusters(labels, c1, c2):
    new_labels = np.copy(labels)
    new_labels[new_labels == c2] = c1
    return new_labels

def compute_dispersion(coords, labels):
    dispersion_dict = {}
    for c in np.unique(labels):
        mask = (labels == c)
        if not np.any(mask):
            dispersion_dict[c] = np.nan
            continue
        cluster_coords = coords[mask]
        center_of_mass = cluster_coords.mean(axis=0)
        distances = np.linalg.norm(cluster_coords - center_of_mass, axis=1)
        dispersion_dict[c] = distances.std()
    return dispersion_dict

def silhouette_mean(coords, labels):
    sil_vals = silhouette_samples(coords, labels)
    return np.mean(sil_vals)

def try_all_merges(coords, labels):
    clusters_unique = np.unique(labels)
    results = []
    for i in range(len(clusters_unique)):
        for j in range(i + 1, len(clusters_unique)):
            c1 = clusters_unique[i]
            c2 = clusters_unique[j]
            fused_labels = merge_clusters(labels, c1, c2)
            new_unique = np.unique(fused_labels)
            if len(new_unique) == 1:
                continue
            s_mean = silhouette_mean(coords, fused_labels)
            disp_dict = compute_dispersion(coords, fused_labels)
            disp_sum = np.nansum(list(disp_dict.values()))
            results.append(((c1, c2), fused_labels, s_mean, disp_dict, disp_sum))
    return results

def get_worst_cluster(dispersion_dict):
    worst_cluster = None
    max_disp = -1
    for c_label, d_val in dispersion_dict.items():
        if d_val > max_disp:
            max_disp = d_val
            worst_cluster = c_label
    return worst_cluster, max_disp

def kmeans_three_points(coords):
    center_of_mass = coords.mean(axis=0)
    distances = np.linalg.norm(coords - center_of_mass, axis=1)
    far_idxs = np.argsort(distances)[-2:]
    initial_centers = np.vstack([center_of_mass, coords[far_idxs[0]], coords[far_idxs[1]]])
    kmeans = KMeans(n_clusters=3, init=initial_centers, n_init=1, random_state=42)
    kmeans.fit(coords)
    sub_labels = kmeans.labels_
    sub_sil = np.mean(silhouette_samples(coords, sub_labels))
    sub_disp_dict = compute_dispersion(coords, sub_labels)
    sub_disp_sum = np.nansum(list(sub_disp_dict.values()))
    return sub_labels, sub_sil, sub_disp_dict, sub_disp_sum

def iterative_fusion_and_subdivision(coords, init_labels, threshold=1.2, max_iterations=10):
    labels = np.copy(init_labels)
    iteration = 0
    while iteration < max_iterations:
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1:
            break
        merge_candidates = try_all_merges(coords, labels)
        if not merge_candidates:
            break
        best_merge = min(merge_candidates, key=lambda x: x[4])
        (pair, fused_labels, fused_sil, fused_disp_dict, fused_disp_sum) = best_merge
        labels = fused_labels
        worst_cluster, max_disp = get_worst_cluster(fused_disp_dict)
        if max_disp > threshold:
            mask = (labels == worst_cluster)
            coords_worst = coords[mask]
            sub_labels, sub_sil, sub_disp_dict, sub_disp_sum = kmeans_three_points(coords_worst)
            offset = worst_cluster * 10
            new_sub_labels = offset + sub_labels
            new_labels_global = np.copy(labels)
            new_labels_global[mask] = new_sub_labels
            labels = new_labels_global
        iteration += 1
    return labels

class ClusterProcessor:
    def __init__(self,defect):
        self.configuracion = CONFIG[0]
        self.nombre_archivo = defect
        self.radio_sonda = self.configuracion['radius']
        self.smoothing_leveled = self.configuracion['smoothing level']
        self.cutoff_radius = self.configuracion['cutoff radius']
        self.outputs_dump = "outputs.dump"
        self.outputs_json = "outputs.json"
        os.makedirs(self.outputs_dump, exist_ok=True)
        os.makedirs(self.outputs_json, exist_ok=True)
    
    def run(self):
        pipeline = import_file(self.nombre_archivo)
        pipeline.modifiers.append(ConstructSurfaceModifier(radius=self.radio_sonda, smoothing_level=self.smoothing_leveled, identify_regions=True, select_surface_particles=True))
        pipeline.modifiers.append(InvertSelectionModifier())
        pipeline.modifiers.append(DeleteSelectedModifier())
        pipeline.modifiers.append(ClusterAnalysisModifier(cutoff=self.cutoff_radius, sort_by_size=True, unwrap_particles=True, compute_com=True))
        data = pipeline.compute()
        num_clusters = data.attributes["ClusterAnalysis.cluster_count"]
        datos_clusters = {"num_clusters": num_clusters}
        clusters_json_path = os.path.join(self.outputs_json, "clusters.json")
        with open(clusters_json_path, "w") as archivo:
            json.dump(datos_clusters, archivo, indent=4)
        key_areas_dump_path = os.path.join(self.outputs_dump, "key_areas.dump")
        try:
            export_file(pipeline, key_areas_dump_path, "lammps/dump", columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", "Cluster"])
            pipeline.modifiers.clear()
        except Exception as e:
            pass
        clusters = [f"Cluster=={i}" for i in range(1, num_clusters + 1)]
        for i, cluster_expr in enumerate(clusters, start=1):
            pipeline_2 = import_file(key_areas_dump_path)
            pipeline_2.modifiers.append(ClusterAnalysisModifier(cutoff=self.cutoff_radius, cluster_coloring=True, unwrap_particles=True, sort_by_size=True))
            pipeline_2.modifiers.append(ExpressionSelectionModifier(expression=cluster_expr))
            pipeline_2.modifiers.append(InvertSelectionModifier())
            pipeline_2.modifiers.append(DeleteSelectedModifier())
            output_file = os.path.join(self.outputs_dump, f"key_area_{i}.dump")
            try:
                export_file(pipeline_2, output_file, "lammps/dump", columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", "Cluster"])
                pipeline_2.modifiers.clear()
            except Exception as e:
                pass
        print(f"Número de áreas clave encontradas: {num_clusters}")
    
    @staticmethod
    def extraer_encabezado(file_path: str) -> list:
        encabezado = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    encabezado.append(line)
                    if line.strip().startswith("ITEM: ATOMS"):
                        break
        except Exception as e:
            pass
        return encabezado
    


class UtilidadesClustering:
    @staticmethod
    def cargar_lista_archivos_criticos(json_path: str) -> list:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                datos = json.load(f)
            return datos.get("clusters_criticos", [])
        except FileNotFoundError:
            print(f"El archivo {json_path} no existe.")
            return []
        except json.JSONDecodeError as e:
            print(f"Error al decodificar el archivo JSON: {e}")
            return []
    
    @staticmethod
    def extraer_datos_completos(file_path: str, decimals: int = 5) -> np.ndarray:
        datos = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"No se encontró el archivo: {file_path}")
            return np.array([])
        start_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("ITEM: ATOMS"):
                start_index = i + 1
                break
        if start_index is None:
            print(f"No se encontró la sección 'ITEM: ATOMS' en {file_path}.")
            return np.array([])
        for line in lines[start_index:]:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                id_val = int(parts[0])
                type_val = int(parts[1])
                x = round(float(parts[2]), decimals)
                y = round(float(parts[3]), decimals)
                z = round(float(parts[4]), decimals)
                cluster_val = int(parts[5])
                datos.append([id_val, type_val, x, y, z, cluster_val])
            except ValueError:
                continue
        return np.array(datos)
    
    @staticmethod
    def extraer_encabezado(file_path: str) -> list:
        encabezado = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    encabezado.append(line)
                    if line.strip().startswith("ITEM: ATOMS"):
                        break
        except Exception as e:
            print(f"Error al extraer encabezado de {file_path}: {e}")
        return encabezado
    
    @staticmethod
    def cargar_min_atoms(json_path: str) -> int:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                datos = json.load(f)
            vecinos = datos.get("cluster_size", [])
            if vecinos and isinstance(vecinos, list):
                return int(vecinos[0])
            else:
                print("No se encontró el valor de 'vecinos' en el archivo JSON, se usa valor por defecto 14.")
                return 14
        except Exception as e:
            print(f"Error al cargar {json_path}: {e}. Se usa valor por defecto 14.")
            return 14


def merge_clusters(labels, c1, c2):
    new_labels = np.copy(labels)
    new_labels[new_labels == c2] = c1
    return new_labels

def compute_dispersion(coords, labels):
    dispersion_dict = {}
    for c in np.unique(labels):
        mask = (labels == c)
        if not np.any(mask):
            dispersion_dict[c] = np.nan
            continue
        cluster_coords = coords[mask]
        center_of_mass = cluster_coords.mean(axis=0)
        distances = np.linalg.norm(cluster_coords - center_of_mass, axis=1)
        dispersion_dict[c] = distances.std()
    return dispersion_dict

def silhouette_mean(coords, labels):
    sil_vals = silhouette_samples(coords, labels)
    return np.mean(sil_vals)

def try_all_merges(coords, labels):
    clusters_unique = np.unique(labels)
    results = []
    for i in range(len(clusters_unique)):
        for j in range(i+1, len(clusters_unique)):
            c1 = clusters_unique[i]
            c2 = clusters_unique[j]
            fused_labels = merge_clusters(labels, c1, c2)
            new_unique = np.unique(fused_labels)
            if len(new_unique) == 1:
                continue
            s_mean = silhouette_mean(coords, fused_labels)
            disp_dict = compute_dispersion(coords, fused_labels)
            disp_sum = np.nansum(list(disp_dict.values()))
            results.append(((c1, c2), fused_labels, s_mean, disp_dict, disp_sum))
    return results

def get_worst_cluster(dispersion_dict):
    worst_cluster = None
    max_disp = -1
    for c_label, d_val in dispersion_dict.items():
        if d_val > max_disp:
            max_disp = d_val
            worst_cluster = c_label
    return worst_cluster, max_disp

def kmeans_three_points(coords):
    center_of_mass = coords.mean(axis=0)
    distances = np.linalg.norm(coords - center_of_mass, axis=1)
    far_idxs = np.argsort(distances)[-2:]
    initial_centers = np.vstack([center_of_mass, coords[far_idxs[0]], coords[far_idxs[1]]])
    kmeans = KMeans(n_clusters=3, init=initial_centers, n_init=1, random_state=42)
    kmeans.fit(coords)
    sub_labels = kmeans.labels_
    sub_sil = np.mean(silhouette_samples(coords, sub_labels))
    sub_disp_dict = compute_dispersion(coords, sub_labels)
    sub_disp_sum = np.nansum(list(sub_disp_dict.values()))
    return sub_labels, sub_sil, sub_disp_dict, sub_disp_sum

def kmeans_default(coords):
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(coords)
    sub_labels = kmeans.labels_
    sub_sil = np.mean(silhouette_samples(coords, sub_labels))
    sub_disp_dict = compute_dispersion(coords, sub_labels)
    sub_disp_sum = np.nansum(list(sub_disp_dict.values()))
    return sub_labels, sub_sil, sub_disp_dict, sub_disp_sum
def iterative_fusion_and_subdivision(coords, init_labels, threshold=1.5, max_iterations=10, min_atoms=14):
    labels = np.copy(init_labels)
    iteration = 0
    while iteration < max_iterations:
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1:
            # Sólo queda 1 clúster.
            break
        merge_candidates = try_all_merges(coords, labels)
        if not merge_candidates:
            break
        best_merge = min(merge_candidates, key=lambda x: x[4])
        (pair, fused_labels, fused_sil, fused_disp_dict, fused_disp_sum) = best_merge
        unique_after_merge = np.unique(fused_labels)
        if len(unique_after_merge) == 2:
            valid_merge = True
            for c in unique_after_merge:
                n_atoms = np.sum(fused_labels == c)
                if n_atoms < min_atoms:
                    valid_merge = False
                    print(f"  > La fusión {pair} produce el clúster {c} con solo {n_atoms} átomos (<{min_atoms}).")
                    break
            if not valid_merge:
                iteration += 1
                continue
        print(f"  Fusión elegida: {pair}, disp_total={fused_disp_sum:.3f}, sil={fused_sil:.3f}")
        labels = fused_labels

        worst_cluster, max_disp = get_worst_cluster(fused_disp_dict)
        print(f"  Clúster con mayor dispersión: {worst_cluster}, valor={max_disp:.3f}")
        mask = (labels == worst_cluster)
        n_atoms = np.sum(mask)
        if max_disp > threshold:
            if n_atoms < min_atoms:
                print(f"  > El clúster {worst_cluster} tiene solo {n_atoms} átomos (<{min_atoms}). No se subdivide.")
            else:
                print(f"  > Dispersión > {threshold} y {n_atoms} átomos. Intentando subdividir clúster {worst_cluster} con KMeans(3)...")
                coords_worst = coords[mask]
                # Dividimos en 3 subclusters
                sub_labels, sub_sil, sub_disp_dict, sub_disp_sum = kmeans_three_points(coords_worst)
                unique_sub = np.unique(sub_labels)
                # Ahora, intentamos fusionar dos de los tres subclusters para obtener dos clusters finales.
                candidate_fusions = []
                for i in range(len(unique_sub)):
                    for j in range(i+1, len(unique_sub)):
                        c1 = unique_sub[i]
                        c2 = unique_sub[j]
                        fused_sub = merge_clusters(sub_labels, c1, c2)
                        fused_unique = np.unique(fused_sub)
                        if len(fused_unique) != 2:
                            continue
                        counts = [np.sum(fused_sub == lab) for lab in fused_unique]
                        candidate_fusions.append(((c1, c2), fused_sub, counts))
                valid_fusion = None
                for candidate in candidate_fusions:
                    (fusion_pair, fused_sub, counts) = candidate
                    if all(count >= min_atoms for count in counts):
                        valid_fusion = candidate
                        break
                if valid_fusion is None:
                    print(f"  > La subdivisión del clúster {worst_cluster} produce dos clusters finales con menos de {min_atoms} átomos. Se revierte la subdivisión.")
                else:
                    (fusion_pair, fused_sub, counts) = valid_fusion
                    print(f"  > Subdivisión aceptada: se fusionan los subclusters {fusion_pair} resultando en dos clusters con conteos {counts}.")
                    # Asignamos estos nuevos labels al clúster worst_cluster en el global.
                    offset = worst_cluster * 10
                    new_sub_labels = offset + fused_sub
                    new_labels_global = np.copy(labels)
                    new_labels_global[mask] = new_sub_labels
                    labels = new_labels_global
        else:
            print(f"  > Dispersión <= {threshold}. No se subdivide.")
        iteration += 1
    print(f"\n[FIN] Iteraciones completadas = {iteration}. Clústeres finales: {np.unique(labels)}")
    return labels


class ClusterProcessorMachine:
    def __init__(self, file_path: str, threshold: float = 1.2, max_iterations: int = 10, min_atoms: int = None):
        self.file_path = file_path
        self.threshold = threshold
        self.max_iterations = max_iterations
        if min_atoms is None:
            self.min_atoms = UtilidadesClustering.cargar_min_atoms("outputs.vfinder/key_single_vacancy.json")
        else:
            self.min_atoms = min_atoms
        self.matriz_total = UtilidadesClustering.extraer_datos_completos(file_path)
        self.header = UtilidadesClustering.extraer_encabezado(file_path)
        pipeline = import_file(file_path)
        data = pipeline.compute()
        self.coords = data.particles.positions
        self.init_labels = data.particles["Cluster"].array
    
    def process_clusters(self):
        self.final_labels = iterative_fusion_and_subdivision(self.coords, self.init_labels, self.threshold, self.max_iterations, self.min_atoms)
        #print("\nClústeres finales en 'final_labels':", np.unique(self.final_labels))
        unique = np.unique(self.final_labels)
        mapping = {old: new for new, old in enumerate(unique)}
        self.final_labels = np.vectorize(mapping.get)(self.final_labels)
        #print("Clústeres remapeados:", np.unique(self.final_labels))
        if self.matriz_total.shape[0] == self.final_labels.shape[0]:
            self.matriz_total[:, 5] = self.final_labels
        else:
            print("¡Atención! La cantidad de filas en la matriz de datos y los clusters no coincide.")
    
    def export_updated_file(self, output_file: str = None):
        if output_file is None:
            output_file = f"{self.file_path}"
        fmt = ("%d %d %.5f %.5f %.5f %d")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(self.header)
                np.savetxt(f, self.matriz_total, fmt=fmt, delimiter=" ")
            #print("Datos exportados exitosamente a:", output_file)
        except Exception as e:
            print(f"Error al exportar {output_file}: {e}")

class ExportClusterList:
    def __init__(self, json_path="outputs.json/key_archivos.json"):
        self.json_path = json_path
        self.load_config()
    
    def load_config(self):
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.clusters_criticos = self.data.get("clusters_criticos", [])
        self.clusters_final = self.data.get("clusters_final", [])
    
    def save_config(self):
        self.data["clusters_final"] = self.clusters_final
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)
    
    def obtener_grupos_cluster(self, file_path):
        clusters = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        atom_header_line = None
        for i, line in enumerate(lines):
            if line.startswith("ITEM: ATOMS"):
                atom_header_line = line.strip()
                data_start = i + 1
                break
        if atom_header_line is None:
            raise ValueError("No se encontró la sección 'ITEM: ATOMS' en el archivo.")
        header_parts = atom_header_line.split()[2:]
        try:
            cluster_index = header_parts.index("Cluster")
        except ValueError:
            raise ValueError("La columna 'Cluster' no se encontró en la cabecera.")
        for line in lines[data_start:]:
            if line.startswith("ITEM:"):
                break
            if line.strip() == "":
                continue
            parts = line.strip().split()
            if len(parts) <= cluster_index:
                continue
            clusters.append(parts[cluster_index])
        unique_clusters = set(clusters)
        return unique_clusters, clusters

    def process_files(self):
        for archivo in self.clusters_criticos:
            try:
                unique_clusters, _ = self.obtener_grupos_cluster(archivo)
            except Exception as e:
                continue
            for i in range(0, len(unique_clusters)):
                pipeline = import_file(archivo)
                pipeline.modifiers.append(ExpressionSelectionModifier(expression=f"Cluster!={i}"))
                pipeline.modifiers.append(DeleteSelectedModifier())
                try:
                    nuevo_archivo = f"{archivo}.{i}"
                    export_file(pipeline, nuevo_archivo, "lammps/dump", 
                                columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", "Cluster"])
                    pipeline.modifiers.clear()
                    self.clusters_final.append(nuevo_archivo)
                except Exception as e:
                    pass
        self.save_config()

class KeyFilesSeparator:
    def __init__(self, config, clusters_json_path):
        self.config = config
        self.cluster_tolerance = config.get("cluster tolerance", 1.7)
        self.clusters_json_path = clusters_json_path
        self.lista_clusters_final = []
        self.lista_clusters_criticos = []
        self.num_clusters = self.cargar_num_clusters()

    def cargar_num_clusters(self):
        if not os.path.exists(self.clusters_json_path):
            return 0
        with open(self.clusters_json_path, "r", encoding="utf-8") as f:
            datos = json.load(f)
        num = datos.get("num_clusters", 0)
        return num

    def extraer_coordenadas(self, file_path):
        coordenadas = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            return coordenadas
        start_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("ITEM: ATOMS"):
                start_index = i + 1
                break
        if start_index is None:
            return coordenadas
        for line in lines[start_index:]:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                coordenadas.append((x, y, z))
            except ValueError:
                continue
        return coordenadas

    def calcular_centro_de_masa(self, coordenadas):
        arr = np.array(coordenadas)
        if arr.size == 0:
            return None
        centro = arr.mean(axis=0)
        return tuple(centro)

    def calcular_dispersion(self, coordenadas, centro_de_masa):
        if coordenadas is None or (hasattr(coordenadas, '__len__') and len(coordenadas) == 0) or centro_de_masa is None:
            return [], 0
        distancias = []
        cx, cy, cz = centro_de_masa
        for (x, y, z) in coordenadas:
            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
            distancias.append(d)
        dispersion = np.std(distancias)
        return distancias, dispersion

    def construir_matriz_coordenadas(self, archivo):
        coords = self.extraer_coordenadas(archivo)
        matriz = []
        for (x, y, z) in coords:
            matriz.append([x, y, z, 0])
        return np.array(matriz)

    def separar_archivos(self):
        for i in range(1, self.num_clusters + 1):
            ruta_archivo = f"outputs.dump/key_area_{i}.dump"
            coords = self.extraer_coordenadas(ruta_archivo)
            centroide = self.calcular_centro_de_masa(coords)
            distancias, dispersion = self.calcular_dispersion(coords, centroide)
            if dispersion > self.cluster_tolerance:
                self.lista_clusters_criticos.append(ruta_archivo)
            else:
                self.lista_clusters_final.append(ruta_archivo)

    def exportar_listas(self, output_path):
        datos_exportar = {
            "clusters_criticos": self.lista_clusters_criticos,
            "clusters_final": self.lista_clusters_final
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(datos_exportar, f, indent=4)

    def run(self):
        self.separar_archivos()
        self.exportar_listas("outputs.json/key_archivos.json")


if __name__ == "__main__":
    config = CONFIG[0]
    relax = config['relax']
    radius_training = config['radius_training']
    radius = config['radius']
    smoothing_level_training = config['smoothing_level_training']
    other_method = config['other method']
    
    processor = ClusterProcessor()
    processor.run()
    config = CONFIG[0]
    separator = KeyFilesSeparator(config, os.path.join("outputs.json", "clusters.json"))
    separator.run()
    json_path = "outputs.json/key_archivos.json"
    
    separator = KeyFilesSeparator(config, os.path.join("outputs.json", "clusters.json"))
    separator.run()
    json_path = "outputs.json/key_archivos.json"
    archivos = ClusterDumpProcessor.cargar_lista_archivos_criticos(json_path)
    for archivo in archivos:
        try:
            processor = ClusterDumpProcessor(archivo, decimals=5)
            processor.load_data()
            processor.process_clusters()
            processor.export_updated_file(f"{archivo}")
        except Exception as e:
            print(f"Error procesando {archivo}: {e}")
    lista_criticos = UtilidadesClustering.cargar_lista_archivos_criticos("outputs.json/key_archivos.json")
    for archivo in lista_criticos:
        processor = ClusterProcessorMachine(archivo, threshold=config['cluster tolerance'], max_iterations=config['iteraciones_clusterig'])
        processor.process_clusters()
        processor.export_updated_file()
    config = CONFIG[0]
    separator = KeyFilesSeparator(config, os.path.join("outputs.json", "clusters.json"))
    separator.run()
    processor_0 = ExportClusterList("outputs.json/key_archivos.json")
    processor_0.process_files()
    processor_1 = SurfaceProcessor()
    processor_1.process_all_files()
    processor_1.export_results()

           