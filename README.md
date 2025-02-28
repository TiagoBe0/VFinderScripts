**Manual VFScript**





El software desarrollado en el presente trabajo, denominado VacancyAnalysis, funciona para detectar, aislar y describir los defectos en una muestra atomística. Tras identificar estos defectos, el sistema extrae las características esenciales de cada uno, lo que permite alimentar tres modelos distintos de machine learning. Cada modelo se entrena para aproximar de manera precisa las vacancias asociadas a cada tipo de defecto. Está escrito en lenguaje Python 3.12.3 y depende de las librerías Numpy(1.23.0), Pandas(2.2.3), Scikit-learn(1.6.1) y XGBoost(2.1.4). Implementa los modificadores del software de código abierto OVITO (6.8.2.1) para extraer informacion relevante de aquellas regiones de la red cristalina que pueden presentar vacancias.


VacancyAnalysis se compone de tres fases de procesamiento ejecutadas de manera secuencial: primero,se generan defectos en una red perfecta de forma intencionada para entrenar los modelos predictivos; segundo, se aíslan los defectos y se extraen los datos relevantes; y tercero, se realiza el procesamiento final que culmina en las predicciones. Su arquitectura modular y adaptable permite una facil escaleabilidad.



VacancyAnalysis se presenta como una alternativa innovadora al algoritmo de Wigner-Seitz. Mientras que el método tradicional, basado en la generación de celdas de Voronoi, puede sobrecontar defectos en muestras altamente deformadas, VacancyAnalysis enfoca sus predicciones en el análisis detallado de las características clave de los nanoporos presentes en la red cristalina. Este enfoque permite una contabilización más precisa y generalizada de las vacancias, adaptándose eficazmente a una amplia variedad de muestras y condiciones experimentales.





VacancyAnalysis es un software avanzado de postprocesamiento de datos derivados de simulaciones atomísticas, diseñado para facilitar el trabajo del investigador. Al integrar la potencia de los modificadores de OVITO para la extracción de información crítica de la muestra atomística con técnicas de machine learning para interpretar estos datos, el software automatiza y agiliza el análisis de defectos. Esta combinación de metodologías tradicionales y modernas no solo optimiza el flujo de trabajo, sino que también incrementa la precisión en la identificación y cuantificación de vacancias, ofreciendo una herramienta robusta y versátil para el estudio y la optimización de materiales.



**REQUERIMIENTOS Y USO**

El archivo input\_params.py contiene toda la información necesaria para inicializar el algoritmo. Este archivo puede ser leído y modificado por cualquier editor de texto. El software requiere
un archivo de entrenamiento (idealmente una red perfecta sin deformacion) y un archivo con defectos en su red. Los archivos suministrados deben ser archivos dump, como aquellos generados por el software LAMMPS [100].\\
Una vez realizado el entrenamiento, estos datos quedan guardados en un archivo.json y pueden ser reutilizados para otras predicciones en caso de no disponer de una muestra sin deformacion.
Desde el script nombrado main.py es que podemos ejecutar la totalidad del algoritmo. Para esto solo debemos ejecutar el comando python3 main.py en una terminal situada en el directorio que contiene los archivos de VacancyAnalysis, la muestra de entrenamiento y la muestra defectuosa.

Se debe ejecutar en un entorno python que contenga a los scripts.py , la muestra de entrenamiento y la muestra defecuosa en formato dump con las columnas 'id' 'type' 'x' 'y' 'z'. Instalar los paquetes ovito, Scikit-Learn, XGBoost , Pandas ,Numpy  y pyplot (en caso de querer usar la funcionalidad para exportar figuras).
Luego se debe  iniciar la ejecucion con python3 ./VFScript run
**Resultados Exportados**

![Mapa de calor 3D](dump-finalCool_160000_3D_heatmap.png)
![Mapas de contorno](dump-finalCool_160000_contour_maps.png)
![Gráfico de barras de clústeres](dump-finalCool_160000_pop_cluster_bar.png)
