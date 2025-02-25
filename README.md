
El software desarrollado en el presente trabajo, denominado VacancyAnalysis, funciona para detectar, aislar y describir los defectos en una muestra atomística. Tras identificar estos defectos, el sistema extrae las características esenciales de cada uno, lo que permite alimentar tres modelos distintos de machine learning. Cada modelo se entrena para aproximar de manera precisa las vacancias asociadas a cada tipo de defecto. Está escrito en lenguaje Python 3.12.3 y depende de las librerías Numpy(1.23.0), Pandas(2.2.3), Scikit-learn(1.6.1) y XGBoost(2.1.4). Implementa los modificadores del software de código abierto OVITO (6.8.2.1) para extraer informacion relevante de aquellas regiones de la red cristalina que pueden presentar vacancias.


VacancyAnalysis se compone de tres fases de procesamiento ejecutadas de manera secuencial: primero,se generan defectos en una red perfecta de forma intencionada para entrenar los modelos predictivos; segundo, se aíslan los defectos y se extraen los datos relevantes; y tercero, se realiza el procesamiento final que culmina en las predicciones. Su arquitectura modular y adaptable permite una facil escaleabilidad.



VacancyAnalysis se presenta como una alternativa innovadora al algoritmo de Wigner-Seitz. Mientras que el método tradicional, basado en la generación de celdas de Voronoi, puede sobrecontar defectos en muestras altamente deformadas, VacancyAnalysis enfoca sus predicciones en el análisis detallado de las características clave de los nanoporos presentes en la red cristalina. Este enfoque permite una contabilización más precisa y generalizada de las vacancias, adaptándose eficazmente a una amplia variedad de muestras y condiciones experimentales.





VacancyAnalysis es un software avanzado de postprocesamiento de datos derivados de simulaciones atomísticas, diseñado para facilitar el trabajo del investigador. Al integrar la potencia de los modificadores de OVITO para la extracción de información crítica de la muestra atomística con técnicas de machine learning para interpretar estos datos, el software automatiza y agiliza el análisis de defectos. Esta combinación de metodologías tradicionales y modernas no solo optimiza el flujo de trabajo, sino que también incrementa la precisión en la identificación y cuantificación de vacancias, ofreciendo una herramienta robusta y versátil para el estudio y la optimización de materiales.
