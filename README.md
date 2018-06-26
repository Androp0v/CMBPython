# CMBPython
El archivo Parameters.py guarda los parámetros físicos utilizados en todos los demás archivos de la simulación.

El archivo ConformalTime.py calcula e interpola el tiempo conformal en función del factor de escala y gráfica con el resultado.

El archivo Recombination.py calcula la historia de recombinación del universo, muestra las gráficas pertinentes y exporta varios valores del grado de ionización y su correspondiente corrimiento al rojo en dos archivos de texto (XeGrid.txt y XeValues.txt) que lee posteriormente el archivo OpticalDepth.py.

El archivo OpticalDepth.py interpreta el resultado exportado por Recombination.py y calcula y grafica la opacidad y visibilidad en función del corrimiento al rojo.
