################################# Ejercicio 1: ##################################
#
# Crear en Python un vector columna A de 20 individuos binarios aleatorios de tipo string.
# Crear un segundo vector columna B de 20 números aleatorios comprendidos en el intervalo (0, 1). 
# Mutar un alelo aleatorio a aquellos genes pertenecientes a los cromosomas de A que tengan en su i-ésima fila un correspondiente de B inferior a 0.09. 
# Almacenar los cromosomas mutados en un vector columna C y mostrarlos por consola.
###################################################################

import numpy as np

NUMERO_DE_GENES = 6
TAMANO_POBLACION = 20
UMBRAL_MUTACION = 0.09


a = np.array(list(np.random.choice(['0', '1'], NUMERO_DE_GENES) for _ in np.arange(TAMANO_POBLACION)))  
a1 = a.copy()
b = np.random.uniform(0.0, 1.0, TAMANO_POBLACION)
c = {}    #inicializacion de vector de salida

for i, (indice_a, indice_b) in enumerate(zip(a1, b)):
    indice_a_mutar= None
    if indice_b < UMBRAL_MUTACION:
        indice_a_mutar = np.random.randint(0, NUMERO_DE_GENES)       
        if (indice_a[indice_a_mutar] == '1'):
            indice_a[indice_a_mutar] = '0'
        elif (indice_a[indice_a_mutar] == '0'):
            indice_a[indice_a_mutar] = '1'
        c[i] = indice_a
print('Individuos A:\n', a)
print('\nVector aleatorio B:', b)
print('\nCromosomas mutados vector C:')

for indice_c, indice_mutado in c.items():
    print(f'índice {indice_c}: =====>Cromosoma mutado: {indice_mutado}')