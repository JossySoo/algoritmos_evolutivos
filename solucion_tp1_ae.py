import random
import numpy as np
import matplotlib.pyplot as plt

###################################################################
###################################################################
######################## Ejercicio 2 ##############################
###################################################################
###################################################################
# Algoritmo Gneñetico que encuentra el maximo de la funcion x^2
# Seleccion por ruleta
# Pc = 0.92
# Pm = 0.1
###################################################################

# Parámetros
TAMANIO_POBLACION = 4
LONGITUD_CROMOSOMA = 10
TASA_MUTACION = 0.1
TASA_CRUCE = 0.92
GENERACIONES = 10

###################################################################
# Aptitud (y = x^2)
###################################################################
def aptitud(cromosoma):
    x = int(cromosoma, 2)
    return x ** 2

###################################################################
# Inicializar la población
###################################################################
def inicializar_poblacion(tamanio_poblacion, longitud_cromosoma):
    poblacion = []
    for z in range(tamanio_poblacion):
        cromosoma = ""
        for t in range(longitud_cromosoma):
            cromosoma = cromosoma+str(random.randint(0, 1))
        poblacion.append(cromosoma)
    return poblacion

###################################################################
# Seleccion por ruleta
###################################################################
def seleccion_ruleta(poblacion, aptitud_total):
    seleccion = random.uniform(0, aptitud_total)
    aptitud_actual = 0
    for individuo in poblacion:
        aptitud_actual = aptitud_actual+aptitud(individuo)
        if aptitud_actual > seleccion:
            return individuo
        
###################################################################
# Cruce monopunto con probabilidad de cruza pc = 0.92
###################################################################
def cruce_mono_punto(progenitor1, progenitor2, tasa_cruce):
    if random.random() < tasa_cruce:
        punto_cruce = random.randint(1, len(progenitor1) - 1)
        descendiente1 = progenitor1[:punto_cruce] + progenitor2[punto_cruce:]
        descendiente2 = progenitor2[:punto_cruce] + progenitor1[punto_cruce:]
    else:
        descendiente1, descendiente2 = progenitor1, progenitor2
    return descendiente1, descendiente2

###################################################################
# mutacion
###################################################################
def mutacion(cromosoma, tasa_mutacion):
    cromosoma_mutado = ""
    for bit in cromosoma:
        if random.random() < tasa_mutacion:
            cromosoma_mutado = cromosoma_mutado+str(int(not int(bit)))
        else:
            cromosoma_mutado = cromosoma_mutado+bit
    return cromosoma_mutado

###################################################################
# aplicacion de operadores geneticos
###################################################################
def algoritmo_genetico(tamaño_poblacion, longitud_cromosoma, tasa_mutacion, tasa_cruce, generaciones):
    poblacion = inicializar_poblacion(tamaño_poblacion, longitud_cromosoma)

    for generacion in range(generaciones):
        print("Generación:", generacion + 1)

        # Calcular aptitud total para luego
        aptitud_total = 0
        for cromosoma in poblacion:
            aptitud_total = aptitud_total+aptitud(cromosoma)

        print("Aptitud total:", aptitud_total)

        # ..................................................................
        # seleccion
        # de progenitores con el metodo ruleta
        # se crea una lista vacia de progenitores primero
        progenitores = []
        for _ in range(tamaño_poblacion):
            progenitores.append(seleccion_ruleta(poblacion, aptitud_total))

        # ..................................................................
        # Cruce
        descendientes = []
        for i in range(0, tamaño_poblacion, 2):
            descendiente1, descendiente2 = cruce_mono_punto(progenitores[i], progenitores[i + 1], tasa_cruce)
            descendientes.extend([descendiente1, descendiente2])

        # ..................................................................
        # mutacion
        descendientes_mutados = []
        for descendiente in descendientes:
            descendientes_mutados.append(mutacion(descendiente, tasa_mutacion))

        # Aqui se aplica elitismo
        # se reemplazar los peores cromosomas con los mejores progenitores
        poblacion.sort(key=aptitud)
        descendientes_mutados.sort(key=aptitud, reverse=True)
        for i in range(len(descendientes_mutados)):
            if aptitud(descendientes_mutados[i]) > aptitud(poblacion[i]):
                poblacion[i] = descendientes_mutados[i]

        # mostrar el mejor individuo de la generacion
        mejor_individuo = max(poblacion, key=aptitud)
        print("Mejor individuo:", int(mejor_individuo, 2), "Aptitud:", aptitud(mejor_individuo))
        print("_________________________________________________________________________________")

    return max(poblacion, key=aptitud)

###################################################################
# algoritmo genetico ejecucion principal
###################################################################
print("_________________________________________________________________________________")
print("_________________________________________________________________________________")
print()
random.seed(153)
mejor_solucion = algoritmo_genetico(TAMANIO_POBLACION, LONGITUD_CROMOSOMA, TASA_MUTACION, TASA_CRUCE, GENERACIONES)
print("Mejor solución:", int(mejor_solucion, 2), "Aptitud:", aptitud(mejor_solucion))


###################################################################
###################################################################
######################## Ejercicio 4 ##############################
###################################################################
###################################################################

# Parámetros
TAMANIO_POBLACION = 10
LONGITUD_CROMOSOMA = 32  # 15 bits para x y 15 bits para y
BITS_POR_VARIABLE=int(LONGITUD_CROMOSOMA/2)
TASA_MUTACION = 0.1
TASA_CRUCE = 0.92
GENERACIONES = 15

# Rango de valores para x y y
X_MIN, X_MAX = -10, 10
Y_MIN, Y_MAX = 0, 20

###################################################################
# Aptitud (y = x^2)
###################################################################

def aptitud(cromosoma):
    x_bin, y_bin = cromosoma[:BITS_POR_VARIABLE], cromosoma[BITS_POR_VARIABLE:]
    x = int(x_bin, 2)
    y = int(y_bin, 2)

    # Escalar x e y al rango correspondiente
    x = X_MIN + (X_MAX - X_MIN) * x / (2**BITS_POR_VARIABLE - 1)
    y = Y_MIN + (Y_MAX - Y_MIN) * y / (2**BITS_POR_VARIABLE - 1)

    # Función de concentración
    return 7.7 + 0.15*x + 0.22*y - 0.05*x**2 - 0.016*y**2 - 0.007*x*y

###################################################################
# Inicializar la población
###################################################################

def inicializar_poblacion(tamanio_poblacion, longitud_cromosoma):
    poblacion = []
    for z in range(tamanio_poblacion):
        cromosoma = ""
        for t in range(longitud_cromosoma):
            cromosoma = cromosoma + str(random.randint(0, 1))
        poblacion.append(cromosoma)
    return poblacion

###################################################################
# Seleccion por ruleta
###################################################################
def seleccion_ruleta(poblacion, aptitud_total):
    seleccion = random.uniform(0, aptitud_total)
    aptitud_actual = 0
    for individuo in poblacion:
        aptitud_actual = aptitud_actual + aptitud(individuo)
        if aptitud_actual > seleccion:
            return individuo

###################################################################
# Cruce monopunto con probabilidad de cruza pc = 0.92
###################################################################

def cruce_mono_punto(progenitor1, progenitor2, tasa_cruce):
    if random.random() < tasa_cruce:
        punto_cruce = random.randint(1, len(progenitor1) - 1)
        descendiente1 = progenitor1[:punto_cruce] + progenitor2[punto_cruce:]
        descendiente2 = progenitor2[:punto_cruce] + progenitor1[punto_cruce:]
    else:
        descendiente1, descendiente2 = progenitor1, progenitor2
    return descendiente1, descendiente2

###################################################################
# mutacion
###################################################################
def mutacion(cromosoma, tasa_mutacion):
    cromosoma_mutado = ""
    for bit in cromosoma:
        if random.random() < tasa_mutacion:
            cromosoma_mutado = cromosoma_mutado + str(int(not int(bit)))
        else:
            cromosoma_mutado = cromosoma_mutado + bit
    return cromosoma_mutado

###################################################################
# aplicacion de operadores geneticos
###################################################################
def algoritmo_genetico(tamanio_poblacion, longitud_cromosoma, tasa_mutacion, tasa_cruce, generaciones):
    poblacion = inicializar_poblacion(tamanio_poblacion, longitud_cromosoma)
    mejores_aptitudes = []

    for generacion in range(generaciones):
        print("Generación:", generacion + 1)

        aptitud_total = sum(aptitud(cromosoma) for cromosoma in poblacion)
        print("Aptitud total:", aptitud_total)

        progenitores = [seleccion_ruleta(poblacion, aptitud_total) for _ in range(tamanio_poblacion)]
        descendientes = []
        for i in range(0, tamanio_poblacion, 2):
            descendiente1, descendiente2 = cruce_mono_punto(progenitores[i], progenitores[i + 1], tasa_cruce)
            descendientes.extend([descendiente1, descendiente2])

        descendientes_mutados = [mutacion(descendiente, tasa_mutacion) for descendiente in descendientes]

        poblacion.sort(key=aptitud)
        descendientes_mutados.sort(key=aptitud, reverse=True)
        for i in range(len(descendientes_mutados)):
            if aptitud(descendientes_mutados[i]) > aptitud(poblacion[i]):
                poblacion[i] = descendientes_mutados[i]

        mejor_individuo = max(poblacion, key=aptitud)
        mejores_aptitudes.append(aptitud(mejor_individuo))

        x_bin, y_bin = mejor_individuo[:BITS_POR_VARIABLE], mejor_individuo[BITS_POR_VARIABLE:]
        x = X_MIN + (X_MAX - X_MIN) * int(x_bin, 2) / (2**BITS_POR_VARIABLE - 1)
        y = Y_MIN + (Y_MAX - Y_MIN) * int(y_bin, 2) / (2**BITS_POR_VARIABLE - 1)
        print(f"Mejor individuo: x = {x:.3f}, y = {y:.3f}, Aptitud = {aptitud(mejor_individuo):.3f}")
        print("_________________________________________________________________________________")

    mejor_individuo = max(poblacion, key=aptitud)
    x_bin, y_bin = mejor_individuo[:BITS_POR_VARIABLE], mejor_individuo[BITS_POR_VARIABLE:]
    x = X_MIN + (X_MAX - X_MIN) * int(x_bin, 2) / (2**BITS_POR_VARIABLE - 1)
    y = Y_MIN + (Y_MAX - Y_MIN) * int(y_bin, 2) / (2**BITS_POR_VARIABLE - 1)
    return mejor_individuo, x, y, mejores_aptitudes

print("_________________________________________________________________________________")
print("_________________________________________________________________________________")
print()
random.seed(15)

# Ejecución del algoritmo genético
mejor_solucion, x_max, y_max, mejores_aptitudes = algoritmo_genetico(TAMANIO_POBLACION, LONGITUD_CROMOSOMA, TASA_MUTACION, TASA_CRUCE, GENERACIONES)

# Gráfica de c(x, y)
X = np.linspace(X_MIN, X_MAX, 400)
Y = np.linspace(Y_MIN, Y_MAX, 400)
X, Y = np.meshgrid(X, Y)
Z = 7.7 + 0.15*X + 0.22*Y - 0.05*X**2 - 0.016*Y**2 - 0.007*X*Y

plt.figure()
cp = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(cp)
plt.plot(x_max, y_max, 'ro')  # Punto máximo encontrado
plt.title('Concentración de contaminante c(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Máximo encontrado'])
plt.show()

# Gráfica de mejores aptitudes por generación
plt.figure()
plt.plot(mejores_aptitudes)
plt.title('Mejores aptitudes encontradas por generación')
plt.xlabel('Generación')
plt.ylabel('Aptitud')
plt.legend(['Mejor Aptitud'])
plt.show()