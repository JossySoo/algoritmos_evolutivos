###################################################################
# Algoritmo Genetico que encuentra el maximo de la funcion (g = (2*c) / (4 + 0.8*c + c**2 + 0.2*c**3))
# Seleccion por Torneo
# Pc = 0.85
# Pm = 0.07
###################################################################
import random
import numpy as np
import matplotlib.pyplot as plt

# Parámetros
# Tomando el intervalo [0, 10] con precisión de 2 decimales (1000-0)*100 = 1000 esta va a ser la poblacion
# los cromosomas van a tener 10 bits: 2^9 <  1000 <= 2^10
TAMANIO_POBLACION = 1000         
LONGITUD_CROMOSOMA = 10
TASA_MUTACION = 0.07
TASA_CRUCE = 0.85
GENERACIONES = 25

########### funcion genotipo #######################################
#funcion para decodificar genotipo acorde a las cifras significativas
####################################################################
def genotipo(cromosoma):
    c=round(10 * int(''.join(cromosoma), 2) / (2**10 - 1), 2)
    return c

###################################################################
# Aptitud (g = (2*c) / (4 + 0.8*c + c**2 + 0.2*c**3))
###################################################################
def aptitud(cromosoma):
    c =genotipo(cromosoma)
    return (2*c) / (4 + 0.8*c + c**2 + 0.2*c**3)

# Definir la función g(c) para graficar
def g(c):
    return (2.0*c) / (4.0 + 0.8*c + c**2 + 0.2*c**3)

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
# Seleccion por torneo
###################################################################
def seleccion_torneo(poblacion):  # tournament selection
    tournament = random.sample(range(len(poblacion)), k=3)
    tournament_fitnesses=[]
    for i in range(0,3):
        tournament_fitnesses = aptitud(poblacion[tournament[i]])
    winner_index = tournament[np.argmax(tournament_fitnesses)]
    return poblacion[winner_index]
#################################################################

###################################################################
# Cruce monopunto con probabilidad de cruza pc = 0.85
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
    mejores_aptitudes = []

    for generacion in range(generaciones):
        #print("Generación:", generacion + 1)

        # Calcular aptitud total para luego
        aptitud_total = 0
        for cromosoma in poblacion:
            aptitud_total = aptitud_total+aptitud(cromosoma)

        #print("Aptitud total:", aptitud_total)

        # ..................................................................
        # seleccion
        # de progenitores con el metodo torneo
        # se crea una lista vacia de progenitores primero
        progenitores = []
        for _ in range(tamaño_poblacion):
            progenitores.append(seleccion_torneo(poblacion))

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
        mejores_aptitudes.append(aptitud(mejor_individuo))
        #print("Mejor individuo:", int(mejor_individuo, 2), "Aptitud:", aptitud(mejor_individuo))

        print(f"Generación {generacion + 1}: Mejor individuo = {genotipo(mejor_individuo)}, Aptitud = {aptitud(mejor_individuo)}")
        print("_________________________________________________________________________________")

    return max(poblacion, key=aptitud), mejores_aptitudes

###################################################################
# algoritmo genetico ejecucion principal
###################################################################
print("_________________________________________________________________________________")
print("_________________________________________________________________________________")
print()
mejor_solucion, mejores_aptitudes = algoritmo_genetico(TAMANIO_POBLACION, LONGITUD_CROMOSOMA, TASA_MUTACION, TASA_CRUCE, GENERACIONES)
print("Mejor solución:", genotipo(mejor_solucion), "Aptitud:", aptitud(mejor_solucion))

c = np.arange(-1,20,0.1)
g = (2*c) / (4 + 0.8*c + c**2 + 0.2*c**3)

plt.figure()

plt.plot(c, g, linestyle='dashed')
plt.legend(['Funcion'])
plt.scatter(genotipo(mejor_solucion),aptitud(mejor_solucion), color='red', label='Máximo encontrado')
plt.title('Tasa de crecimiento de  una levadura que produce cierto antibiotico')
plt.xlabel('c')
plt.ylabel('g')
plt.legend()
plt.show()

# Gráfica de mejores aptitudes por generación
plt.figure()
plt.plot(range(1, GENERACIONES + 1), mejores_aptitudes)
plt.title('Mejores aptitudes encontradas por generación')
plt.xlabel('Generación')
plt.ylabel('Aptitud')
plt.legend(['Mejor Aptitud'])
plt.show()

