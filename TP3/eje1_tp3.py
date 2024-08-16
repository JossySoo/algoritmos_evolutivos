# .................................................................
# Ejemplo de PSO con restricciones
#      maximizar     375x1 + 275x2 +475x3 + 325x4
#
#      sujeto a:
#                     x1        <= 4
#                           2x2 <= 12
#                     3x1 + 2x2 <= 18
# .................................................................
import numpy as np
from matplotlib import pyplot as plt

# función objetivo a maximizar
def f(x):
    return 375* x[0] + 275* x[1] + 475* x[2] + 325* x[3]  # funcion objetivo: 3x1 + 5x2

# primera restriccion
def g1(x):
    return 2.5* x[0] + 1.5* x[1] + 2.75* x[2] + 2* x[3] <= 640

# segunda restriccion
def g2(x):
    return 3.5* x[0] + 3* x[1] + 3* x[2] + 2* x[3] <= 960  # restriccion: 2x2 <= 12

# tercera restriccion
def g3(x):
    return  x[0] >= 0
# cuarta restriccion
def g4(x):
    return  x[1] >= 0
# quinta restriccion
def g5(x):
    return  x[2] >= 0
# sexta restriccion
def g6(x):
    return  x[3] >= 0


# parametros
n_particles = 20  # numero de particulas en el enjambre
n_dimensions = 4  # dimensiones del espacio de busqueda (4 variables)
max_iterations = 50  # numero máximo de iteraciones para la optimizacion
c1 = c2 = 1.4944  # coeficientes de aceleracion
w = 0.6  # Factor de inercia

np.random.seed(153)

# inicialización de particulas
x = np.zeros((n_particles, n_dimensions))  # matriz para las posiciones de las particulas
v = np.zeros((n_particles, n_dimensions))  # matriz para las velocidades de las particulas
pbest = np.zeros((n_particles, n_dimensions))  # matriz para los mejores valores personales
pbest_fit = -np.inf * np.ones(n_particles)  # mector para las mejores aptitudes personales (inicialmente -infinito)
gbest = np.zeros(n_dimensions)  # mejor solución global
gbest_fit = -np.inf  # mejor aptitud global (inicialmente -infinito)

# inicializacion de particulas factibles
for i in range(n_particles):
    while True:  # bucle para asegurar que la particula sea factible
        x[i] = np.random.uniform(0, 10, n_dimensions)  # inicializacion posicion aleatoria en el rango [0, 10]
        if g1(x[i]) and g2(x[i]) and g3(x[i])and g4(x[i])and g5(x[i])and g6(x[i]):  # se comprueba si la posicion cumple las restricciones
            break  # Salir del bucle si es factible
    v[i] = np.random.uniform(-1, 1, n_dimensions)  # inicializar velocidad aleatoria
    pbest[i] = x[i].copy()  # ee establece el mejor valor personal inicial como la posicion actual
    fit = f(x[i])  # calculo la aptitud de la posicion inicial
    if fit > pbest_fit[i]:  # si la aptitud es mejor que la mejor conocida
        pbest_fit[i] = fit  # se actualiza el mejor valor personal

gbest_array_iter=[]

# Optimizacion
for _ in range(max_iterations):  # Repetir hasta el número máximo de iteraciones
    for i in range(n_particles):
        fit = f(x[i])  # Se calcula la aptitud de la posicion actual
        # Se comprueba si la nueva aptitud es mejor y si cumple las restricciones
        if fit > pbest_fit[i] and g1(x[i]) and g2(x[i]) and g3(x[i])and g4(x[i])and g5(x[i])and g6(x[i]):
            pbest_fit[i] = fit  # Se actualiza la mejor aptitud personal
            pbest[i] = x[i].copy()  # Se actualizar la mejor posicion personal
            if fit > gbest_fit:  # Si la nueva aptitud es mejor que la mejor global
                gbest_fit = fit  # Se actualizar la mejor aptitud global
                gbest = x[i].copy()  # Se actualizar la mejor posicion global

        # actualizacion de la velocidad de la particula
        v[i] = w * v[i] + c1 * np.random.rand() * (pbest[i] - x[i]) + c2 * np.random.rand() * (gbest - x[i])
        x[i] += v[i]  # Se actualiza la posicion de la particula

        # se asegura de que la nueva posicion esté dentro de las restricciones
        if not (g1(x[i]) and g2(x[i]) and g3(x[i])and g4(x[i])and g5(x[i])and g6(x[i])):
            # Si la nueva posicion no es válida, revertir a la mejor posicion personal
            x[i] = pbest[i].copy()

    gbest_array_iter.append(gbest)



# Se imprime la mejor solucion encontrada y también su valor optimo
gbest_array_iter=np.array(gbest_array_iter)
print("-----------Solución parte A--------------")
print(f"Mejor solucion: [{gbest[0]:.4f}, {gbest[1]:.4f}, {gbest[2]:.4f}, {gbest[3]:.4f}]")
print(f"Valor óptimo: {gbest_fit}")

print("Tiempo total de fabricación: ",2.5* gbest[0] + 1.5* gbest[1] + 2.75* gbest[2] + 2* gbest[3])
print("Tiempo total de acabados: ",3.5* gbest[0] + 3* gbest[1] + 3* gbest[2] + 2* gbest[3])

# Gráfica gbest por iteracion
iteraciones1=np.arange(0, max_iterations)
plt.plot(iteraciones1, gbest_array_iter[:,0], label='gbest_x1')
plt.plot(iteraciones1, gbest_array_iter[:,1], label='gbest_x2')
plt.plot(iteraciones1, gbest_array_iter[:,2], label='gbest_x3')
plt.plot(iteraciones1, gbest_array_iter[:,3], label='gbest_x4')
plt.legend()
plt.title(f'gbest en funcion de iteración usando {n_particles} partículas')
plt.xlabel('Iteraciones')
plt.ylabel('gbest')
plt.show()

####################################################################################
################### Reduciomos el tiempo de acabado de B en 1 ######################
####################################################################################

# segunda restriccion
def g2(x):
    return 3.5* x[0] + 2* x[1] + 3* x[2] + 2* x[3] <= 960  # restriccion: 2x2 <= 12

np.random.seed(153)

# inicialización de particulas
x = np.zeros((n_particles, n_dimensions))  # matriz para las posiciones de las particulas
v = np.zeros((n_particles, n_dimensions))  # matriz para las velocidades de las particulas
pbest = np.zeros((n_particles, n_dimensions))  # matriz para los mejores valores personales
pbest_fit = -np.inf * np.ones(n_particles)  # mector para las mejores aptitudes personales (inicialmente -infinito)
gbest = np.zeros(n_dimensions)  # mejor solución global
gbest_fit = -np.inf  # mejor aptitud global (inicialmente -infinito)

# inicializacion de particulas factibles
for i in range(n_particles):
    while True:  # bucle para asegurar que la particula sea factible
        x[i] = np.random.uniform(0, 10, n_dimensions)  # inicializacion posicion aleatoria en el rango [0, 10]
        if g1(x[i]) and g2(x[i]) and g3(x[i])and g4(x[i])and g5(x[i])and g6(x[i]):  # se comprueba si la posicion cumple las restricciones
            break  # Salir del bucle si es factible
    v[i] = np.random.uniform(-1, 1, n_dimensions)  # inicializar velocidad aleatoria
    pbest[i] = x[i].copy()  # ee establece el mejor valor personal inicial como la posicion actual
    fit = f(x[i])  # calculo la aptitud de la posicion inicial
    if fit > pbest_fit[i]:  # si la aptitud es mejor que la mejor conocida
        pbest_fit[i] = fit  # se actualiza el mejor valor personal

gbest_array_iter=[]

# Optimizacion
for _ in range(max_iterations):  # Repetir hasta el número máximo de iteraciones
    for i in range(n_particles):
        fit = f(x[i])  # Se calcula la aptitud de la posicion actual
        # Se comprueba si la nueva aptitud es mejor y si cumple las restricciones
        if fit > pbest_fit[i] and g1(x[i]) and g2(x[i]) and g3(x[i])and g4(x[i])and g5(x[i])and g6(x[i]):
            pbest_fit[i] = fit  # Se actualiza la mejor aptitud personal
            pbest[i] = x[i].copy()  # Se actualizar la mejor posicion personal
            if fit > gbest_fit:  # Si la nueva aptitud es mejor que la mejor global
                gbest_fit = fit  # Se actualizar la mejor aptitud global
                gbest = x[i].copy()  # Se actualizar la mejor posicion global

        # actualizacion de la velocidad de la particula
        v[i] = w * v[i] + c1 * np.random.rand() * (pbest[i] - x[i]) + c2 * np.random.rand() * (gbest - x[i])
        x[i] += v[i]  # Se actualiza la posicion de la particula

        # se asegura de que la nueva posicion esté dentro de las restricciones
        if not (g1(x[i]) and g2(x[i]) and g3(x[i])and g4(x[i])and g5(x[i])and g6(x[i])):
            # Si la nueva posicion no es válida, revertir a la mejor posicion personal
            x[i] = pbest[i].copy()

    gbest_array_iter.append(gbest)



# Se imprime la mejor solucion encontrada y también su valor optimo
gbest_array_iter=np.array(gbest_array_iter)
print("-----------Solución parte E--------------")
print(f"Mejor solucion: [{gbest[0]:.4f}, {gbest[1]:.4f}, {gbest[2]:.4f}, {gbest[3]:.4f}]")
print(f"Valor óptimo: {gbest_fit}")

print("Tiempo total de fabricación: ",2.5* gbest[0] + 1.5* gbest[1] + 2.75* gbest[2] + 2* gbest[3])
print("Tiempo total de acabados: ",3.5* gbest[0] + 2* gbest[1] + 3* gbest[2] + 2* gbest[3])

# Gráfica gbest por iteracion
iteraciones1=np.arange(0, max_iterations)
plt.plot(iteraciones1, gbest_array_iter[:,0], label='gbest_x1')
plt.plot(iteraciones1, gbest_array_iter[:,1], label='gbest_x2')
plt.plot(iteraciones1, gbest_array_iter[:,2], label='gbest_x3')
plt.plot(iteraciones1, gbest_array_iter[:,3], label='gbest_x4')
plt.legend()
plt.title(f'gbest en funcion de iteración usando {n_particles} partículas')
plt.xlabel('Iteraciones')
plt.ylabel('gbest')
plt.show()


####################################################################################
###################### Prueba con mayor número de partículas #######################
####################################################################################

n_particles = 80

np.random.seed(153)

# inicialización de particulas
x = np.zeros((n_particles, n_dimensions))  # matriz para las posiciones de las particulas
v = np.zeros((n_particles, n_dimensions))  # matriz para las velocidades de las particulas
pbest = np.zeros((n_particles, n_dimensions))  # matriz para los mejores valores personales
pbest_fit = -np.inf * np.ones(n_particles)  # mector para las mejores aptitudes personales (inicialmente -infinito)
gbest = np.zeros(n_dimensions)  # mejor solución global
gbest_fit = -np.inf  # mejor aptitud global (inicialmente -infinito)

# inicializacion de particulas factibles
for i in range(n_particles):
    while True:  # bucle para asegurar que la particula sea factible
        x[i] = np.random.uniform(0, 10, n_dimensions)  # inicializacion posicion aleatoria en el rango [0, 10]
        if g1(x[i]) and g2(x[i]) and g3(x[i])and g4(x[i])and g5(x[i])and g6(x[i]):  # se comprueba si la posicion cumple las restricciones
            break  # Salir del bucle si es factible
    v[i] = np.random.uniform(-1, 1, n_dimensions)  # inicializar velocidad aleatoria
    pbest[i] = x[i].copy()  # ee establece el mejor valor personal inicial como la posicion actual
    fit = f(x[i])  # calculo la aptitud de la posicion inicial
    if fit > pbest_fit[i]:  # si la aptitud es mejor que la mejor conocida
        pbest_fit[i] = fit  # se actualiza el mejor valor personal

gbest_array_iter=[]

# Optimizacion
for _ in range(max_iterations):  # Repetir hasta el número máximo de iteraciones
    for i in range(n_particles):
        fit = f(x[i])  # Se calcula la aptitud de la posicion actual
        # Se comprueba si la nueva aptitud es mejor y si cumple las restricciones
        if fit > pbest_fit[i] and g1(x[i]) and g2(x[i]) and g3(x[i])and g4(x[i])and g5(x[i])and g6(x[i]):
            pbest_fit[i] = fit  # Se actualiza la mejor aptitud personal
            pbest[i] = x[i].copy()  # Se actualizar la mejor posicion personal
            if fit > gbest_fit:  # Si la nueva aptitud es mejor que la mejor global
                gbest_fit = fit  # Se actualizar la mejor aptitud global
                gbest = x[i].copy()  # Se actualizar la mejor posicion global

        # actualizacion de la velocidad de la particula
        v[i] = w * v[i] + c1 * np.random.rand() * (pbest[i] - x[i]) + c2 * np.random.rand() * (gbest - x[i])
        x[i] += v[i]  # Se actualiza la posicion de la particula

        # se asegura de que la nueva posicion esté dentro de las restricciones
        if not (g1(x[i]) and g2(x[i]) and g3(x[i])and g4(x[i])and g5(x[i])and g6(x[i])):
            # Si la nueva posicion no es válida, revertir a la mejor posicion personal
            x[i] = pbest[i].copy()

    gbest_array_iter.append(gbest)



# Se imprime la mejor solucion encontrada y también su valor optimo
gbest_array_iter=np.array(gbest_array_iter)
print("-----------Solución parte F--------------")
print(f"Mejor solucion: [{gbest[0]:.4f}, {gbest[1]:.4f}, {gbest[2]:.4f}, {gbest[3]:.4f}]")
print(f"Valor óptimo: {gbest_fit}")

print("Tiempo total de fabricación: ",2.5* gbest[0] + 1.5* gbest[1] + 2.75* gbest[2] + 2* gbest[3])
print("Tiempo total de acabados: ",3.5* gbest[0] + 2* gbest[1] + 3* gbest[2] + 2* gbest[3])

# Gráfica gbest por iteracion
iteraciones1=np.arange(0, max_iterations)
plt.plot(iteraciones1, gbest_array_iter[:,0], label='gbest_x1')
plt.plot(iteraciones1, gbest_array_iter[:,1], label='gbest_x2')
plt.plot(iteraciones1, gbest_array_iter[:,2], label='gbest_x3')
plt.plot(iteraciones1, gbest_array_iter[:,3], label='gbest_x4')
plt.legend()
plt.title(f'gbest en funcion de iteración usando {n_particles} partículas')
plt.xlabel('Iteraciones')
plt.ylabel('gbest')
plt.show()