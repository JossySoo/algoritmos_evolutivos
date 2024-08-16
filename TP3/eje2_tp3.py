# .................................................................
# Ejemplo de PSO con restricciones
#      maximizar     500x1 + 400x2
#
#      sujeto a:
#                     300x1+400x2 <= 127000
#                      2x1+ 3x2   <= 4270
#                       x1        >=0
#                       x2        >=0
# .................................................................
import numpy as np
from matplotlib import pyplot as plt

# función objetivo a maximizar
def f(x):
    return 500 * x[0] + 400 * x[1]  # funcion objetivo: 3x1 + 5x2
# primera restriccion
def g1(x):
    return 300 * x[0] + 400 * x[1] <= 127000  # restriccion: x1 <= 4
# segunda restriccion
def g2(x, k):   # constante k para poder parametrizar la cantidad de impresoras 2 
    return 20 * x[0] + k * x[1]  <= 4270  # restriccion: 2x2 <= 12
# tercera restriccion
def g3(x):
    return  x[0] >= 0  # restriccion: 3x1 + 2x2 <= 18
# cuarta restriccion
def g4(x):
    return  x[1] >= 0  # restriccion: 3x1 + 2x2 <= 18

#funcion parametrizable de optimizacion a
def algo(k, n_particles, n_dimensions, max_iterations, c1, c2, w):
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
            if g1(x[i]) and g2(x[i], k) and g3(x[i]) and g4(x[i]):  # se comprueba si la posicion cumple las restricciones
                break  # Salir del bucle si es factible
        v[i] = np.random.uniform(-1, 1, n_dimensions)  # inicializar velocidad aleatoria
        pbest[i] = x[i].copy()  # ee establece el mejor valor personal inicial como la posicion actual
        fit = f(x[i])  # calculo la aptitud de la posicion inicial
        if fit > pbest_fit[i]:  # si la aptitud es mejor que la mejor conocida
            pbest_fit[i] = fit  # se actualiza el mejor valor personal
    gbest_array_x=[]
    gbest_array_y=[]
# Optimizacion
    for _ in range(max_iterations):  # Repetir hasta el número máximo de iteraciones
        for i in range(n_particles):
            fit = f(x[i])  # Se calcula la aptitud de la posicion actual
            # Se comprueba si la nueva aptitud es mejor y si cumple las restricciones
            if fit > pbest_fit[i] and g1(x[i]) and g2(x[i], k) and g3(x[i]) and g4(x[i]):  
                pbest_fit[i] = fit  # Se actualiza la mejor aptitud personal
                pbest[i] = x[i].copy()  # Se actualizar la mejor posicion personal
                if fit > gbest_fit:  # Si la nueva aptitud es mejor que la mejor global
                    gbest_fit = fit  # Se actualizar la mejor aptitud global
                    gbest = x[i].copy()  # Se actualizar la mejor posicion global

        # actualizacion de la velocidad de la particula
            v[i] = w * v[i] + c1 * np.random.rand() * (pbest[i] - x[i]) + c2 * np.random.rand() * (gbest - x[i])
            x[i] += v[i]  # Se actualiza la posicion de la particula

        # se asegura de que la nueva posicion esté dentro de las restricciones
            if not (g1(x[i]) and g2(x[i], k) and g3(x[i]) and g3(x[i])):        
                # Si la nueva posicion no es válida, revertir a la mejor posicion personal
                x[i] = pbest[i].copy()

        gbest_array_x.append(gbest[0])
        gbest_array_y.append(gbest[1])
    return gbest, gbest_fit, gbest_array_x , gbest_array_y 

# Gráfica gbest por iteracion
def graficar_gbest_iteracion(gbest_array_x,gbest_array_y,num_particulas):
    iteraciones1=np.arange(0, len(gbest_array_x))
    plt.plot(iteraciones1, gbest_array_x, label='gbest_x1')
    plt.plot(iteraciones1, gbest_array_y, label='gbest_x2')
    plt.legend()
    plt.title(f'gbest en funcion de iteración usando {num_particulas} partículas')
    plt.xlabel('Iteraciones')
    plt.ylabel('gbest')
    plt.show()

#llamada a la funcion del algoritmo de optimizacion como parametro la cantidad de impresoras 2
# parametros
n_particulas = 10  # numero de particulas en el enjambre
n_dimensiones = 2  # dimensiones del espacio de busqueda (x1 y x2)
max_iterationes = 80  # numero máximo de iteraciones para la optimizacion
c1 = c2 = 2  # coeficientes de aceleracion
w = 0.5  # Factor de inercia
impresora_2=[10, 11]   #ver Cantidad horas requeridas para fabricar la Impresora 2 para cumplir punto E de la consigna
                       # 10 es el valor original y 11 el requerido por consigna 


for k in impresora_2:
    #llamada de la funcion parametrizable para encontrar el maximo, aca se puede mo
    gbest, gbest_fit, gbest_array_x , gbest_array_y =algo(k, n_particulas, n_dimensiones, max_iterationes, c1 , c2, w)

   # llamada a la funcion de graficacion
    graficar_gbest_iteracion(gbest_array_x,gbest_array_y, n_particulas)
    print(f"Mejor solucion: [{gbest[0]:.4f}, {gbest[1]:.4f}] Valor optimo: {gbest_fit} Cantidad de horas Impresora 2==> {k} Cantidad de Particulas==> {n_particulas}\n")
  
