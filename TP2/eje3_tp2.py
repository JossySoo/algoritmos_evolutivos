# ..................................................................................
# algoritmo PSO que minimiza la funcion unimodal f(x, y) = (x-a)^2 + (y-b)^2
# ..................................................................................

import numpy as np
from matplotlib import pyplot as plt
from pyswarm import pso

#funcion para validar las constantes de entrada
def validar_constante(nombre,constante_inferior, constante_superior):
    while True:
        valor = float(int(input(f'Constante {nombre} ---> Valor entre -50 y 50: ')))
        if constante_inferior <= valor <= constante_superior:    
            break
        else:
            continue
    return valor

# funcion objetivo 
def funcion_objetivo(x, y, a ,b):
    return (x - a)**2 + (y + b)**2

#funcion para graficar PSO para poder reutilizar entre los items C y G
def graficar_pso(funcion_objetivo, gbest, limite_inf, limite_sup, a, b, etiqueta):
    coord_x = np.linspace(limite_inf, limite_sup, 100)
    coord_y = np.linspace(limite_inf, limite_sup, 100)
    coord_x, coord_y = np.meshgrid(coord_x, coord_y)
    coord_z = funcion_objetivo(coord_x, coord_y, a, b)
    minimo_x=  0
    minimo_y = 0
    minimo_z = funcion_objetivo(gbest[0], gbest[1], a, b)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(coord_x, coord_y, coord_z, cmap='turbo',alpha=0.5)
    ax.scatter(minimo_x, minimo_y, minimo_z, color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('funcion f(x, y)')
    plt.title(f'Funcion Objetivo f(x, y) = (x - {a})^2 + (y + {b})^2 {etiqueta}')
    plt.show()

# Gráfica gbest por iteracion
def graficar_gbest_iteracion(gbest_array_x,gbest_array_y,num_particulas):
    iteraciones1=np.arange(0, len(gbest_array_x))
    plt.plot(iteraciones1, gbest_array_x, label='gbest_x')
    plt.plot(iteraciones1, gbest_array_y, label='gbest_y')
    plt.legend()
    plt.title(f'gbest en funcion de iteración usando {num_particulas} partículas')
    plt.xlabel('Iteracion')
    plt.ylabel('gbest')
    plt.show()
 

# Funcion para obtener pso para poder variar los valores del algoritmo
def obtener_pso(num_particulas,dim,cantidad_iteraciones,c1,c2,w,limite_inf,limite_sup):
    particulas = np.random.uniform(limite_inf, limite_sup, (num_particulas, dim))  # posiciones iniciales de las particulas

    velocidades = np.zeros((num_particulas, dim))  # inicializacion de la matriz de velocidades en cero

    # inicializacion de pbest y gbest
    pbest = particulas.copy()  # mejores posiciones personales iniciales

    fitness_pbest = np.empty(num_particulas)  # mejores fitness personales iniciales
    for i in range(num_particulas):
        fitness_pbest[i] = funcion_objetivo(particulas[i][0], particulas[i][1], a , b)

    gbest = pbest[np.argmin(fitness_pbest)]  # mejor posicion global inicial
    fitness_gbest = np.min(fitness_pbest)  # fitness global inicial
    gbest_array_x=[]
    gbest_array_y=[]
    # busqueda
    for iteracion in range(cantidad_iteraciones):
        for i in range(num_particulas):  # iteracion sobre cada partícula
            r1, r2 = np.random.rand(), np.random.rand()  # generacion dos numeros aleatorios

        # actualizacion de la velocidad de la particula en cada dimension
            for d in range(dim):
                velocidades[i][d] = (w * velocidades[i][d] + c1 * r1 * (pbest[i][d] - particulas[i][d]) + c2 * r2 * (gbest[d] - particulas[i][d]))

            for d in range(dim):
                particulas[i][d] = particulas[i][d] + velocidades[i][d]  # cctualizacion de la posicion de la particula en cada dimension

            # mantenimiento de las partículas dentro de los limites
                particulas[i][d] = np.clip(particulas[i][d], limite_inf, limite_sup)

            fitness = funcion_objetivo(particulas[i][0], particulas[i][1], a , b)  # Evaluacion de la funcion objetivo para la nueva posicion       

        # actualizacion el mejor personal
            if fitness < fitness_pbest[i]:
                fitness_pbest[i] = fitness  # actualizacion del mejor fitness personal
                pbest[i] = particulas[i].copy()  # actualizacion de la mejor posicion personal

            # actualizacion del mejor global
                if fitness < fitness_gbest:
                    fitness_gbest = fitness  # actualizacion del mejor fitness global
                    gbest = particulas[i].copy()  # actualizacion de la mejor posicion global

    # imprimir el mejor global en cada iteracion
        gbest_array_x.append(gbest[0])
        gbest_array_y.append(gbest[1])

# resultado
    solucion_optima = gbest  # mejor posicion global final
    valor_optimo = fitness_gbest  # mejor fitness global final

    print(f'Usando w:{w} Solucion optima (x, y): {solucion_optima} Valor optimo(imagen): {valor_optimo}')
    return solucion_optima, valor_optimo, gbest_array_x,gbest_array_y
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# parametros
num_particulas = 20  # numero de particulas
dim = 2  # dimensiones
cantidad_iteraciones = 10  # maximo numero de iteraciones
c1 = 2.0  # componente cognitivo
c2 = 2.0  # componente social
w = 0.7  # factor de inercia
limite_inf = -100  # limite inferior de busqueda
limite_sup = 100  # limite superior de busqueda

# capturando constantes de entrada
a= validar_constante('a',-50, 50)
b= validar_constante('b',-50, 50)

# Ejecucion con W= 0.7
solucion_optima, valor_optimo, gbest_array_x,gbest_array_y=obtener_pso(num_particulas,dim,cantidad_iteraciones,c1,c2,w,limite_inf,limite_sup)
graficar_pso(funcion_objetivo, solucion_optima, limite_inf, limite_sup, a, b,'sin pyswarm')

graficar_gbest_iteracion(gbest_array_x,gbest_array_y,num_particulas)

# Ejecucion con coeficiente de inercia w = 0.0
w = 0  # factor de inercia
solucion_optima, valor_optimo, gbest_array_x,gbest_array_y=obtener_pso(num_particulas,dim,cantidad_iteraciones,c1,c2,w,limite_inf,limite_sup)
graficar_pso(funcion_objetivo, solucion_optima, limite_inf, limite_sup, a, b,'sin pyswarm')
graficar_gbest_iteracion(gbest_array_x,gbest_array_y,num_particulas)


# Reescribir el algoritmo PSO para que cumpla nuevamente con los ítems A  hasta F pero usando la biblioteca pyswarm (from pyswarm import pso)
#

w = 0.7
funcion_objetivo_pyswarm = lambda x: (x[0] - a)**2 + (x[1] + b)**2
solucion_optima, valor_optimo = pso(funcion_objetivo_pyswarm,[limite_inf, limite_inf],[limite_sup, limite_sup],swarmsize=num_particulas,omega=w,phip=c1,phig=c2,maxiter=cantidad_iteraciones)
print(f'Usando w:{w} Solucion optima (x, y): {solucion_optima} Valor optimo(imagen): {valor_optimo}')
graficar_pso(funcion_objetivo, solucion_optima, limite_inf, limite_sup, a, b,'utilizando pyswarm')


w = 0
funcion_objetivo_pyswarm = lambda x: (x[0] - a)**2 + (x[1] + b)**2
solucion_optima, valor_optimo = pso(funcion_objetivo_pyswarm,[limite_inf, limite_inf],[limite_sup, limite_sup],swarmsize=num_particulas,omega=w,phip=c1,phig=c2,maxiter=cantidad_iteraciones)
print(f'Usando w:{w} Solucion optima (x, y): {solucion_optima} Valor optimo(imagen): {valor_optimo}')
graficar_pso(funcion_objetivo, solucion_optima, limite_inf, limite_sup, a, b,'utilizando pyswarm')
