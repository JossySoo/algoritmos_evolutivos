# ..................................................................................
# algoritmo PSO que minimiza la funcion unimodal f(x) = sin(x) + sin(x^2)
# ..................................................................................

import numpy as np
import matplotlib.pyplot as plt

# funcion objetivo 
def funcion_objetivo(x):
    return 2*np.sin(x)-(x**2)/2

# funcion para encontrar PSO con numero de particulas como parametro de entrada
def pso(n_particulas):
    num_particulas = n_particulas  # numero de particulas
    dim = 1  # dimensiones
    cantidad_iteraciones = 80  # maximo numero de iteraciones
    c1 = 1  # componente cognitivo
    c2 = 2  # componente social
    w = 0.7  # factor de inercia
    limite_inf = 0  # limite inferior de busqueda
    limite_sup = 4  # limite superior de busqueda

    # inicializacion
    particulas = np.random.uniform(limite_inf, limite_sup, (num_particulas, dim))  #  posiciones iniciales de las particulas

    velocidades = np.zeros((num_particulas, dim))  # inicializacion de la matriz de velocidades en cero

    # inicializacion de pbest y gbest
    pbest = particulas.copy()  # mejores posiciones personales iniciales

    fitness_pbest = np.empty(num_particulas)  # mejores fitness personales iniciales

 
    for i in range(num_particulas):
    
        fitness_pbest[i] = funcion_objetivo(particulas[i])


    gbest = pbest[np.argmin(fitness_pbest)]  # mejor posicion global inicial
    fitness_gbest = np.min(fitness_pbest)  # fitness global inicial

    gbest_array = [] #mejor gbest
    iter = []  #iteraciones
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

            fitness = funcion_objetivo(particulas[i])  # Evaluacion de la funcion objetivo para la nueva posicion

            # actualizacion el mejor personal
            if fitness > fitness_pbest[i]:
            # if fitness < fitness_pbest[i]:    #cambio    
           
                fitness_pbest[i] = fitness  # actualizacion del mejor fitness personal
                pbest[i] = particulas[i].copy()  # actualizacion de la mejor posicion personal

                # actualizacion del mejor global
                if fitness > fitness_gbest:
                #if fitness < fitness_gbest:    
                    fitness_gbest = fitness  # actualizacion del mejor fitness global
                    gbest = particulas[i].copy()  # actualizacion de la mejor posicion global
        gbest_array.append(gbest)
        iter.append(iteracion)

    # resultado
    solucion_optima = gbest  # mejor posicion global final
    valor_optimo = fitness_gbest  # mejor fitness global final

    return solucion_optima, valor_optimo, gbest_array

num_particulas=2
np.random.seed(75)

solucion_opt, valor_opt, gbest_arr=pso(num_particulas)
print(f"Solucion optima (x): {solucion_opt} --- Valor objetivo(imagen): {valor_opt} --usando {num_particulas} partículas")
x = np.arange(0,4,0.1)
y = 2*np.sin(x)-(x**2)/2
plt.figure()

plt.plot(x, y)
plt.legend(['Funcion'])
plt.scatter(solucion_opt,valor_opt, color='green', label=f'Solucion optima:{solucion_opt} ---Valor objetivo:{valor_opt}')

plt.title(f'PSO usando {num_particulas} partículas')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
# Gráfica gbest por iteracion
plt.figure()
iteraciones1=np.arange(0, len(gbest_arr))
plt.plot(iteraciones1, gbest_arr)
plt.title(f'gbest en funcion de iteración usando {num_particulas} partículas')
plt.xlabel('Iteracion')
plt.ylabel('gbest')
plt.legend(['gbest'])
plt.show()


#### Gráfico de gbest por iteracion con diferentes cantidades de partículas

import matplotlib.pyplot as plt
import numpy as np

# Asumimos que la función pso(num_particulas) está definida y retorna (solucion_opt, valor_opt, gbest_arr)

# Cantidades de partículas que quieres evaluar
num_particulas_list = [2, 4, 10, 100, 200, 400]
gbest_results = []

# Ejecutar PSO para cada cantidad de partículas y guardar los resultados
for num_particulas in num_particulas_list:
    solucion_opt, valor_opt, gbest_arr = pso(num_particulas)
    gbest_results.append(gbest_arr)

# Graficar los resultados
plt.figure()

# Graficar cada línea con un color diferente
for i, gbest_arr in enumerate(gbest_results):
    iteraciones = np.arange(0, len(gbest_arr))
    plt.plot(iteraciones, gbest_arr, label=f'{num_particulas_list[i]} partículas')

plt.title('gbest en función de iteración para diferentes cantidades de partículas')
plt.xlabel('Iteración')
plt.ylabel('gbest')
plt.legend()  # Mostrar la leyenda para identificar cada línea
plt.show()