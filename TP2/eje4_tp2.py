# ..................................................................................
# Algoritmo PSO que resuelve un sistema de ecuaciones
# ..................................................................................

import numpy as np
import matplotlib.pyplot as plt
from pyswarm import pso

# función objetivo
# Minimizamos el error cuadrático de las 2 ecuaciones en el sistema
def funcion_objetivo(x):
    x1, x2 = x[0], x[1]
    eq1 = (3*x1 + 2*x2 - 9)**2
    eq2 = (x1 - 5*x2 - 4)**2
    return eq1 + eq2

# Parámetros del PSO
np.random.seed(82)
c1 = 1.49  # Coeficiente cognitivo
c2 = 1.49  # Coeficiente social
w = 0.3  # Peso de inercia
lb = [-100, -100]  # limite inf
ub = [100, 100]  # limite sup

num_particulas = 30  # numero de particulas
cantidad_iteraciones = 50  # numero maximo de iteraciones

# Llamada a la función pso
solucion_optima, valor_optimo = pso(funcion_objetivo, lb, ub, swarmsize=num_particulas, 
                                    maxiter=cantidad_iteraciones,omega=w, phip=c1, phig=c2, debug=False)

# Resultados
print("\nSolución óptima (x, y):", solucion_optima)
print("Valor óptimo:", valor_optimo)

x1, x2 = solucion_optima[0], solucion_optima[1]
Sol_eq1 = round(3*x1 + 2*x2 - 9,4)
Sol_eq2 = round(x1 - 5*x2 - 4,4)
print("Solución a ecuación 1: ",Sol_eq1)
print("Solución a ecuación 2: ",Sol_eq2)