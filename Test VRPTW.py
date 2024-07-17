"""
TWVRP: Problema de enrutamiento de vehículos con ventanas temporales.
"""
from gurobipy import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import cycle
import random

# Nodos
n = 11 # N�mero de nodos
clientes = [i for i in range(n) if i!=0] # Un conjunto de clientes que no incluye el valor "0"
nodos = [0] + clientes # Un conjunto de nodos, que son el n�mero de clientes m�s "0"
arcos = [(i, j) for i in nodos for j in nodos if i!=j] # Todos los arcos entre dos puntos

# Demanda
np.random.seed(0)
q = {n:np.random.randint(10, 15) for n in clientes}
q[0] = 0 # El nodo 0 no tiene demanda

# Ventanas temporales
e = {0:0, 1:10, 2:10, 3:10, 4:20, 5:20, 6:20, 7:40, 8:40, 9:40, 10:40} # M�nimo tiempo de llegada
l = {0:200, 1:100, 2:100, 3:100, 4:150, 5:150, 6:150, 7:180, 8:180, 9:180, 10:180} # M�ximo tiempo de llegada
# El formato es -> numero_nodo:tiempo_entrega
# Por ejemplo, 2:10 indica que el m�nimo tiempo de llegada al nodo 2 es de 10 minutos.
# Como 0 es el depot, tiene como hora m�xima de llegada el mayor valor de todos, ya que es el �ltimo nodo cuando acaba la jornada laboral

s = {n:np.random.randint(3,5) for n in clientes} # Tiempo de servicio en el nodo i
# Valor aleatorio entre 3 y 5 minutos para la entrega de los productos en cada nodo
s[0] = 0 # El tiempo de servicio en el nodo "0" es 0

# Veh�culos
vehiculos = [1, 2, 3, 4]

#Q = 50
Q = {1:50, 2:50, 3:25, 4:25} # Cada veh�culo(1, 2, 3, 4) y su capacidad (25 o 50)

# Coordenadas
X = np.random.rand(len(nodos))*100
Y = np.random.rand(len(nodos))*100

# Definicion de distancias y tiempos
distancia = {(i, j): np.hypot(X[i] - X[j], Y[i] - Y[j]) for i in nodos for j in nodos if i!=j}
tiempo = {(i, j): np.hypot(X[i] - X[j], Y[i] - Y[j]) for i in nodos for j in nodos if i!=j}
# Por simplicidad voy a asumir que la distancia y el tiempo son iguales.

plt.figure(figsize=(12, 5))
plt.scatter(X, Y, color = 'blue') # Puntos de entrega en azul

# Centro de distribuci�n
plt.scatter(X[0], Y[0], color = 'red', marker = 'D') # Depot en rojo, marcado como un diamante
plt.annotate("CD", (X[0] - 1, Y[0] - 5.5), fontsize = 10)
# Cada {%d} corresponde con los valores (0, e[0], l[0]), EN ORDEN
# Cuando escribo t_{%d}$, "_" indica que el valor ser� sub�ndice de t
for i in clientes:
    plt.annotate(f'Cliente {i+1}', (X[i] + 1, Y[i]), fontsize = 10)
# Como se ha explicado antes: $q_{i}=%q[i] | $t_{i}$=(e[i],l[i])

plt.xlabel("Distancia X")
plt.ylabel("Distancia Y")
# plt.title("Nodos Problema de Ruteo de Veh�culos con Ventanas Temporales")

plt.show()

# Creaci�n de los arcos
arco_var = [(i, j, k) for i in nodos for j in nodos for k in vehiculos if i!=j]
arco_tiempos = [(i, k) for i in nodos for k in vehiculos]

# Modelo
model = Model('VRPTW')

# Variables de decisi�n
x = model.addVars(arco_var, vtype=GRB.BINARY, name = 'x')
t = model.addVars(arco_tiempos, vtype=GRB.CONTINUOUS, name = 't')

# Funci�n objetivo
model.setObjective(quicksum(distancia[i, j] * x[i, j, k] for i, j, k in arco_var), GRB.MINIMIZE)

# Restricciones
# 1. Llegadas y salidas del centro de distribuci�n
model.addConstrs(quicksum(x[0, j ,k] for j in clientes) <= 1 for k in vehiculos) 
model.addConstrs(quicksum(x[i, 0 ,k] for i in clientes) <= 1 for k in vehiculos) 

# 2. Un veh�culo por nodo
model.addConstrs(quicksum(x[i, j, k] for j in nodos for k in vehiculos if i!=j) == 1 for i in clientes)

# 3. Conservaci�n de flujo
model.addConstrs(quicksum(x[i, j, k] for j in nodos if i!=j) - quicksum(x[j, i, k] for j in nodos if i!=j) == 0 for i in nodos for k in vehiculos)

# 4. Capacidad del veh�culo
model.addConstrs(quicksum(q[i]*quicksum(x[i, j, k] for j in nodos if i!=j) for i in clientes) <= Q[k] for k in vehiculos)

# 5. Ventana de tiempo
model.addConstrs((x[i, j, k] == 1) >> (t[i, k] + s[i] + tiempo[i, j] == t[j, k]) for i in clientes for j in clientes for k in vehiculos if i!=j)

model.addConstrs(t[i, k] >= e[i] for i, k in arco_tiempos)
model.addConstrs(t[i, k] <= l[i] for i, k in arco_tiempos)


model.optimize()

# Imprimir la soluci�n
print("Funci�n objetivo:", str(round(model.ObjVal, 2)))
for v in model.getVars():
    if v.x > 0.9:
        print(str(v.VarName)+"="+str(v.x))
        
# Graficar la soluci�n
rutas = []
truck = []
K = vehiculos
N = nodos
for k in vehiculos:
    for i in nodos:
        if i!=0 and x[0, i, k].x > 0.9:
            aux = [0, i]
            while i!=0:
                j = i
                for h in nodos:
                    if j!=h and x[j, h, k].x > 0.9:
                        aux.append(h)
                        i = h
            rutas.append(aux)
            truck.append(k)
print(rutas)
print(truck)

# Acumulacion de tiempos
tiempo_acum = list()
for n in range(len(rutas)):
    for k in range(len(rutas[n])-1):
        if k == 0:
            aux = [0]
        else:
            i = rutas[n][k]
            j = rutas[n][k + 1]
            t = tiempo[i, j] + s[i] + aux[-1] 
            aux.append(t)
    tiempo_acum.append(aux)

# Gr�fica soluci�n
plt.figure(figsize=(12, 5))
plt.scatter(X, Y, color = 'blue')


# Centro de distribuci�n
plt.scatter(X[0], Y[0], color = 'red', marker = 'D') 
plt.annotate("CD", (X[0] - 1, Y[0] - 5.5), fontsize = 10)

# Representaci�n de las rutas
Color = ['blue', 'green', 'red', 'yellow', 'purple', 'orange', 'black', 'brown']


for r in range(len(rutas)):
    for n in range(len(rutas[r])-1): # Debido a que el �ltimo punto ser� j y no puede ser n + 1
        i = rutas[r][n]
        j = rutas[r][n + 1]
        plt.plot([X[i], X[j]], [Y[i], Y[j]], color = Color[r] ,alpha = 0.4)
    
for r in range(len(tiempo_acum)):
    for n in range(len(tiempo_acum[r])):
        i = rutas[r][n]
        plt.annotate('$q_{%d} = %d$ | $t_{%d} = %d$' %(i, q[i], i, tiempo_acum[r][n]), (X[i]+1, Y[i]), fontsize = 10)
    
patch = [mpatches.Patch(color = Color[n], label = "veh�culo "+str(truck[n])+ " | cap = "+str(Q[truck[n]])) for n in range(len(truck))]

# plt.legend(handles=patch, loc='best')
plt.xlabel("Distancia X")
plt.ylabel("Distancia Y")
# plt.title("Problema de Ruteo de Veh�culos con Ventanas Temporales")

plt.show()



