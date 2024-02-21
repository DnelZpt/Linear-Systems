"""
Módulo para simular un circuito lineal RLC usando su representación
en Función de Transferencia y la clase LinearSystem, que para este caso está adaptada
a solo usarse con la ft del sistema, permitiendo ver su respuesta ante un escalón unitario.

Daniel Zapata Yarce - Cód. 1004965048
Marzo 2023
IE09E-Tópicos Especiales II: Python para Ingenieros
Universidad Tecnológica de Pereira
"""

import numpy as np
import matplotlib.pyplot as plt


class SistemaLineal:
    """
    Esta clase permite representar un sistema lineal a partir de su Función de
    Transferencia (ft).

    Además, permite ver la respuesta del sistema ante la función escalón unitario.
    """

    def __init__(self, *args):
        """
        Inicialización del sistema lineal
        :param args: (num, den)
            num: lista con todos los elementos del numerador (si un coeficiente es cero, debe ser puesto)
            den: lista con los elementos del denominador, se asume que el coeficiente de mayor orden es siempre 1,
                por lo que este no se escribe en la lista
        """
        self.params = dict()  # Diccionario con los parámetros del sistema
        # Se organizan los parámetros del sistema
        self.params.update({'num': np.array(args[0])})
        self.params.update({'den': np.array(args[1])})
        self.orden = len(args[0])  # Se define el orden del sistema como la cantidad de datos en el numerador
        # Es más fácil manejar el sistema con su representación en espacios de estado
        self.ee()

    def ee(self):
        """
        Representa el sistema, dada su ft, en espacios de estados
        :return: Nada (procesamiento interno)
        """
        # Se Crea la matriz A con la diagonal superior de unos y  se actualiza 'params'
        self.params.update({'A': np.diag(np.ones(self.orden - 1), k=1)})
        # Todos los elementos de la columna 0 de A son actualizados con el negativo de 'den'
        self.params['A'][:, 0] = - self.params['den']
        # B se crea con los elementos de 'num' en forma vector
        self.params.update({'B': self.params['num'].reshape((self.orden, 1))})
        # C es creada como un vector de ceros con su primer valor = 1
        self.params.update({'C': np.zeros(self.orden)})
        self.params['C'][0] = 1

    def forward_euler(self, entrada, dt):
        """
        Calcula la respuesta del sistema ante una entrada dada usando el método
        de diferencias hacia adelante de Euler. Se asumen condiciones iniciales iguales a cero
        :param entrada: señal de entrada
        :param dt: delta de tiempo
        :return: respuesta del sistema en el dominio indicado por la entrada
        """
        # Se asignan las matrices A, B y C para mayor limpieza de escritura
        ma = self.params['A']
        mb = self.params['B']
        mc = self.params['C']
        vec_x = np.zeros((self.orden, 1))  # Condiciones iniciales del sistema
        sal_y = np.zeros_like(entrada)  # Vector inicializado de respuesta
        # Método de euler hacia adelante
        for i in range(len(entrada)):
            sal_y[i] = mc @ vec_x
            vec_x = vec_x + (ma @ vec_x + mb * entrada[i]) * dt

        return sal_y

    def escalon(self, am=1, dt=0.1, t_final=14.0):
        """
        Grafica la respuesta del sistema ante una entrada de tipo escalón
        :param am: amplitud del escalón (por defecto es 1)
        :param dt: delta de tiempo
        :param t_final: máximo tiempo de simulación
        :return: Nada (Solo muestra la gráfica de respuesta ante un escalón)
        """
        tiempo = np.arange(0, t_final, dt)  # Vector de tiempo de simulación
        entrada = am * np.ones_like(tiempo)  # Función escalón unitario
        salida = self.forward_euler(entrada, dt)  # Cálculo de la respuesta del sistema ante la entrada escalón
        # Creación de la gráfica de respuesta del sistema
        plt.plot(tiempo, salida)
        plt.title('Respuesta al escalón')
        plt.xlabel('Tiempo[s]')
        plt.ylabel('Amplitud')
        plt.grid()
        plt.show()


if __name__ == '__main__':
    """
    Considerando que la Función de Transferencia obtenida, para el circuito RLC, fue: 
        
                    (1/RC)*s
        H(s) = -------------------
                s² + (1/RC)*s + 1/LC
                
    Con los valores de elementos de circuito dados: R=10, C=0.1 y L=0.1:
                
                     s
        H(s) = --------------
                s² + s + 100
    """
    resist = 10  # Resistencia
    induct = 0.1  # Inductancia
    capac = 0.1  # Capacitancia
    delta_t = 0.001  # Delta de tiempo
    # Declaración de sistema RLC
    rlc = SistemaLineal([1 / (resist * capac), 0], [1 / (resist * capac), 1 / (induct * capac)])
    # Visualización de la respuesta del circuito ante una entrada escalón unitario
    rlc.escalon(dt=delta_t)
