"""
Implement a simulator for plot a linear system response to a given input.

Author: Daniel Zapata
April 2023
"""
import numpy as np
import matplotlib.pyplot as plt


class LinearSystem:
    """
    This class allows us to represent a linear system based on its Transfer Function (ft).

     In addition, it allows you to see the response of the system to the unit step function.
     and the system's response to generalized inputs.
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

    def plot_any_input(self, inp, dt=0.01, final_t=14.0, label=''):
        """
        Plots the input system response for any input type
        :param inp: system input (numpy array)
        :param dt: deltha time
        :param final_t: Max plot time
        :param label: Label title for identify every plot
        :return: None
        """
        time = np.arange(0, final_t, dt)  # Vector de time de simulación
        output = self.forward_euler(inp, dt)  # Cálculo de la respuesta del sistema ante la entrada escalón

        # Create the plot with the system input response
        plt.plot(time, output)
        if label:
            plt.title('Input Response Circuit ' + label)
        else:
            plt.title('Input Response')

        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.show()


if __name__ == '__main__':
    """
    Suppose a RC circuit: 
    The RC circuit in SS is:  (Note: The circuit has only one Energy Storage Element. So, its order is one) 
    
    y(t) = Vc(t) = x(t)
    x'(t) = (-1/RC)*x(t) + (1/RC)*E(t) 

    And its TF:
        Vc(s)           1 / RC
       ------ = H(s) = ---------
        E(s)            s + 1/RC
        
    The RLC circuit in SS {x'(t) = A x(t) + B u(t) and y(t) = Vc(t) = C x(t)}:
     
                 |   0      1 |         |   0  |
            A =  |            |     B = |      |     C = [1   0]
                 | -1/LC  -R/L|         | 1/LC |
    Its TF is: 
                     1 / LC
       H(s) = ---------------------
               s^2 + (R/L) s + 1/LC
    """

    # Now, lets solve the Last question Q3

    # Assume values for circuit elements
    capac = 1.0
    resist = 1.0
    induc = 1.0
    # Declare Frequency values
    freq1 = 1E3  # [Hz]
    freq2 = 5  # [Hz]

    vol_peak = 1.0

    t_final = 14
    dt = 0.01
    t_vect = np.arange(0, t_final, dt)

    input_t = vol_peak * np.sin(2 * np.pi * freq1 * t_vect) + vol_peak * np.sin(2 * np.pi * freq2 * t_vect)

    # Declare TF for every circuit divided by numerator and denominator
    # RC Circuit:
    tf_numRC = [1 / (resist * capac)]  # Transfer function numerator for RC circuit
    tf_denRC = [1 / (resist * capac)]  # TF denominator for RC circuit

    # RLC Circuit:
    tf_numRLC = [0, 1 / (capac * induc)]
    tf_denRLC = [resist / induc, 1 / (induc * capac)]

    # Create Lineal System object:
    rc = LinearSystem(tf_numRC, tf_denRC)
    rlc = LinearSystem(tf_numRLC, tf_denRLC)

    # Plot System Response for every circuit
    rc.plot_any_input(input_t, dt=dt, final_t=t_final, label='RC')
    rlc.plot_any_input(input_t, dt=dt, final_t=t_final, label='RLC')

