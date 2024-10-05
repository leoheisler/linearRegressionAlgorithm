import matplotlib.pyplot as plt # type: ignore
import numpy as np

def buildGraph(x,y,t0,t1):
    # Criação do grafico com a linha encontrada pela regressão
    maxValue = max(x)
    interval = np.linspace(0,maxValue,100)
    funct = t1*interval + t0
    plt.plot(interval, funct, color='red', label=f'y = {t1}x + {t0}')

    # colocar os pontos originais passados
    plt.scatter(x, y, color='blue', marker='o')
    plt.xlabel('Valores de X')
    plt.ylabel('Valores de Y')
    plt.title('Grafo de Pontos')
    plt.grid(True)
    plt.show()


#função linear de regressão
def applyCoefficients(x,t0,t1):
    return x*t1 + t0

#função de custo de erro quadrático
def costFunction(x,y,t0,t1):
    
    numEntries = len(x)
    quadradicErr = 0

    for i in range (0,numEntries):
        instance = applyCoefficients(x[i],t0,t1)
        quadradicErr += pow(instance - y[i],2)/numEntries

    return quadradicErr

#função que calcula o vetor gradiente        
def gradientVector(a,x,y,t0,t1):

    numEntries = len(x)
    gradientT0 = 0
    gradientT1 = 0

    for i in range(0,numEntries):
        instance = applyCoefficients(x[i],t0,t1)
        gradientT0 += 2*(instance - y[i])/numEntries
        gradientT1 += 2*x[i]*(instance - y[i])/numEntries

    newT0 = t0 - a*gradientT0
    newT1 = t1 - a*gradientT1

    return (newT0,newT1)