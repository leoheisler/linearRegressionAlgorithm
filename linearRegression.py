import auxFunctions as af

#valores a serem analisados
x = (1,2,8,10)
y = (5,8,26,32)

#tamanho do dataset
m = len(x)

#chutes iniciais
t0 = 100000
t1 = 800000

#valor do indice de aprendizado
a = 0.5

#num max iteracoes
itr = 1000

for i in range(0, itr):
    print(i)
    print("t0: ", t0, "t1: ", t1)
    err = af.costFunction(x,y,t0,t1)
    print("error: ", err)
    print("fator apr: ", a)
    print("-/-/-/-/-")

    #atualiza coeficientes de acordo com o  vetor gradiente
    (t0Temp,t1Temp) = af.gradientVector(a,x,y,t0,t1)

    #verifica o erro dos novos coeficientes encontrados
    errTemp = af.costFunction(x,y,t0Temp,t1Temp)
    if errTemp > err:
        #atualiza fator de aprendizado
        a = a/2
    else:
        (t0,t1) = (t0Temp,t1Temp)
    
    
    

#construcao do grafico
af.buildGraph(x,y,t0,t1)


