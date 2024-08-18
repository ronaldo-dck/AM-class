import itertools

def gerar_arquiteturas():
    # Número de nodos por camada oculta
    nodos = list(range(21, 43, 6))
    
    # Lista para armazenar as arquiteturas
    arquiteturas = []
    
    # Arquiteturas com 1 camada oculta
    for n in nodos:
        arquiteturas.append([n])
    
    # Arquiteturas com 2 camadas ocultas
    for n1, n2 in itertools.product(nodos, repeat=2):
        arquiteturas.append([n1, n2])
    
    # # Arquiteturas com 3 camadas ocultas
    # for n1, n2, n3 in itertools.product(nodos, repeat=3):
    #     arquiteturas.append([n1, n2, n3])
    
    return arquiteturas



