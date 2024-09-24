import itertools

def gerar_arquiteturas():
    nodos = list(range(21, 43, 6))
    
    arquiteturas = []
    
    for n in nodos:
        arquiteturas.append([n])
    
    for n1, n2 in itertools.product(nodos, repeat=2):
        arquiteturas.append([n1, n2])
    
    return arquiteturas



