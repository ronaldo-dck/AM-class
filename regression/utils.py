import itertools


def gerar_arquiteturas():
    nodos = [[7, 7],
             [7, 14],
             [7, 7, 7],
             [7, 14, 7],
             [7, 14, 14],]
                                                                                                                            
    arquiteturas = nodos

    # for n in nodos:
    #     arquiteturas.append([n])

    # for n1, n2 in itertools.product(nodos, repeat=2):
    #     arquiteturas.append([n1, n2])

    return arquiteturas
