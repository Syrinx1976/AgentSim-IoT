# -*- coding: cp1252 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from scipy.spatial import distance
import random
import time
import numpy as np
from scipy.stats import zipf
from bisect import bisect
import networkx as nx
import executar_selecao_dos_dispositivos
import executar_selecao_knapsack
import sys

# Parâmetros da coleta
Energia_para_sensing = 0.00000022
Energia_para_transmissao = 0.000000200
Energia_para_recepcao = 0.000000150
Tamanho_max_coleta_knapsack_oportunista = 3 * 20

def DrawGraph(G,color):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels = True, edge_color = color)  #with_labels=true is to show the node number in the output graph
    edge_labels = nx.get_edge_attributes(G,'length')
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels,  font_size = 11) #prints weight on all the edges
    return pos

def CreateGraph():
    G = nx.Graph()
    f = open('input_christofides.txt')
    n = int(f.readline())
    wtMatrix = []
    for i in range(n):
        list1 = map(int, (f.readline()).split())
        wtMatrix.append(list1)

    #Adds egdes along with their weights to the graph
    for i in range(n) :
        for j in range(n)[i:] :
            if wtMatrix[i][j] > 0 :
                    G.add_edge(i, j, length = wtMatrix[i][j])
    return G

def func_Sequencia_nos(graph, Nos_fontes_a_visitar):

    if len(Nos_fontes_a_visitar) == 1:

        Caminho_definido_parte1 = shortest_path(graph, 'GW', Nos_fontes_a_visitar[0])[1]

        Caminho_definido_parte2 = shortest_path(graph, Nos_fontes_a_visitar[0], 'GW')[1]

        Caminho_definido_0 = Caminho_definido_parte1 + Caminho_definido_parte2

        for i in Caminho_definido_0:
            if Caminho_definido_0.index(i) <= len(Caminho_definido_0)-2:
                if i == Caminho_definido_0[Caminho_definido_0.index(i)+1]:
                    Caminho_definido_0.pop(Caminho_definido_0.index(i))

        return Caminho_definido_0, Nos_fontes_a_visitar


    else:
        L = Nos_fontes_a_visitar

        arquivo = []
        for i in L:
            line = []
            for x in L:
                if i != x:
                    line.append(shortest_path(graph, i, x)[0])
                else:
                    line.append(0)
            arquivo.append(line)


        f = open('input_christofides.txt','w')
        f.write(str(len(L))+'\n')
        for i in arquivo:
            count = 0
            while count <= len(i)-1:
                for x in i:
                    f.write(str(int(100 * x))+' ')
                    count = count+1
            f.write('\n')

        f.close()

        G = CreateGraph()
        #plt.figure(1)
        pos = DrawGraph(G,'black')
        opGraph = christofedes(G, pos)
        #pprint(vars(opGraph)['_succ'])
        Sequencia1 = (vars(opGraph)['_succ'])
        Sequencia = []
        Sequencia2 = []
        Chaves = Sequencia1.keys()
        next = []
        for i in Chaves:
            next.append(Sequencia1[i].keys())

        for i in Chaves:
            Sequencia.append([L[i],L[next[i][0]]])

        for i in Sequencia:
            for x in i:
                if x not in Sequencia2:
                    Sequencia2.append(x)

        Sequencia = Sequencia2
        print '\nSequencia otimizada com Christofides:', Sequencia
        Caminho_definido = []

        Caminho_definido.append(shortest_path(graph, 'GW', Sequencia[0])[1])
        for node in Sequencia:
            if Sequencia.index(node)+1 <= len(Sequencia)-1:
                no_origem = node
                no_destino = Sequencia[Sequencia.index(node)+1]
                Caminho_definido.append(shortest_path(graph, no_origem, no_destino)[1])

        Caminho_definido.append(shortest_path(graph, Sequencia[-1], 'GW')[1])

        Caminho_definido_1 = []
        for node in Caminho_definido:
            for n in node:
                Caminho_definido_1.append(n)

        for node in Caminho_definido_1:
            if Caminho_definido_1.index(node)+1 <= len(Caminho_definido_1)-1:
                if node == Caminho_definido_1[Caminho_definido_1.index(node)+1]:
                    Caminho_definido_1.pop(Caminho_definido_1.index(node)+1)

        print '\nCaminho definido:', Caminho_definido_1
        return Caminho_definido_1, Sequencia

# A utility function that return the smallest unprocessed edge
def getMin(G, mstFlag):
    min = sys.maxsize  # assigning largest numeric value to min
    for i in [(u, v, edata['length']) for u, v, edata in G.edges( data = True) if 'length' in edata ]:
        if mstFlag[i] == False and i[2] < min:
            min = i[2]
            min_edge = i
    return min_edge

# A utility function to find root or origin of the node i in MST
def findRoot(parent, i):
    if parent[i] == i:
        return i
    return findRoot(parent, parent[i])


# A function that does union of set x and y based on the order
def union(parent, order, x, y):
    xRoot = findRoot(parent, x)
    yRoot = findRoot(parent, y)
     # Attach smaller order tree under root of high order tree
    if order[xRoot] < order[yRoot]:
        parent[xRoot] = yRoot
    elif order[xRoot] > order[yRoot]:
        parent[yRoot] = xRoot
    # If orders are same, then make any one as root and increment its order by one
    else :
        parent[yRoot] = xRoot
        order[xRoot] += 1

#function that performs kruskals algorithm on the graph G
def genMinimumSpanningTree(G):
    MST = nx.Graph()
    eLen = len(G.edges()) # eLen denotes the number of edges in G
    vLen = len(G.nodes()) # vLen denotes the number of vertices in G
    mst = [] # mst contains the MST edges
    mstFlag = {} # mstFlag[i] will hold true if the edge i has been processed for MST
    for i in [ (u, v, edata['length']) for u, v, edata in G.edges(data = True) if 'length' in edata ]:
        mstFlag[i] = False

    parent = [None] * vLen # parent[i] will hold the vertex connected to i, in the MST
    order = [None] * vLen    # order[i] will hold the order of appearance of the node in the MST
    for v in range(vLen):
        parent[v] = v
        order[v] = 0
    while len(mst) < vLen - 1 :
        curr_edge = getMin(G, mstFlag) # pick the smallest egde from the set of edges
        mstFlag[curr_edge] = True # update the flag for the current edge
        y = findRoot(parent, curr_edge[1])
        x = findRoot(parent, curr_edge[0])
        # adds the edge to MST, if including it doesn't form a cycle
        if x != y:
            mst.append(curr_edge)
            union(parent, order, x, y)
        # Else discard the edge
    for X in mst:
        if (X[0], X[1]) in G.edges():
                MST.add_edge(X[0], X[1], length = G[X[0]][X[1]]['length'])
    return MST


#utility function that adds minimum weight matching edges to MST
def minimumWeightedMatching(MST, G, odd_vert):
    while odd_vert:
        v = odd_vert.pop()
        length = float("inf")
        u = 1
        closest = 0
        for u in odd_vert:
            if G[v][u]['length'] < length :
                length = G[v][u]['length']
                closest = u
        MST.add_edge(v, closest, length = length)
        odd_vert.remove(closest)

def christofedes(G ,pos):
    opGraph=nx.DiGraph()
    #optimal_dist = 0
    MST = genMinimumSpanningTree(G) # generates minimum spanning tree of graph G, using Prim's algo
    odd_vert = [] #list containing vertices with odd degree
    for i in MST.nodes():
        if MST.degree(i)%2 != 0:
            odd_vert.append(i) #if the degree of the vertex is odd, then append it to odd_vert list
    minimumWeightedMatching(MST, G, odd_vert) #adds minimum weight matching edges to MST
    # now MST has the Eulerian circuit
    start = list(MST.nodes())[0]
    visited = [False] * len(MST.nodes())
    # finds the hamiltonian circuit
    curr = start
    visited[curr] = True
    for nd in MST.neighbors(curr):
            if visited[nd] == False or nd == start:
                next = nd
                break
    while next != start:
        visited[next]=True
        opGraph.add_edge(curr,next,length = G[curr][next]['length'])
        nx.draw_networkx_edges(G, pos, arrows = True, edgelist = [(curr, next)], width = 2.5, alpha = 0.6, edge_color = 'r')
        # optimal_dist = optimal_dist + G[curr][next]['length']
        # finding the shortest Eulerian path from MST
        curr = next
        for nd in MST.neighbors(curr):
            if visited[nd] == False:
                next = nd
                break
        if next == curr:
            for nd in G.neighbors(curr):
                if visited[nd] == False:
                    next = nd
                    break
        if next == curr:
            next = start
    opGraph.add_edge(curr,next,length = G[curr][next]['length'])
    nx.draw_networkx_edges(G, pos, edgelist = [(curr, next)], width = 2.5, alpha = 0.6, edge_color = 'r')

    return opGraph

def media(L):
  'Calcula RTTs medios dos arquivos Resultados_medias.dat'
  soma=0
  for i in range(len(L)):
    soma+=L[i]
  media=float(float(soma)/(len(L)))
  return media


def shortest_path(graph, origin, destination):

    return nx.bidirectional_dijkstra(graph, origin, destination, weight='weight')

def weighted_choice(choices):
    values, weights = zip(*choices)
    total = 0
    cum_weights = []
    for w in weights:
        total += w
        cum_weights.append(total)
    x = random.random() * total
    i = bisect(cum_weights, x)
    return values[i]

def zipf_weights(alpha_zipf):

    k = np.linspace(0, 10, 5)

    a = alpha_zipf

    prob_k1 = zipf.pmf(k=1, a=a)

    prob_k2 = zipf.pmf(k=2, a=a)

    prob_k3 = zipf.pmf(k=3, a=a)

    prob_k4 = zipf.pmf(k=4, a=a)

    prob_k5 = zipf.pmf(k=5, a=a)

    return [prob_k1, prob_k2, prob_k3, prob_k4, prob_k5]

def coleta_de_dados(Pontos_cobertos_k1,Pontos_cobertos_k2,Pontos_cobertos_k3,Pontos_cobertos_k4,Pontos_cobertos_k5,pontos_restantes_k1,pontos_restantes_k2,pontos_restantes_k3,pontos_restantes_k4,pontos_restantes_k5,Conjunto_pontos_cobertos_k1,Conjunto_pontos_cobertos_k2,Conjunto_pontos_cobertos_k3,Conjunto_pontos_cobertos_k4,Conjunto_pontos_cobertos_k5,Universo_set_cover_problem,Dict_cobertura_AoI_k1,Dict_cobertura_AoI_k2,Dict_cobertura_AoI_k3,Dict_cobertura_AoI_k4,Dict_cobertura_AoI_k5,Dic_dispositivo_visitado_por_MA,Dic_falhas_por_dispositivos_normalizado,Dic_sensing_ranges,Lista_pontos,plt,dispositivos,Dic_reqs_dispositivos,Dispositivos_selecionados_por_conteudo,Dic_registro_energia_remanescente_dispositivos,Dic_numero_reqs_por_conteudo_por_no,contador_reqs_conteudos,Dic_pop_conteudo,G,Dic_registro_custos_originais,Dic_energia_media_consumida_por_conjunto_de_sensing_nodes_k1,Dic_energia_media_consumida_por_conjunto_de_sensing_nodes_k2,Dic_energia_media_consumida_por_conjunto_de_sensing_nodes_k3,Dic_energia_media_consumida_por_conjunto_de_sensing_nodes_k4,Dic_energia_media_consumida_por_conjunto_de_sensing_nodes_k5,Dic_falhas_por_dispositivos, Lista_conteudos, Dic_tempo_ultima_atualizacao_por_conteudo_por_no,tempo_validade_c,tipo_distribuicao, Dic_valores_coletados_por_conteudo_por_no, List_cache_hits, Lista_media_nos_intermediarios, Media_qtde_nos_fontes):

    #Parametros iniciais:
    graph = G
    Rate_Poisson = 5.0
    tempo_validade_cache = tempo_validade_c
    alpha_zipf = 2.0
    num_requisicoes = 5000
    tempo_max_coleta = 2000
    MA_size = None
    Contador_envio_MA = 0
    Dic_energia_inicial_na_selecao_por_conjunto_de_sensing_nodes_k1 = {}
    Dic_energia_inicial_na_selecao_por_conjunto_de_sensing_nodes_k2 = {}
    Dic_energia_inicial_na_selecao_por_conjunto_de_sensing_nodes_k3 = {}
    Dic_energia_inicial_na_selecao_por_conjunto_de_sensing_nodes_k4 = {}
    Dic_energia_inicial_na_selecao_por_conjunto_de_sensing_nodes_k5 = {}
    Dic_energia_final_na_selecao_por_conjunto_de_sensing_nodes_k1 = {}
    Dic_energia_final_na_selecao_por_conjunto_de_sensing_nodes_k2 = {}
    Dic_energia_final_na_selecao_por_conjunto_de_sensing_nodes_k3 = {}
    Dic_energia_final_na_selecao_por_conjunto_de_sensing_nodes_k4 = {}
    Dic_energia_final_na_selecao_por_conjunto_de_sensing_nodes_k5 = {}
    Dic_selecionados_knapsack_conteudo_por_dispositivo = {}
    Dic_valores_distribuidos_aoi = {}
    Resultados_RMSD_respostas_clientes = []
    Contagem_nos_fontes = []

    if modo_debug == 'on':
        print 'Escolher o tipo de Prioridade: (1) Distribuicao uniforme, (2) Distribuicao Zipf:'

    Tipo_distribuicao = tipo_distribuicao

    if Tipo_distribuicao == 1:
        Dic_prioridades = {'prob_k1':0.2,'prob_k2':0.2,'prob_k3':0.2,'prob_k4':0.2,'prob_k5':0.2}

        if modo_debug == 'on':
            print '\n\n',Dic_prioridades

    if Tipo_distribuicao == 2:

        Dic_prioridades = {'prob_k1':zipf_weights(alpha_zipf)[0],'prob_k2':zipf_weights(alpha_zipf)[1],'prob_k3':zipf_weights(alpha_zipf)[2],'prob_k4':zipf_weights(alpha_zipf)[3],'prob_k5':1 - (zipf_weights(alpha_zipf)[0] + zipf_weights(alpha_zipf)[1] + zipf_weights(alpha_zipf)[2] + zipf_weights(alpha_zipf)[3])}

        if modo_debug == 'on':
            print '\n\n',Dic_prioridades

    Lista_instantes_poisson = []
    for i in range(0, num_requisicoes):
        Lista_instantes_poisson.append(random.expovariate(1/Rate_Poisson))


    t = 0
    Dic_numero_reqs_por_conteudo_por_no_na_rodada = {}
    for op in dispositivos:

        Dic_numero_reqs_por_conteudo_por_no_na_rodada.update({op.nome:{'k1':0,'k2':0,'k3':0,'k4':0,'k5':0}})
        Dic_selecionados_knapsack_conteudo_por_dispositivo.update({op.nome:[]})


    for valor_aoi in dispositivos:
        Dic_valores_distribuidos_aoi.update({valor_aoi.nome:{'valor_k1':valor_aoi.valor_conteudo_1,'valor_k2':valor_aoi.valor_conteudo_2,'valor_k3':valor_aoi.valor_conteudo_3,'valor_k4':valor_aoi.valor_conteudo_4,'valor_k5':valor_aoi.valor_conteudo_5}})

    valores_medidos_k1 = []
    valores_medidos_k2 = []
    valores_medidos_k3 = []
    valores_medidos_k4 = []
    valores_medidos_k5 = []

    for valor_medio in Dic_valores_distribuidos_aoi:
        for valor_medido in Dic_valores_distribuidos_aoi[valor_medio]:
            if valor_medido == 'valor_k1':
                valores_medidos_k1.append(Dic_valores_distribuidos_aoi[valor_medio][valor_medido])
            if valor_medido == 'valor_k2':
                valores_medidos_k2.append(Dic_valores_distribuidos_aoi[valor_medio][valor_medido])
            if valor_medido == 'valor_k3':
                valores_medidos_k3.append(Dic_valores_distribuidos_aoi[valor_medio][valor_medido])
            if valor_medido == 'valor_k4':
                valores_medidos_k4.append(Dic_valores_distribuidos_aoi[valor_medio][valor_medido])
            if valor_medido == 'valor_k5':
                valores_medidos_k5.append(Dic_valores_distribuidos_aoi[valor_medio][valor_medido])

    Media_AoI_k1 = media(valores_medidos_k1)
    Media_AoI_k2 = media(valores_medidos_k2)
    Media_AoI_k3 = media(valores_medidos_k3)
    Media_AoI_k4 = media(valores_medidos_k4)
    Media_AoI_k5 = media(valores_medidos_k5)

    for i in Lista_instantes_poisson:

        if t + i <= tempo_max_coleta:

            Chave_execucao_coleta = 'on'

            t = t + i

            List_cache_hits[1] += 1

            if modo_debug == 'on':
                print '\033[31m'+'\n\n##### Início de nova coleta #####\n\n'+'\033[0;0m'
                print '\nInstante atual:', t

            #determinando qual grandeza foi selecionada para essa requisição
            Lista_probabilidades = [("k1",Dic_prioridades['prob_k1']),("k2",Dic_prioridades['prob_k2']),("k3",Dic_prioridades['prob_k3']),("k4",Dic_prioridades['prob_k4']),("k5",Dic_prioridades['prob_k5'])]

            Grandeza_selecionada = weighted_choice(Lista_probabilidades)

            contador_reqs_conteudos[Grandeza_selecionada] += 1

            Dic_tempo_ultima_atualizacao_por_conteudo_por_no, Nos_fontes_a_visitar, Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k1, Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2, Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k3, Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k4, Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k5, Dict_cobertura_AoI_knapsack = executar_selecao_dos_dispositivos.selecao_dos_dispositivos(Dic_sensing_ranges,Pontos_cobertos_k1,Pontos_cobertos_k2,Pontos_cobertos_k3,Pontos_cobertos_k4,Pontos_cobertos_k5,pontos_restantes_k1,pontos_restantes_k2,pontos_restantes_k3,pontos_restantes_k4,pontos_restantes_k5,Conjunto_pontos_cobertos_k1,Conjunto_pontos_cobertos_k2,Conjunto_pontos_cobertos_k3,Conjunto_pontos_cobertos_k4,Conjunto_pontos_cobertos_k5,Universo_set_cover_problem,Dict_cobertura_AoI_k1,Dict_cobertura_AoI_k2,Dict_cobertura_AoI_k3,Dict_cobertura_AoI_k4,Dict_cobertura_AoI_k5, tempo_validade_cache, Dic_falhas_por_dispositivos_normalizado, plt, Grandeza_selecionada, Dic_falhas_por_dispositivos, dispositivos, Dic_tempo_ultima_atualizacao_por_conteudo_por_no, t, Dic_valores_coletados_por_conteudo_por_no)

            if len(Nos_fontes_a_visitar) == 0:

                if modo_debug == 'on':
                    print '\n\nNenhuma coleta necessária!'
                List_cache_hits[0] += 1
                Chave_execucao_coleta = 'off'

                # ===========================================
                # Resposta precisa ser enviada para o cliente:
                # ===========================================
                Valores_resposta_media = []
                Lista_dispositivos_valores_coletados = []

                for valor_coletado in Dic_valores_coletados_por_conteudo_por_no:
                    if Dic_valores_coletados_por_conteudo_por_no[valor_coletado][Grandeza_selecionada] != None:
                        if (t - Dic_tempo_ultima_atualizacao_por_conteudo_por_no[valor_coletado][Grandeza_selecionada]) <= tempo_validade_cache:
                                Valores_resposta_media.append(Dic_valores_coletados_por_conteudo_por_no[valor_coletado][Grandeza_selecionada])
                                Lista_dispositivos_valores_coletados.append([valor_coletado,Dic_valores_coletados_por_conteudo_por_no[valor_coletado][Grandeza_selecionada]])

                media_val_resposta = media(Valores_resposta_media)

                # ===========================
                # Calculando o erro nesse caso:
                # ===========================

                print 'Lista', Lista_dispositivos_valores_coletados

                diferencas_quadraticas_respostas_reais = []

                for grupamento in Lista_dispositivos_valores_coletados:

                    diferenca = media_val_resposta - Dic_valores_distribuidos_aoi[grupamento[0]]['valor_'+Grandeza_selecionada]
                    diferencas_quadraticas_respostas_reais.append(diferenca**2)


                somatorio_diferencas_quadraticas = 0

                for diferencas in diferencas_quadraticas_respostas_reais:

                    somatorio_diferencas_quadraticas += diferencas

                RMSD_val_resposta_cliente = somatorio_diferencas_quadraticas/len(Lista_dispositivos_valores_coletados)

                RMSD_val_resposta_cliente = float((RMSD_val_resposta_cliente)**(0.5))

                print '\n\nRMSD_val_resposta_cliente:', RMSD_val_resposta_cliente

                Resultados_RMSD_respostas_clientes.append(RMSD_val_resposta_cliente)


            if Chave_execucao_coleta == 'on':

                Caminho_definido, Sequencia_christofides_nos_fontes = func_Sequencia_nos(graph, Nos_fontes_a_visitar)

                # Definindo tamanho inicial do MA
                Lista_de_controle = [Dic_dispositivo_visitado_por_MA[i] for i in Caminho_definido if i != 'GW']

                if None in Lista_de_controle:
                    MA_size = 1024
                if None not in Lista_de_controle:
                    MA_size = 1024

                # iniciando a coleta com Mobile Agent

                #retira as duplicatas do caminho definido

                Caminho_definido_sem_nos_fontes = [i for i in Caminho_definido if i not in Nos_fontes_a_visitar]

                Caminho_definido_sem_nos_fontes_sem_duplicados = []

                for i in Caminho_definido_sem_nos_fontes:
                    if i not in Caminho_definido_sem_nos_fontes_sem_duplicados and i != 'GW':
                        Caminho_definido_sem_nos_fontes_sem_duplicados.append(i)

                if modo_debug == 'on':

                    print '\nTamanho do MA:', MA_size
                    print '\nNós fontes a visitar:', Nos_fontes_a_visitar
                    print '\nsem nós fontes:', Caminho_definido_sem_nos_fontes
                    print '\nsem duplicados:', Caminho_definido_sem_nos_fontes_sem_duplicados

                Media_qtde_nos_fontes.append(len(Nos_fontes_a_visitar))

                # análise dos nós intermediários pelos knapsack

                Selecionados_knapsack, Dic_tempo_ultima_atualizacao_por_conteudo_por_no, Dic_selecionados_knapsack_por_conteudo = executar_selecao_knapsack.selecao_knapsack(Dic_prioridades,Grandeza_selecionada,tempo_validade_cache,t,Lista_conteudos,Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k1, Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2, Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k3, Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k4,Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k5,Universo_set_cover_problem,Dict_cobertura_AoI_knapsack,Nos_fontes_a_visitar,Tamanho_max_coleta_knapsack_oportunista,dispositivos,Caminho_definido_sem_nos_fontes_sem_duplicados,Lista_pontos,Dic_tempo_ultima_atualizacao_por_conteudo_por_no)

                # Atualizando os timers do cache para nós fontes e nós intermediários atualizados:

                for i in Nos_fontes_a_visitar:

                    Dic_tempo_ultima_atualizacao_por_conteudo_por_no[i][Grandeza_selecionada] = t

                if modo_debug == 'on':
                    print '\n\nSelecionados por conteudo (aqui!!):', Dic_selecionados_knapsack_por_conteudo

                Caminho_definido.pop(0)
                Caminho_definido.pop(-1)

                grupo_nos_intermediarios_selecionados = []
                tamanho_payload_oportunista = 0
                tamanho_payload_reservado = 0
                conteudos_oportunista = []
                conteudos_coletado_no_fonte = False
                posicao = 0

                for conteudo in Selecionados_knapsack:
                    for dispositivo in Selecionados_knapsack[conteudo]:
                        if dispositivo not in grupo_nos_intermediarios_selecionados:
                            grupo_nos_intermediarios_selecionados.append(dispositivo)

                if modo_debug == 'on':
                    print '\n\n\nNós intermediários selecionados pelo knapsack por conteúdo:', Selecionados_knapsack
                    print '\nNós intermediários selecionados pelo knapsack:', grupo_nos_intermediarios_selecionados
                    print '\nNós fontes (sequência Christofides):', Sequencia_christofides_nos_fontes


                Indice_do_no_fonte_para_iniciar_coleta, Indice_k1_para_iniciar_coleta, Indice_k2_para_iniciar_coleta, Indice_k3_para_iniciar_coleta, Indice_k4_para_iniciar_coleta, Indice_k5_para_iniciar_coleta, Indices_nos_fontes, Indices_conteudo_k1, Indices_conteudo_k2, Indices_conteudo_k3, Indices_conteudo_k4, Indices_conteudo_k5 = determinacao_dos_indices_de_inicio_de_coleta_todos_casos(Sequencia_christofides_nos_fontes, Selecionados_knapsack, Caminho_definido)

                Metade_itinerario = []
                n_controle_metade_itinerario = 0
                for i in Caminho_definido:
                    if n_controle_metade_itinerario >= len(Caminho_definido)/2:
                        Metade_itinerario.append(i)
                    n_controle_metade_itinerario += 1


                if modo_debug == 'on':
                    print '\n\nSelecionados_knapsack:', Selecionados_knapsack
                    print '\n\nDic_selecionados_knapsack_por_conteudo:', Dic_selecionados_knapsack_por_conteudo


                for i in Selecionados_knapsack:
                    for j in Selecionados_knapsack[i]:
                        if j in Metade_itinerario:
                          Dic_tempo_ultima_atualizacao_por_conteudo_por_no[j][i] = t

                if modo_debug == 'on':
                    print '\n\n\n\n\nIndices:', Indice_do_no_fonte_para_iniciar_coleta, Indice_k1_para_iniciar_coleta, Indice_k2_para_iniciar_coleta, Indice_k3_para_iniciar_coleta, Indice_k4_para_iniciar_coleta, Indice_k5_para_iniciar_coleta

                for i in Selecionados_knapsack:
                    for j in Selecionados_knapsack[i]:
                        if i not in Dic_selecionados_knapsack_conteudo_por_dispositivo[j]:
                            Dic_selecionados_knapsack_conteudo_por_dispositivo[j].append(i)

                if modo_debug == 'on':
                    print '\n\n\nSelecionados knapsack por conteudo:', Selecionados_knapsack,'\n\n'

                Quantidade_sensores_intermediarios = 0

                for contador_intermediarios in Selecionados_knapsack:
                    Quantidade_sensores_intermediarios += len(Selecionados_knapsack[contador_intermediarios])

                Lista_media_nos_intermediarios.append(Quantidade_sensores_intermediarios)

                if modo_debug == 'on':
                    print '\ntamanho do MA antes de começar a coleta:',MA_size

                controle_indice = 0
                prob_falha = 0
                dist_acum = []

                for dispositivo in Caminho_definido:

                        if controle_indice < len(Caminho_definido)-1:
                            for obj in dispositivos:
                                if obj.nome == dispositivo:
                                    j = obj
                                if obj.nome == Caminho_definido[controle_indice+1]:
                                    i = obj

                            prob_falha = G.edges[dispositivo,Caminho_definido[controle_indice+1]]['prob_falha']
                            # energia gasta para receber o MA
                            if controle_indice == 0:
                                distancia_recepcao = round(distance.euclidean((50,50),j.coordenada_dispositivo),2)
                            if controle_indice > 0:
                                for obj1 in dispositivos:
                                   if obj1.nome == Caminho_definido[controle_indice-1]:
                                       q = obj1

                                distancia_recepcao = round(distance.euclidean(q.coordenada_dispositivo,j.coordenada_dispositivo),2)

                            distancia = round(distance.euclidean(j.coordenada_dispositivo, i.coordenada_dispositivo),2)

                        Dic_registro_energia_remanescente_dispositivos[dispositivo] -= Energia_para_recepcao * distancia_recepcao * MA_size

                        if controle_indice == Indice_k1_para_iniciar_coleta:
                            MA_size += 20
                            if modo_debug == 'on':
                                print dispositivo,controle_indice,'O tamanho do MA aumentará em 20 bytes porque este é o primeiro Nó k1 para coleta.'

                        if controle_indice == Indice_k2_para_iniciar_coleta:
                            MA_size += 20
                            if modo_debug == 'on':
                                print dispositivo,controle_indice,'O tamanho do MA aumentará em 20 bytes porque este é o primeiro Nó k2 para coleta.'

                        if controle_indice == Indice_k3_para_iniciar_coleta:
                            MA_size += 20
                            if modo_debug == 'on':
                                print dispositivo,controle_indice,'O tamanho do MA aumentará em 20 bytes porque este é o primeiro Nó k3 para coleta.'

                        if controle_indice == Indice_k4_para_iniciar_coleta:
                            MA_size += 20
                            if modo_debug == 'on':
                                print dispositivo,controle_indice,'O tamanho do MA aumentará em 20 bytes porque este é o primeiro Nó k4 para coleta.'

                        if controle_indice == Indice_k5_para_iniciar_coleta:
                            MA_size += 20
                            if modo_debug == 'on':
                                print dispositivo,controle_indice,'O tamanho do MA aumentará em 20 bytes porque este é o primeiro Nó k5 para coleta.'


                        Dic_registro_energia_remanescente_dispositivos[dispositivo] -= Energia_para_transmissao * distancia * MA_size

                        Dado1 = float(random.randrange(0,100,1))/100

                        while Dado1 < prob_falha:
                            print 'ocorreu falha!!!! Retransmitindo MA. Dispositivo:',dispositivo
                            Dic_registro_energia_remanescente_dispositivos[dispositivo] -= Energia_para_transmissao * distancia * MA_size
                            print "Tamanho do MA", MA_size
                            Dado1 = float(random.randrange(0,100,1))/100

                        controle_indice += 1


                Indices_nos_fontes.sort()
                Indices_conteudo_k1.sort()
                Indices_conteudo_k2.sort()
                Indices_conteudo_k3.sort()
                Indices_conteudo_k4.sort()
                Indices_conteudo_k5.sort()

                if modo_debug == 'on':

                    print 'Caminho definido:', Caminho_definido, len(Caminho_definido)
                    print 'Escolhidos Knapsack:', Selecionados_knapsack, len(Selecionados_knapsack)
                    print 'Indices Nos fontes:', Indices_nos_fontes, len(Indices_nos_fontes), Indice_do_no_fonte_para_iniciar_coleta
                    print 'Indices conteudo k1:', Indices_conteudo_k1, len(Indices_conteudo_k1), Indice_k1_para_iniciar_coleta
                    print 'Indices conteudo k2:', Indices_conteudo_k2, len(Indices_conteudo_k2), Indice_k2_para_iniciar_coleta
                    print 'Indices conteudo k3:', Indices_conteudo_k3, len(Indices_conteudo_k3), Indice_k3_para_iniciar_coleta
                    print 'Indices conteudo k4:', Indices_conteudo_k4, len(Indices_conteudo_k4), Indice_k4_para_iniciar_coleta
                    print 'Indices conteudo k5:', Indices_conteudo_k5, len(Indices_conteudo_k5), Indice_k5_para_iniciar_coleta

                Indices_nos_fontes_em_ordem_sem_duplicados = []
                Indices_k1_em_ordem_sem_duplicados = []
                Indices_k2_em_ordem_sem_duplicados = []
                Indices_k3_em_ordem_sem_duplicados = []
                Indices_k4_em_ordem_sem_duplicados = []
                Indices_k5_em_ordem_sem_duplicados = []

                for indice in Indices_nos_fontes:
                    if indice not in Indices_nos_fontes_em_ordem_sem_duplicados:
                        Indices_nos_fontes_em_ordem_sem_duplicados.append(indice)


                for indice in Indices_conteudo_k1:
                    if indice not in Indices_k1_em_ordem_sem_duplicados:
                        Indices_k1_em_ordem_sem_duplicados.append(indice)

                for indice in Indices_conteudo_k2:
                    if indice not in Indices_k2_em_ordem_sem_duplicados:
                        Indices_k2_em_ordem_sem_duplicados.append(indice)

                for indice in Indices_conteudo_k3:
                    if indice not in Indices_k3_em_ordem_sem_duplicados:
                        Indices_k3_em_ordem_sem_duplicados.append(indice)

                for indice in Indices_conteudo_k4:
                    if indice not in Indices_k4_em_ordem_sem_duplicados:
                        Indices_k4_em_ordem_sem_duplicados.append(indice)

                for indice in Indices_conteudo_k5:
                    if indice not in Indices_k5_em_ordem_sem_duplicados:
                        Indices_k5_em_ordem_sem_duplicados.append(indice)

                if modo_debug == 'on':

                    print '\n\nIndices nos fontes sem duplicados:', Indices_nos_fontes_em_ordem_sem_duplicados, len(Indices_nos_fontes_em_ordem_sem_duplicados),'\n\n'

                    print '\n\nIndices conteudo k1 sem duplicados:', Indices_k1_em_ordem_sem_duplicados, len(Indices_k1_em_ordem_sem_duplicados),'\n\n'

                    print '\n\nIndices conteudo k2 sem duplicados:', Indices_k2_em_ordem_sem_duplicados, len(Indices_k2_em_ordem_sem_duplicados),'\n\n'

                    print '\n\nIndices conteudo k3 sem duplicados:', Indices_k3_em_ordem_sem_duplicados, len(Indices_k3_em_ordem_sem_duplicados),'\n\n'

                    print '\n\nIndices conteudo k4 sem duplicados:', Indices_k4_em_ordem_sem_duplicados, len(Indices_k4_em_ordem_sem_duplicados),'\n\n'

                    print '\n\nIndices conteudo k5 sem duplicados:', Indices_k5_em_ordem_sem_duplicados, len(Indices_k5_em_ordem_sem_duplicados),'\n\n'

                Indices_nos_fontes_em_ordem_sem_duplicados.sort()

                Indices_k1_em_ordem_sem_duplicados.sort()

                Indices_k2_em_ordem_sem_duplicados.sort()

                Indices_k3_em_ordem_sem_duplicados.sort()

                Indices_k4_em_ordem_sem_duplicados.sort()

                Indices_k5_em_ordem_sem_duplicados.sort()


                # Finalmente determinação dos valores:
                Lista_valores_coletados_nos_fontes = []
                Lista_valores_coletados_k1 = []
                Lista_valores_coletados_k2 = []
                Lista_valores_coletados_k3 = []
                Lista_valores_coletados_k4 = []
                Lista_valores_coletados_k5 = []


                for index in Indices_nos_fontes_em_ordem_sem_duplicados:

                    for obj in dispositivos:

                        if obj.nome == Caminho_definido[index]:
                            j = obj

                    if Grandeza_selecionada == 'k1':
                        Lista_valores_coletados_nos_fontes.append([Caminho_definido[index],index,j.valor_conteudo_1])

                    if Grandeza_selecionada == 'k2':
                        Lista_valores_coletados_nos_fontes.append([Caminho_definido[index],index,j.valor_conteudo_2])

                    if Grandeza_selecionada == 'k3':
                        Lista_valores_coletados_nos_fontes.append([Caminho_definido[index],index,j.valor_conteudo_3])

                    if Grandeza_selecionada == 'k4':
                        Lista_valores_coletados_nos_fontes.append([Caminho_definido[index],index,j.valor_conteudo_4])

                    if Grandeza_selecionada == 'k5':
                        Lista_valores_coletados_nos_fontes.append([Caminho_definido[index],index,j.valor_conteudo_5])


                if len(Indices_k1_em_ordem_sem_duplicados) > 0:

                    for index in Indices_k1_em_ordem_sem_duplicados:

                        for obj in dispositivos:

                            if obj.nome == Caminho_definido[index]:
                                j = obj

                        Lista_valores_coletados_k1.append([Caminho_definido[index],index,j.valor_conteudo_1])

                if len(Indices_k2_em_ordem_sem_duplicados) > 0:

                    for index in Indices_k2_em_ordem_sem_duplicados:

                        for obj in dispositivos:

                            if obj.nome == Caminho_definido[index]:
                                j = obj

                        Lista_valores_coletados_k2.append([Caminho_definido[index],index,j.valor_conteudo_2])


                if len(Indices_k3_em_ordem_sem_duplicados) > 0:

                    for index in Indices_k3_em_ordem_sem_duplicados:

                        for obj in dispositivos:

                            if obj.nome == Caminho_definido[index]:
                                j = obj

                        Lista_valores_coletados_k3.append([Caminho_definido[index],index,j.valor_conteudo_3])


                if len(Indices_k4_em_ordem_sem_duplicados) > 0:

                    for index in Indices_k4_em_ordem_sem_duplicados:

                        for obj in dispositivos:

                            if obj.nome == Caminho_definido[index]:
                                j = obj

                        Lista_valores_coletados_k4.append([Caminho_definido[index],index,j.valor_conteudo_4])

                if len(Indices_k5_em_ordem_sem_duplicados) > 0:

                    for index in Indices_k5_em_ordem_sem_duplicados:

                        for obj in dispositivos:

                            if obj.nome == Caminho_definido[index]:
                                j = obj

                        Lista_valores_coletados_k5.append([Caminho_definido[index],index,j.valor_conteudo_5])

                if modo_debug == 'on':

                    print '\n\nValores coletados nos fontes:', Grandeza_selecionada, Lista_valores_coletados_nos_fontes, len(Lista_valores_coletados_nos_fontes)
                    print '\n\nValores conteúdos k1:', Lista_valores_coletados_k1, len(Lista_valores_coletados_k1)
                    print '\n\nValores conteúdos k2:', Lista_valores_coletados_k2, len(Lista_valores_coletados_k2)
                    print '\n\nValores conteúdos k3:', Lista_valores_coletados_k3, len(Lista_valores_coletados_k3)
                    print '\n\nValores conteúdos k4:', Lista_valores_coletados_k4, len(Lista_valores_coletados_k4)
                    print '\n\nValores conteúdos k5:', Lista_valores_coletados_k5, len(Lista_valores_coletados_k5)

                # calculo da média para Nós fontes:
                somente_valores_nos_fontes = [k[2] for k in Lista_valores_coletados_nos_fontes]
                somente_valores_k1 = [k[2] for k in Lista_valores_coletados_k1]
                somente_valores_k2 = [k[2] for k in Lista_valores_coletados_k2]
                somente_valores_k3 = [k[2] for k in Lista_valores_coletados_k3]
                somente_valores_k4 = [k[2] for k in Lista_valores_coletados_k4]
                somente_valores_k5 = [k[2] for k in Lista_valores_coletados_k5]

                if modo_debug == 'on':

                    print '\n\nSomente valores coletados nos fontes:', somente_valores_nos_fontes, len(somente_valores_nos_fontes)
                    print '\n\nSomente valores conteúdos k1:', somente_valores_k1, len(somente_valores_k1)
                    print '\n\nSomente valores conteúdos k2:', somente_valores_k2, len(somente_valores_k2)
                    print '\n\nSomente valores conteúdos k3:', somente_valores_k3, len(somente_valores_k3)
                    print '\n\nSomente valores conteúdos k4:', somente_valores_k4, len(somente_valores_k4)
                    print '\n\nSomente valores conteúdos k5:', somente_valores_k5, len(somente_valores_k5)


                # calculo da media nos fontes:
                moeda = 0
                media_val_nos_fontes = Lista_valores_coletados_nos_fontes[0][2]


                while moeda < len(Lista_valores_coletados_nos_fontes)-1:

                    media_val_nos_fontes = (float(media_val_nos_fontes+Lista_valores_coletados_nos_fontes[moeda+1][2])/2)
                    moeda += 1

                if modo_debug == 'on':
                    print '\n\nMedia nos fontes:', media_val_nos_fontes

                for valores_agregados in Lista_valores_coletados_nos_fontes:
                    Dic_valores_coletados_por_conteudo_por_no[valores_agregados[0]][Grandeza_selecionada] = media_val_nos_fontes

                # calculo da media conteudo k1:
                if len(Lista_valores_coletados_k1) != 0:

                    moeda = 0
                    media_val_k1 = Lista_valores_coletados_k1[0][2]


                    while moeda < len(Lista_valores_coletados_k1)-1:

                        media_val_k1 = (float(media_val_k1+Lista_valores_coletados_k1[moeda+1][2])/2)
                        moeda += 1

                    print '\n\nMedia k1:', media_val_k1

                    for valores_agregados in Lista_valores_coletados_k1:
                        Dic_valores_coletados_por_conteudo_por_no[valores_agregados[0]]['k1'] = media_val_k1

                # calculo da media conteudo k2:
                if len(Lista_valores_coletados_k2) != 0:

                    moeda = 0
                    media_val_k2 = Lista_valores_coletados_k2[0][2]


                    while moeda < len(Lista_valores_coletados_k2)-1:

                        media_val_k2 = (float(media_val_k2+Lista_valores_coletados_k2[moeda+1][2])/2)
                        moeda += 1

                    if modo_debug == 'on':
                        print '\n\nMedia k2:', media_val_k2

                    for valores_agregados in Lista_valores_coletados_k2:
                        Dic_valores_coletados_por_conteudo_por_no[valores_agregados[0]]['k2'] = media_val_k2

                # calculo da media conteudo k3:
                if len(Lista_valores_coletados_k3) != 0:

                    moeda = 0
                    media_val_k3 = Lista_valores_coletados_k3[0][2]


                    while moeda < len(Lista_valores_coletados_k3)-1:

                        media_val_k3 = (float(media_val_k3+Lista_valores_coletados_k3[moeda+1][2])/2)
                        moeda += 1

                    if modo_debug == 'on':
                        print '\n\nMedia k3:', media_val_k3

                    for valores_agregados in Lista_valores_coletados_k3:
                        Dic_valores_coletados_por_conteudo_por_no[valores_agregados[0]]['k3'] = media_val_k3

                # calculo da media conteudo k4:
                if len(Lista_valores_coletados_k4) != 0:

                    moeda = 0
                    media_val_k4 = Lista_valores_coletados_k4[0][2]


                    while moeda < len(Lista_valores_coletados_k4)-1:

                        media_val_k4 = (float(media_val_k4+Lista_valores_coletados_k4[moeda+1][2])/2)
                        moeda += 1

                    if modo_debug == 'on':
                        print '\n\nMedia k4:', media_val_k4

                    for valores_agregados in Lista_valores_coletados_k4:
                        Dic_valores_coletados_por_conteudo_por_no[valores_agregados[0]]['k4'] = media_val_k4


                # calculo da media conteudo k5:
                if len(Lista_valores_coletados_k5) != 0:

                    moeda = 0
                    media_val_k5 = Lista_valores_coletados_k5[0][2]


                    while moeda < len(Lista_valores_coletados_k5)-1:

                        media_val_k5 = (float(media_val_k5+Lista_valores_coletados_k5[moeda+1][2])/2)
                        moeda += 1

                    if modo_debug == 'on':
                        print '\n\nMedia k5:', media_val_k5

                    for valores_agregados in Lista_valores_coletados_k5:
                        Dic_valores_coletados_por_conteudo_por_no[valores_agregados[0]]['k5'] = media_val_k5

                    if modo_debug == 'on':
                        print '\n\nGrandeza selecionada', Grandeza_selecionada


                # ===========================================
                # Resposta precisa ser enviada para o cliente:
                # ===========================================
                Valores_resposta_media = []
                Lista_dispositivos_valores_coletados = []

                for valor_coletado in Dic_valores_coletados_por_conteudo_por_no:
                    if (Dic_valores_coletados_por_conteudo_por_no[valor_coletado][Grandeza_selecionada]) != None:

                        if (t - Dic_tempo_ultima_atualizacao_por_conteudo_por_no[valor_coletado][Grandeza_selecionada]) <= tempo_validade_cache:
                                Valores_resposta_media.append(Dic_valores_coletados_por_conteudo_por_no[valor_coletado][Grandeza_selecionada])
                                Lista_dispositivos_valores_coletados.append([valor_coletado,Dic_valores_coletados_por_conteudo_por_no[valor_coletado][Grandeza_selecionada]])

                if modo_debug == 'on':
                    print 'Aqui é o erro:', Valores_resposta_media

                media_val_resposta = media(Valores_resposta_media)

                if modo_debug == 'on':
                    print 'Media resposta:', media_val_resposta

                # ===========================
                # Calculando o erro nesse caso:
                # ===========================
                if modo_debug == 'on':
                    print 'Lista', Lista_dispositivos_valores_coletados

                diferencas_quadraticas_respostas_reais = []

                for grupamento in Lista_dispositivos_valores_coletados:

                    diferenca = media_val_resposta - Dic_valores_distribuidos_aoi[grupamento[0]]['valor_'+Grandeza_selecionada]
                    diferencas_quadraticas_respostas_reais.append(diferenca**2)


                somatorio_diferencas_quadraticas = 0

                for diferencas in diferencas_quadraticas_respostas_reais:

                    somatorio_diferencas_quadraticas += diferencas

                RMSD_val_resposta_cliente = somatorio_diferencas_quadraticas/len(Lista_dispositivos_valores_coletados)

                RMSD_val_resposta_cliente = float((RMSD_val_resposta_cliente)**(0.5))

                if modo_debug == 'on':

                    print '\n\nRMSD_val_resposta_cliente:', RMSD_val_resposta_cliente

                Resultados_RMSD_respostas_clientes.append(RMSD_val_resposta_cliente)



                for i in Caminho_definido:
                    if Dic_dispositivo_visitado_por_MA[i] == None:
                        Dic_dispositivo_visitado_por_MA[i] = True



    return Resultados_RMSD_respostas_clientes,Dic_registro_energia_remanescente_dispositivos, Contador_envio_MA, List_cache_hits, Lista_media_nos_intermediarios, Media_qtde_nos_fontes


def determinacao_dos_indices_de_inicio_de_coleta_todos_casos(Nos_fontes_a_visitar, Selecionados_knapsack, Caminho_definido):
    if modo_debug == 'on':
        print '\n\nCaminho_definido:', Caminho_definido, len(Caminho_definido)
    index_caminho_definido = []
    Caminho_definido_invertido = Caminho_definido[::-1]
    Ultimo_indice = len(Caminho_definido)-1

    N_INDEX = 0
    for i in Caminho_definido:

        ultima_aparicao_do_no = Ultimo_indice-Caminho_definido_invertido.index(i)
        if ultima_aparicao_do_no != N_INDEX:
            index_caminho_definido.append([i,ultima_aparicao_do_no])

        if ultima_aparicao_do_no == N_INDEX:
            index_caminho_definido.append([i,ultima_aparicao_do_no])
        N_INDEX += 1

    # verificando onde iniciara coleta de nos fontes
    Indices_nos_fontes = []


    for no_caminho in Caminho_definido:

        for no_indice in Nos_fontes_a_visitar:

            if no_caminho == no_indice:

                Indices_nos_fontes.append([no_indice,Caminho_definido.index(no_caminho)])

    indices_conteudo_k1 = []
    indices_conteudo_k2 = []
    indices_conteudo_k3 = []
    indices_conteudo_k4 = []
    indices_conteudo_k5 = []

    for t in index_caminho_definido:
        if t[0] in Selecionados_knapsack['k1']:
            if t[1] >= len(Caminho_definido)/2:
                indices_conteudo_k1.append(t[1])

    for t in index_caminho_definido:
        if t[0] in Selecionados_knapsack['k2']:
            if t[1] >= len(Caminho_definido)/2:
                indices_conteudo_k2.append(t[1])

    for t in index_caminho_definido:
        if t[0] in Selecionados_knapsack['k3']:
            if t[1] >= len(Caminho_definido)/2:
                indices_conteudo_k3.append(t[1])

    for t in index_caminho_definido:
        if t[0] in Selecionados_knapsack['k4']:
            if t[1] >= len(Caminho_definido)/2:
                indices_conteudo_k4.append(t[1])

    for t in index_caminho_definido:
        if t[0] in Selecionados_knapsack['k5']:
            if t[1] >= len(Caminho_definido)/2:
                indices_conteudo_k5.append(t[1])


    if len(indices_conteudo_k1) == 0:
        indice_k1 = None
    if len(indices_conteudo_k1) != 0:
        indice_k1 = min(indices_conteudo_k1)

    if len(indices_conteudo_k2) == 0:
        indice_k2 = None
    if len(indices_conteudo_k2) != 0:
        indice_k2 = min(indices_conteudo_k2)

    if len(indices_conteudo_k3) == 0:
        indice_k3 = None
    if len(indices_conteudo_k3) != 0:
        indice_k3 = min(indices_conteudo_k3)

    if len(indices_conteudo_k4) == 0:
        indice_k4 = None
    if len(indices_conteudo_k4) != 0:
        indice_k4 = min(indices_conteudo_k4)

    if len(indices_conteudo_k5) == 0:
        indice_k5 = None
    if len(indices_conteudo_k5) != 0:
        indice_k5 = min(indices_conteudo_k5)

    indices_nos_fontes_a_visitar = [i[1] for i in Indices_nos_fontes]

    if modo_debug == 'on':

        print '\n\nCaminho_definido:', Caminho_definido
        print '\n\nNós fontes a visitar:', Nos_fontes_a_visitar
        print 'Coleta de nos fontes iniciará no nó de índice', min(indices_nos_fontes_a_visitar)
        print '\n\n\n\n\n', Selecionados_knapsack
        print '\n\nColeta o conteudo k1 iniciará no nó de índice', indice_k1
        print '\n\nColeta o conteudo k2 iniciará no nó de índice', indice_k2
        print '\n\nColeta o conteudo k3 iniciará no nó de índice', indice_k3
        print '\n\nColeta o conteudo k4 iniciará no nó de índice', indice_k4
        print '\n\nColeta o conteudo k5 iniciará no nó de índice', indice_k5

    return min(indices_nos_fontes_a_visitar), indice_k1, indice_k2, indice_k3, indice_k4, indice_k5, indices_nos_fontes_a_visitar, indices_conteudo_k1, indices_conteudo_k2, indices_conteudo_k3, indices_conteudo_k4, indices_conteudo_k5
