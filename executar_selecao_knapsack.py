# -*- coding: cp1252 -*-
import random
random.seed()
import networkx as nx
import sys
import time

Lista_knapsack_itens = []
Lista_knapsack_tamanhos = []

def printknapSack(W, wt, val, n):
    Lista_knapsack_itens = []
    Lista_knapsack_tamanhos = []

    #print '\n', val,'\n'

    K = [[0 for w in range(W + 1)]
            for i in range(n + 1)]

    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1]
                  + K[i - 1][w - wt[i - 1]],
                               K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    # stores the result of Knapsack
    res = K[n][W]
    print 'Premio:',round((res)/1000000000,3)

    w = W
    for i in range(n, 0, -1):
        if res <= 0:
            break

        if res == K[i - 1][w]:
            continue
        else:

            # This item is included.
            Lista_knapsack_itens.append(i-1)
            Lista_knapsack_tamanhos.append(wt[i-1])

            # Since this weight is included
            # its value is deducted
            res = res - val[i - 1]
            w = w - wt[i - 1]

    return Lista_knapsack_itens, Lista_knapsack_tamanhos



def selecao_knapsack(Dic_prioridades,Grandeza_selecionada,tempo_validade_cache,t,Lista_conteudos,Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k1, Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2, Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k3, Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k4, Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k5, Universo_set_cover_problem,Dict_cobertura_AoI_knapsack,Nos_fontes_a_visitar,Tamanho_max_coleta_knapsack_oportunista,dispositivos,Caminho_definido_sem_nos_fontes_sem_duplicados,Lista_pontos,Dic_tempo_ultima_atualizacao_por_conteudo_por_no):

    # Universo_set_cover_problem

    # criando lista com objetos dos nos intermediarios para avaliacao:
    objetos_dispositivos_intermediarios_avaliados = [i for i in dispositivos if i.nome in Caminho_definido_sem_nos_fontes_sem_duplicados]

    area_ainda_nao_coberta_k1 = set([])
    area_ainda_nao_coberta_k2 = set([])
    area_ainda_nao_coberta_k3 = set([])
    area_ainda_nao_coberta_k4 = set([])
    area_ainda_nao_coberta_k5 = set([])

    Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k1 = []
    for i in Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k1:
        if i != 'GW':
            if Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k1[i] != None:
                for j in Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k1[i]:
                    if j not in Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k1:
                        Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k1.append(j)

    Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2 = []
    for i in Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2:
        if i != 'GW':
            if Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2[i] != None:
                for j in Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2[i]:
                    if j not in Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2:
                        Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2.append(j)

    Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k3 = []
    for i in Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k3:
        if i != 'GW':
            if Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k3[i] != None:
                for j in Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k3[i]:
                    if j not in Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k3:
                        Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k3.append(j)

    Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k4 = []
    for i in Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k4:
        if i != 'GW':
            if Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k4[i] != None:
                for j in Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k4[i]:
                    if j not in Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k4:
                        Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k4.append(j)

    Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k5 = []
    for i in Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k5:
        if i != 'GW':
            if Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k5[i] != None:
                for j in Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k5[i]:
                    if j not in Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k5:
                        Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k5.append(j)


    conjunto_Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k1 = set(Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k1)
    conjunto_Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2 = set(Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2)
    conjunto_Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k3 = set(Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k3)
    conjunto_Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k4 = set(Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k4)
    conjunto_Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k5 = set(Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k5)


    # Pontuação de cada conteudo no conjunto de nós intermediários

    Pontuacao_k1 = 0
    Pontuacao_k2 = 0
    Pontuacao_k3 = 0
    Pontuacao_k4 = 0
    Pontuacao_k5 = 0

    Dic_dispositivos_itinerario_expirados_por_conteudo = {}
    for i in Lista_conteudos:
        Dic_dispositivos_itinerario_expirados_por_conteudo.update({i:[]})


    for conteudo in Lista_conteudos:
        for dispositivo in objetos_dispositivos_intermediarios_avaliados:
            if Dic_tempo_ultima_atualizacao_por_conteudo_por_no[dispositivo.nome][conteudo] == None or (t - Dic_tempo_ultima_atualizacao_por_conteudo_por_no[dispositivo.nome][conteudo]) > tempo_validade_cache/2:
                Dic_dispositivos_itinerario_expirados_por_conteudo[conteudo].append(dispositivo.nome)


    #calculo da pontuação knapsack
    for conteudo in Lista_conteudos:
            for dispositivo in Dic_dispositivos_itinerario_expirados_por_conteudo[conteudo]:
                conjunto_teste = set(Dict_cobertura_AoI_knapsack[dispositivo])

                if conteudo == 'k1':

                    Pontuacao_k1 += 1000 * round(Dic_prioridades['prob_k1'] * (len(conjunto_teste - conjunto_teste.intersection(conjunto_Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k1))),3)

                if conteudo == 'k2':

                    Pontuacao_k2 += 1000 * round(Dic_prioridades['prob_k2'] * (len(conjunto_teste - conjunto_teste.intersection(conjunto_Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2))),3)

                if conteudo == 'k3':

                    Pontuacao_k3 += 1000 * round(Dic_prioridades['prob_k3'] * (len(conjunto_teste - conjunto_teste.intersection(conjunto_Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k3))),3)

                if conteudo == 'k4':

                    Pontuacao_k4 += 1000 * round(Dic_prioridades['prob_k4'] * (len(conjunto_teste - conjunto_teste.intersection(conjunto_Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k4))),3)

                if conteudo == 'k5':

                    Pontuacao_k5 += 1000 * round(Dic_prioridades['prob_k5'] * (len(conjunto_teste - conjunto_teste.intersection(conjunto_Lista_dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k5))),3)


    W = Tamanho_max_coleta_knapsack_oportunista

    if Grandeza_selecionada == 'k1':
       val = [round(Pontuacao_k2,3),round(Pontuacao_k3,3),round(Pontuacao_k4,3),round(Pontuacao_k5,3)]
    if Grandeza_selecionada == 'k2':
       val = [round(Pontuacao_k1,3),round(Pontuacao_k3,3),round(Pontuacao_k4,3),round(Pontuacao_k5,3)]
    if Grandeza_selecionada == 'k3':
       val = [round(Pontuacao_k1,3),round(Pontuacao_k2,3),round(Pontuacao_k4,3),round(Pontuacao_k5,3)]
    if Grandeza_selecionada == 'k4':
       val = [round(Pontuacao_k1,3),round(Pontuacao_k2,3),round(Pontuacao_k3,3),round(Pontuacao_k5,3)]
    if Grandeza_selecionada == 'k5':
       val = [round(Pontuacao_k1,3),round(Pontuacao_k2,3),round(Pontuacao_k3,3),round(Pontuacao_k4,3)]

    wt = [20 for k in val]

    n = len(val)



    Lista_knapsack_itens, Lista_knapsack_tamanhos = printknapSack(W, wt, val, n)

    Lista_knapsack_itens.sort()


    if Grandeza_selecionada == 'k1':
        Lista_conteudos = ['k2','k3','k4','k5']
        Itens_selecionados_prev = [Lista_conteudos[n] for n in Lista_knapsack_itens]

    if Grandeza_selecionada == 'k2':
        Lista_conteudos = ['k1','k3','k4','k5']
        Itens_selecionados_prev = [Lista_conteudos[n] for n in Lista_knapsack_itens]

    if Grandeza_selecionada == 'k3':
        Lista_conteudos = ['k1','k2','k4','k5']
        Itens_selecionados_prev = [Lista_conteudos[n] for n in Lista_knapsack_itens]

    if Grandeza_selecionada == 'k4':
        Lista_conteudos = ['k1','k2','k3','k5']
        Itens_selecionados_prev = [Lista_conteudos[n] for n in Lista_knapsack_itens]

    if Grandeza_selecionada == 'k5':
        Lista_conteudos = ['k1','k2','k3','k4']
        Itens_selecionados_prev = [Lista_conteudos[n] for n in Lista_knapsack_itens]

    Itens_selecionados = []

    for i in Itens_selecionados_prev:
        if i not in Itens_selecionados:
            Itens_selecionados.append(i)


    print 'Itens selecionados importante!!!', Itens_selecionados, W



    Dic_selecionados_knapsack_por_conteudo = {'k1':[],'k2':[],'k3':[],'k4':[],'k5':[]}

    for i in Itens_selecionados:
        Dic_selecionados_knapsack_por_conteudo[i] = Dic_dispositivos_itinerario_expirados_por_conteudo[i]


    return Dic_selecionados_knapsack_por_conteudo, Dic_tempo_ultima_atualizacao_por_conteudo_por_no, Dic_selecionados_knapsack_por_conteudo
