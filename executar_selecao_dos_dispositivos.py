# -*- coding: cp1252 -*-
from collections import defaultdict, deque
from scipy.spatial import distance
import random
import time
import numpy as np
from scipy.stats import zipf
from bisect import bisect
import networkx as nx

def selecao_dispositivos_detalhada(Dispositivo_atualizado_no_cache,conjunto_area_coberta_pelo_cache,Dic_sensing_ranges,Grandeza_selecionada,t,Dict_cobertura_AoI,Universo_set_cover_problem,Conjunto_pontos_cobertos,pontos_restantes,Dic_tempo_ultima_atualizacao_por_conteudo_por_no,Pontos_cobertos):

        if modo_debug == 'on':
            print '\033[31m'+'\n\n##### Seleção e coleta da rodada #####\n\n'+'\033[0;0m'
            print 'Universo original:',len(Universo_set_cover_problem)
            print 'cobertura do cache:',len(conjunto_area_coberta_pelo_cache)

        Universo_set_cover_problem = Universo_set_cover_problem - conjunto_area_coberta_pelo_cache

        if modo_debug == 'on':
            print 'Universo compensado:',len(Universo_set_cover_problem)

        if Universo_set_cover_problem == set([]):

            if modo_debug == 'on':
                print '\n\nToda AoI para o conteúdo', Grandeza_selecionada,'está coberta pelo cache! Nenhuma nova seleção nem coleta de dados na rede é necessária.'

            Selecionados = []

            return Selecionados

        if Universo_set_cover_problem != set([]):
            n_rodadas = 0


            if len(pontos_restantes) == 0:
                  print 'entra aqui'

                  while n_rodadas <= 0:

                      class conjuntos:
                          def __init__(self,nome):
                              self.nome = nome
                              self.elementos = None
                              self.index = None

                      # Universo de pontos para analisar

                      U = set(Universo_set_cover_problem)
                      R = U

                      S = []

                      for i in Dict_cobertura_AoI:
                          i = set(Dict_cobertura_AoI[i])
                          S.append(i)

                      nomes = Dict_cobertura_AoI.keys()

                      Dic_pontuacao_selecao_dispositivos = {}

                      for node in Dic_tempo_ultima_atualizacao_por_conteudo_por_no:
                          Dic_pontuacao_selecao_dispositivos.update({node:0})


                      for no in Dic_tempo_ultima_atualizacao_por_conteudo_por_no:
                          for conteudo in Dic_tempo_ultima_atualizacao_por_conteudo_por_no[no]:
                              if conteudo == Grandeza_selecionada and no != 'GW':
                                  if Dic_tempo_ultima_atualizacao_por_conteudo_por_no[no][conteudo] != None:
                                      Dic_pontuacao_selecao_dispositivos[no] = Dic_tempo_ultima_atualizacao_por_conteudo_por_no[no][conteudo]
                                  if Dic_tempo_ultima_atualizacao_por_conteudo_por_no[no][conteudo] == None:
                                      Dic_pontuacao_selecao_dispositivos[no] = 0

                      w = []
                      for i in nomes:
                          for j in Dic_pontuacao_selecao_dispositivos:
                                   if i == j:
                                       w.append(Dic_pontuacao_selecao_dispositivos[j])


                      dado1 = round(random.random(),3)
                      if dado1 >= 0.50:
                          w = w[::-1]
                          nomes = nomes[::-1]

                      n = 0

                      for i in S:
                          i = conjuntos(nomes[n])
                          n += 1

                      C = []
                      costs = []

                      def findMin(S, R):
                          minCost = 99999.0
                          minElement = -1
                          for i, s in enumerate(S):
                              try:
                                  cost = w[i]/(len(s.intersection(R)))
                                  if cost < minCost:
                                      minCost = cost
                                      minElement = i
                              except:
                                  # Division by zero, ignore
                                  pass
                          return S[minElement], w[minElement], minElement

                      Selecionados = []

                      while len(R) != 0:
                          S_i, cost, index = findMin(S, R)
                          C.append(S_i)
                          R = R.difference(S_i)
                          costs.append(cost)
                          Selecionados.append(nomes[index])

                      # prova real
                      if len(Selecionados) != 0:
                          teste = [set(Dict_cobertura_AoI[i]) for i in Selecionados]

                          teste = set.union(*teste)

                          n_rodadas += 1

                      else:

                          n_rodadas += 1

            else:

                Selecionados = []
                if modo_debug == 'on':
                    print "\nA região não está completamente coberta!"

                n_rodadas += 1


            for i in Selecionados:
                if i in Dispositivo_atualizado_no_cache:
                    Selecionados.pop(Selecionados.index(i))

            if modo_debug == 'on':
                print '\n\nDispositivos selecionados para range do conteúdo',Grandeza_selecionada,'(',Dic_sensing_ranges[Grandeza_selecionada],' metros):', Selecionados


            return Selecionados

def selecao_dos_dispositivos(Dic_sensing_ranges,Pontos_cobertos_k1,Pontos_cobertos_k2,Pontos_cobertos_k3,Pontos_cobertos_k4,Pontos_cobertos_k5,pontos_restantes_k1,pontos_restantes_k2,pontos_restantes_k3,pontos_restantes_k4,pontos_restantes_k5,Conjunto_pontos_cobertos_k1,Conjunto_pontos_cobertos_k2,Conjunto_pontos_cobertos_k3,Conjunto_pontos_cobertos_k4,Conjunto_pontos_cobertos_k5,Universo_set_cover_problem,Dict_cobertura_AoI_k1,Dict_cobertura_AoI_k2,Dict_cobertura_AoI_k3,Dict_cobertura_AoI_k4,Dict_cobertura_AoI_k5,tempo_validade_cache, Dic_falhas_por_dispositivos_normalizado, plt, Grandeza_selecionada, Dic_falhas_por_dispositivos, dispositivos, Dic_tempo_ultima_atualizacao_por_conteudo_por_no, t, Dic_valores_coletados_por_conteudo_por_no):

    # Essa variável irá controlar se a selecao será feita somente com conteúdo do cache ou se será necessária selecao na rede. 0 quer dizer que os conteudos em cache fornecem parte ou toda resposta
    Variavel_controle = 0


    Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k1 = {}
    Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2 = {}
    Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k3 = {}
    Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k4 = {}
    Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k5 = {}
    Dispositivo_atualizado_no_cache = []


    for i in Dic_tempo_ultima_atualizacao_por_conteudo_por_no:
        Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k1.update({i:[]})

    for i in Dic_tempo_ultima_atualizacao_por_conteudo_por_no:
        Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2.update({i:[]})

    for i in Dic_tempo_ultima_atualizacao_por_conteudo_por_no:
        Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k3.update({i:[]})

    for i in Dic_tempo_ultima_atualizacao_por_conteudo_por_no:
        Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k4.update({i:[]})

    for i in Dic_tempo_ultima_atualizacao_por_conteudo_por_no:
        Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k5.update({i:[]})


    # Pontos da AoI cobertos por cada sensor
    if Grandeza_selecionada == 'k1':
        Dict_cobertura_AoI = Dict_cobertura_AoI_k1

    if Grandeza_selecionada == 'k2':
        Dict_cobertura_AoI = Dict_cobertura_AoI_k2

    if Grandeza_selecionada == 'k3':
        Dict_cobertura_AoI = Dict_cobertura_AoI_k3

    if Grandeza_selecionada == 'k4':
        Dict_cobertura_AoI = Dict_cobertura_AoI_k4

    if Grandeza_selecionada == 'k5':
        Dict_cobertura_AoI = Dict_cobertura_AoI_k5

    if Grandeza_selecionada == 'k1':
        Pontos_cobertos = Pontos_cobertos_k1

    if Grandeza_selecionada == 'k2':
        Pontos_cobertos = Pontos_cobertos_k2

    if Grandeza_selecionada == 'k3':
        Pontos_cobertos = Pontos_cobertos_k3

    if Grandeza_selecionada == 'k4':
        Pontos_cobertos = Pontos_cobertos_k4

    if Grandeza_selecionada == 'k5':
        Pontos_cobertos = Pontos_cobertos_k5


    if Grandeza_selecionada == 'k1':
        Conjunto_pontos_cobertos = Conjunto_pontos_cobertos_k1

    if Grandeza_selecionada == 'k2':
        Conjunto_pontos_cobertos = Conjunto_pontos_cobertos_k2

    if Grandeza_selecionada == 'k3':
        Conjunto_pontos_cobertos = Conjunto_pontos_cobertos_k3

    if Grandeza_selecionada == 'k4':
        Conjunto_pontos_cobertos = Conjunto_pontos_cobertos_k4

    if Grandeza_selecionada == 'k5':
        Conjunto_pontos_cobertos = Conjunto_pontos_cobertos_k5

    # Conjunto_pontos_cobertos = set(Pontos_cobertos.keys())
    if Grandeza_selecionada == 'k1':
        pontos_restantes = pontos_restantes_k1
    if Grandeza_selecionada == 'k2':
        pontos_restantes = pontos_restantes_k2
    if Grandeza_selecionada == 'k3':
        pontos_restantes = pontos_restantes_k3
    if Grandeza_selecionada == 'k4':
        pontos_restantes = pontos_restantes_k4
    if Grandeza_selecionada == 'k5':
        pontos_restantes = pontos_restantes_k5


    # atualizando a cobertura do cache por conteudo
    for dispositivo in Dic_tempo_ultima_atualizacao_por_conteudo_por_no:
        if dispositivo != 'GW':
            if Dic_tempo_ultima_atualizacao_por_conteudo_por_no[dispositivo]['k1'] == None:
                Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k1[dispositivo] = None

            if Dic_tempo_ultima_atualizacao_por_conteudo_por_no[dispositivo]['k1'] != None and Dic_valores_coletados_por_conteudo_por_no[dispositivo]['k1'] != None:
                if (t - Dic_tempo_ultima_atualizacao_por_conteudo_por_no[dispositivo]['k1']) <= tempo_validade_cache:

                    Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k1[dispositivo] = True

                if (t - Dic_tempo_ultima_atualizacao_por_conteudo_por_no[dispositivo]['k1']) > tempo_validade_cache:

                    Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k1[dispositivo] = None

            if Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k1[dispositivo] == True:
                Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k1[dispositivo] = Dict_cobertura_AoI[dispositivo]
                if dispositivo not in Dispositivo_atualizado_no_cache:
                    Dispositivo_atualizado_no_cache.append(dispositivo)

            if Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2[dispositivo] == True:
                Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2[dispositivo] = Dict_cobertura_AoI[dispositivo]

            if Dic_tempo_ultima_atualizacao_por_conteudo_por_no[dispositivo]['k2'] == None:
                Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2[dispositivo] = None

            if Dic_tempo_ultima_atualizacao_por_conteudo_por_no[dispositivo]['k2'] != None and Dic_valores_coletados_por_conteudo_por_no[dispositivo]['k2'] != None:
                if (t - Dic_tempo_ultima_atualizacao_por_conteudo_por_no[dispositivo]['k2']) <= tempo_validade_cache:

                    Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2[dispositivo] = True

                if (t - Dic_tempo_ultima_atualizacao_por_conteudo_por_no[dispositivo]['k2']) > tempo_validade_cache:

                    Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2[dispositivo] = None

            if Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2[dispositivo] == True:
                Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2[dispositivo] = Dict_cobertura_AoI[dispositivo]
                if dispositivo not in Dispositivo_atualizado_no_cache:
                    Dispositivo_atualizado_no_cache.append(dispositivo)

            if Dic_tempo_ultima_atualizacao_por_conteudo_por_no[dispositivo]['k3'] == None:
                Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k3[dispositivo] = None

            if Dic_tempo_ultima_atualizacao_por_conteudo_por_no[dispositivo]['k3'] != None and Dic_valores_coletados_por_conteudo_por_no[dispositivo]['k3'] != None:
                if (t - Dic_tempo_ultima_atualizacao_por_conteudo_por_no[dispositivo]['k3']) <= tempo_validade_cache:

                    Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k3[dispositivo] = True

                if (t - Dic_tempo_ultima_atualizacao_por_conteudo_por_no[dispositivo]['k3']) > tempo_validade_cache:

                    Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k3[dispositivo] = None

            if Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k3[dispositivo] == True:
                Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k3[dispositivo] = Dict_cobertura_AoI[dispositivo]
                if dispositivo not in Dispositivo_atualizado_no_cache:
                    Dispositivo_atualizado_no_cache.append(dispositivo)

            if Dic_tempo_ultima_atualizacao_por_conteudo_por_no[dispositivo]['k4'] == None:
                Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k4[dispositivo] = None

            if Dic_tempo_ultima_atualizacao_por_conteudo_por_no[dispositivo]['k4'] != None and Dic_valores_coletados_por_conteudo_por_no[dispositivo]['k4'] != None:
                if (t - Dic_tempo_ultima_atualizacao_por_conteudo_por_no[dispositivo]['k4']) <= tempo_validade_cache:

                    Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k4[dispositivo] = True

                if (t - Dic_tempo_ultima_atualizacao_por_conteudo_por_no[dispositivo]['k4']) > tempo_validade_cache:

                    Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k4[dispositivo] = None

            if Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k4[dispositivo] == True:
                Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k4[dispositivo] = Dict_cobertura_AoI[dispositivo]
                if dispositivo not in Dispositivo_atualizado_no_cache:
                    Dispositivo_atualizado_no_cache.append(dispositivo)

            if Dic_tempo_ultima_atualizacao_por_conteudo_por_no[dispositivo]['k5'] == None:
                Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k5[dispositivo] = None

            if Dic_tempo_ultima_atualizacao_por_conteudo_por_no[dispositivo]['k5'] != None and Dic_valores_coletados_por_conteudo_por_no[dispositivo]['k5'] != None:
                if (t - Dic_tempo_ultima_atualizacao_por_conteudo_por_no[dispositivo]['k5']) <= tempo_validade_cache:

                    Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k5[dispositivo] = True

                if (t - Dic_tempo_ultima_atualizacao_por_conteudo_por_no[dispositivo]['k5']) > tempo_validade_cache:

                    Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k5[dispositivo] = None

            if Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k5[dispositivo] == True:
                Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k5[dispositivo] = Dict_cobertura_AoI[dispositivo]
                if dispositivo not in Dispositivo_atualizado_no_cache:
                    Dispositivo_atualizado_no_cache.append(dispositivo)

    # area coberta pelo cache
    area_coberta_pelo_cache = []

    if Grandeza_selecionada == 'k1':
        for dispositivo in Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k1:
            if dispositivo != 'GW':
                if Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k1[dispositivo] != None and Dic_valores_coletados_por_conteudo_por_no[dispositivo]['k1'] != None:
                    for i in Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k1[dispositivo]:
                        area_coberta_pelo_cache.append(i)

    if Grandeza_selecionada == 'k2':
        for dispositivo in Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2:
            if dispositivo != 'GW':
                if Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2[dispositivo] != None and Dic_valores_coletados_por_conteudo_por_no[dispositivo]['k2'] != None:
                    for i in Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2[dispositivo]:
                        area_coberta_pelo_cache.append(i)

    if Grandeza_selecionada == 'k3':
        for dispositivo in Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k3:
            if dispositivo != 'GW':
                if Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k3[dispositivo] != None and Dic_valores_coletados_por_conteudo_por_no[dispositivo]['k3'] != None:
                    for i in Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k3[dispositivo]:
                        area_coberta_pelo_cache.append(i)

    if Grandeza_selecionada == 'k4':
        for dispositivo in Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k4:
            if dispositivo != 'GW':
                if Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k4[dispositivo] != None and Dic_valores_coletados_por_conteudo_por_no[dispositivo]['k4'] != None:
                    for i in Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k4[dispositivo]:
                        area_coberta_pelo_cache.append(i)

    if Grandeza_selecionada == 'k5':
        for dispositivo in Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k5:
            if dispositivo != 'GW':
                if Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k5[dispositivo] != None and Dic_valores_coletados_por_conteudo_por_no[dispositivo]['k5'] != None:
                    for i in Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k5[dispositivo]:
                        area_coberta_pelo_cache.append(i)


    conjunto_area_coberta_pelo_cache = set(area_coberta_pelo_cache)

    Selecionados = selecao_dispositivos_detalhada(Dispositivo_atualizado_no_cache,conjunto_area_coberta_pelo_cache,Dic_sensing_ranges,Grandeza_selecionada,t,Dict_cobertura_AoI,Universo_set_cover_problem,Conjunto_pontos_cobertos,pontos_restantes,Dic_tempo_ultima_atualizacao_por_conteudo_por_no,Pontos_cobertos)

    Dict_cobertura_AoI_knapsack = {}

    for i in Dict_cobertura_AoI:
        Dict_cobertura_AoI_knapsack.update({i:Dict_cobertura_AoI[i]})


    return Dic_tempo_ultima_atualizacao_por_conteudo_por_no, Selecionados, Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k1, Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k2, Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k3, Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k4, Dic_dispositivos_atualizados_no_cache_e_suas_coberturas_k5, Dict_cobertura_AoI_knapsack
