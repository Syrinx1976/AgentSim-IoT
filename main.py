# -*- coding: cp1252 -*-
import gerar_aoi_dispositivos_c_coordenadas_rev0
from collections import defaultdict, deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random
import time
import executar_coleta_em_dispositivos_ativos_com_MA
import calculos
import networkx as nx
import sys
import os


def media(L):
  'Calcula RTTs medios dos arquivos Resultados_medias.dat'
  soma=0
  for i in range(len(L)):
    soma+=L[i]
  media=float(float(soma)/(len(L)))
  return media

# Parametros de entrada
# numero de dispositivos distribuidos na AoI
n = int(sys.argv[1])
tipo_distribuicao = int(sys.argv[2])
tempo_validade_c = int(sys.argv[3])
prob_falha_links = float(sys.argv[4])
largura_top = int(sys.argv[5])
comprimento_top = int(sys.argv[6])
tamanho_inicial_do_MA = int(sys.argv[7])
modo_debug = str(sys.argv[8])
coord_gateway_x = int(sys.argv[9])
coord_gateway_y = int(sys.argv[10])
numero_rodadas = int(sys.argv[11])

# numero de arestas para construcao do grafo
m = 0

# abre arquivos para armazenamento temporario dos resultados
g = open('temp_armazenamento_media_intermd.txt','w')
g.close()

g = open('temp_armazenamento_media_nos_fontes.txt','w')
g.close()

g = open('temp_armazenamento_cache_hits.txt','w')
g.close()

g = open('temp_armazenamento_media_RMSD.txt','w')
g.close()

g = open('temp_armazenamento_media_energia.txt','w')
g.close()

# inicio de loop com processo de recebimento de requisicoes e
rodada = 0
while rodada < numero_rodadas:

    # Instancias dos Dicionarios e Listas principais
    Dic_sensing_ranges = {}
    Dic_reqs_dispositivos = {}
    Dic_falhas_por_dispositivos = {}
    Lista_conteudos = ['k1','k2','k3','k4','k5']
    Dispositivos_selecionados_por_conteudo = {'k1':None,'k2':None,'k3':None,'k4':None,'k5':None}
    Dic_dispositivo_visitado_por_MA = {}
    Dic_registro_energia_remanescente_dispositivos = {}
    Dic_tempo_ultima_atualizacao_por_conteudo_por_no = {}
    Dic_valores_coletados_por_conteudo_por_no = {}
    contador_reqs_conteudos = {'k1':0,'k2':0,'k3':0,'k4':0,'k5':0}
    Dic_selecionado_primeiro = {}
    Dic_selecionado_segundo = {}
    Dic_selecionado_terceiro = {}
    Dic_pop_conteudo = {'k1':0,'k2':0,'k3':0,'k4':0,'k5':0}
    Dic_todos_nos_selecionados = {}
    Dic_energia_media_consumida_por_conjunto_de_sensing_nodes_k1 = {}
    Dic_energia_media_consumida_por_conjunto_de_sensing_nodes_k2 = {}
    Dic_energia_media_consumida_por_conjunto_de_sensing_nodes_k3 = {}
    Dic_energia_media_consumida_por_conjunto_de_sensing_nodes_k4 = {}
    Dic_energia_media_consumida_por_conjunto_de_sensing_nodes_k5 = {}
    Dic_resultados_erros = {}
    Lista_media_nos_intermediarios = []
    Media_qtde_nos_fontes = []

    # Criação do grafo e coordenadas
    dispositivos, Lista_pontos, G, Dic_registro_custos_originais = gerar_aoi_dispositivos_c_coordenadas_rev0.gera_aoi_dispositivos(plt, largura_top, comprimento_top, n, m, prob_falha_links, modo_debug)

    for i in dispositivos:

        Dic_sensing_ranges.update({i.conteudo_sensor_1:i.sensing_range_conteudo_1,i.conteudo_sensor_2:i.sensing_range_conteudo_2,i.conteudo_sensor_3:i.sensing_range_conteudo_3,i.conteudo_sensor_4:i.sensing_range_conteudo_4,i.conteudo_sensor_5:i.sensing_range_conteudo_5})
        Dic_reqs_dispositivos.update({i.nome:0})
        Dic_dispositivo_visitado_por_MA.update({i.nome:None})
        Dic_registro_energia_remanescente_dispositivos.update({i.nome:i.energia})
        Dic_tempo_ultima_atualizacao_por_conteudo_por_no.update({i.nome:{'k1':None,'k2':None,'k3':None,'k4':None,'k5':None}})
        Dic_valores_coletados_por_conteudo_por_no.update({i.nome:{'k1':None,'k2':None,'k3':None,'k4':None,'k5':None}})
        Dic_selecionado_primeiro.update({i.nome:0})
        Dic_selecionado_segundo.update({i.nome:0})
        Dic_selecionado_terceiro.update({i.nome:0})
        Dic_falhas_por_dispositivos.update({i.nome:0})
        List_cache_hits = [0,0]

    # Pontos da AoI cobertos por cada sensor
    Dict_cobertura_AoI_k1 = {}
    Dict_cobertura_AoI_k2 = {}
    Dict_cobertura_AoI_k3 = {}
    Dict_cobertura_AoI_k4 = {}
    Dict_cobertura_AoI_k5 = {}


    for i in dispositivos:
      if i.nome != 'GW':
          Dict_cobertura_AoI_k1.update({i.nome:[]})
          Dict_cobertura_AoI_k2.update({i.nome:[]})
          Dict_cobertura_AoI_k3.update({i.nome:[]})
          Dict_cobertura_AoI_k4.update({i.nome:[]})
          Dict_cobertura_AoI_k5.update({i.nome:[]})


    for conteudo in Lista_conteudos:
        if conteudo == 'k1':
            for i in dispositivos:
              for j in Lista_pontos:

                  if (distance.euclidean(j.coordenadas_pontos, i.coordenada_dispositivo)) <= Dic_sensing_ranges['k1']:
                      if i.nome != 'GW':
                          Dict_cobertura_AoI_k1[i.nome].append(j.nome)

        if conteudo == 'k2':
            for i in dispositivos:
              for j in Lista_pontos:

                  if (distance.euclidean(j.coordenadas_pontos, i.coordenada_dispositivo)) <= Dic_sensing_ranges['k2']:
                      if i.nome != 'GW':
                          Dict_cobertura_AoI_k2[i.nome].append(j.nome)

        if conteudo == 'k3':
            for i in dispositivos:
              for j in Lista_pontos:

                  if (distance.euclidean(j.coordenadas_pontos, i.coordenada_dispositivo)) <= Dic_sensing_ranges['k3']:
                      if i.nome != 'GW':
                          Dict_cobertura_AoI_k3[i.nome].append(j.nome)

        if conteudo == 'k4':
            for i in dispositivos:
              for j in Lista_pontos:

                  if (distance.euclidean(j.coordenadas_pontos, i.coordenada_dispositivo)) <= Dic_sensing_ranges['k4']:
                      if i.nome != 'GW':
                          Dict_cobertura_AoI_k4[i.nome].append(j.nome)

        if conteudo == 'k5':
            for i in dispositivos:
              for j in Lista_pontos:

                  if (distance.euclidean(j.coordenadas_pontos, i.coordenada_dispositivo)) <= Dic_sensing_ranges['k5']:
                      if i.nome != 'GW':
                          Dict_cobertura_AoI_k5[i.nome].append(j.nome)


    Lista_pontos_nomes = [i.nome for i in Lista_pontos]
    Pontos_cobertos_k1 = {}
    Pontos_cobertos_k2 = {}
    Pontos_cobertos_k3 = {}
    Pontos_cobertos_k4 = {}
    Pontos_cobertos_k5 = {}

    for conteudo in Lista_conteudos:
        if conteudo == 'k1':
            for i in Lista_pontos_nomes:
              for j in Dict_cobertura_AoI_k1:
                  if i in Dict_cobertura_AoI_k1[j]:
                      Pontos_cobertos_k1.update({i:[]})

        if conteudo == 'k2':
            for i in Lista_pontos_nomes:
              for j in Dict_cobertura_AoI_k2:
                  if i in Dict_cobertura_AoI_k2[j]:
                      Pontos_cobertos_k2.update({i:[]})

        if conteudo == 'k3':
            for i in Lista_pontos_nomes:
              for j in Dict_cobertura_AoI_k3:
                  if i in Dict_cobertura_AoI_k3[j]:
                      Pontos_cobertos_k3.update({i:[]})

        if conteudo == 'k4':
            for i in Lista_pontos_nomes:
              for j in Dict_cobertura_AoI_k4:
                  if i in Dict_cobertura_AoI_k4[j]:
                      Pontos_cobertos_k4.update({i:[]})

        if conteudo == 'k5':
            for i in Lista_pontos_nomes:
              for j in Dict_cobertura_AoI_k5:
                  if i in Dict_cobertura_AoI_k5[j]:
                      Pontos_cobertos_k5.update({i:[]})

    Universo_set_cover_problem = set(Lista_pontos_nomes)

    Conjunto_pontos_cobertos_k1 = set(Pontos_cobertos_k1.keys())
    Conjunto_pontos_cobertos_k2 = set(Pontos_cobertos_k2.keys())
    Conjunto_pontos_cobertos_k3 = set(Pontos_cobertos_k3.keys())
    Conjunto_pontos_cobertos_k4 = set(Pontos_cobertos_k4.keys())
    Conjunto_pontos_cobertos_k5 = set(Pontos_cobertos_k5.keys())


    pontos_restantes_k1 = Universo_set_cover_problem - Conjunto_pontos_cobertos_k1
    pontos_restantes_k2 = Universo_set_cover_problem - Conjunto_pontos_cobertos_k2
    pontos_restantes_k3 = Universo_set_cover_problem - Conjunto_pontos_cobertos_k3
    pontos_restantes_k4 = Universo_set_cover_problem - Conjunto_pontos_cobertos_k4
    pontos_restantes_k5 = Universo_set_cover_problem - Conjunto_pontos_cobertos_k5

    # Execução do processo de coleta
    Dic_registro_energia_remanescente_dispositivos, Contador_envio_MA, Resultados_RMSD_respostas_clientes, List_cache_hits, Media_qtde_nos_fontes = executar_coleta_em_dispositivos_ativos_com_MA.coleta_de_dados(Pontos_cobertos_k1, Pontos_cobertos_k2, Pontos_cobertos_k3, Pontos_cobertos_k4, Pontos_cobertos_k5, pontos_restantes_k1, pontos_restantes_k2, pontos_restantes_k3, pontos_restantes_k4, pontos_restantes_k5, Conjunto_pontos_cobertos_k1, Conjunto_pontos_cobertos_k2, Conjunto_pontos_cobertos_k3, Conjunto_pontos_cobertos_k4, Conjunto_pontos_cobertos_k5, Universo_set_cover_problem, Dict_cobertura_AoI_k1, Dict_cobertura_AoI_k2, Dict_cobertura_AoI_k3, Dict_cobertura_AoI_k4, Dict_cobertura_AoI_k5, Dic_dispositivo_visitado_por_MA, Dic_sensing_ranges, Lista_pontos, plt, dispositivos, Dic_reqs_dispositivos, Dispositivos_selecionados_por_conteudo, Dic_registro_energia_remanescente_dispositivos, contador_reqs_conteudos, Dic_pop_conteudo, G, Dic_registro_custos_originais, Dic_energia_media_consumida_por_conjunto_de_sensing_nodes_k1, Dic_energia_media_consumida_por_conjunto_de_sensing_nodes_k2, Dic_energia_media_consumida_por_conjunto_de_sensing_nodes_k3, Dic_energia_media_consumida_por_conjunto_de_sensing_nodes_k4, Dic_energia_media_consumida_por_conjunto_de_sensing_nodes_k5, Dic_falhas_por_dispositivos, Lista_conteudos, Dic_tempo_ultima_atualizacao_por_conteudo_por_no,tempo_validade_c,tipo_distribuicao, Dic_valores_coletados_por_conteudo_por_no, List_cache_hits, Media_qtde_nos_fontes, tamanho_inicial_do_MA, modo_debug, coord_gateway_x, coord_gateway_y)

    # Calculo media_nos_fontes

    Media_qtde_nos_fontes_1 = media(Media_qtde_nos_fontes)

    # Calculo cache Cache_hits_todos

    Cache_hits_percent = round((float(List_cache_hits[0])/float(List_cache_hits[1])),10)

    # Calculo do gasto de energia
    soma = 0
    for i in Dic_registro_energia_remanescente_dispositivos:
        soma += Dic_registro_energia_remanescente_dispositivos[i]

    Media_energia = soma/len(Dic_registro_energia_remanescente_dispositivos)

    Media_RMSD = media(Resultados_RMSD_respostas_clientes)

    # Registro dos resultados do processo de coleta corrente

    g = open('temp_armazenamento_media_nos_fontes.txt','a')
    g.write(str(Media_qtde_nos_fontes_1)+'\n')
    g.close()

    g = open('temp_armazenamento_cache_hits.txt','a')
    g.write(str(Cache_hits_percent)+'\n')
    g.close()

    g = open('temp_armazenamento_media_RMSD.txt','a')
    g.write(str(Media_RMSD)+'\n')
    g.close()

    g = open('temp_armazenamento_media_energia.txt','a')
    g.write(str(Media_energia)+'\n')
    g.close()

    g = open('temp_armazenamento_contador_envio.txt','a')
    g.write(str(Contador_envio_MA)+'\n')
    g.close()

    rodada += 1

h = open('temp_armazenamento_media_nos_fontes.txt')
wtMatrix = h.readlines()
planilha_final_qtde_nos_fontes = []
for i in wtMatrix:
    planilha_final_qtde_nos_fontes.append(float(i.split('\n')[0]))

h = open('temp_armazenamento_cache_hits.txt')
wtMatrix = h.readlines()
planilha_final_cache_hits = []
for i in wtMatrix:
    planilha_final_cache_hits.append(float(i.split('\n')[0]))

h = open('temp_armazenamento_media_RMSD.txt')
wtMatrix = h.readlines()
planilha_final_Media_RMSD = []
for i in wtMatrix:
    planilha_final_Media_RMSD.append(float(i.split('\n')[0]))

h = open('temp_armazenamento_media_energia.txt')
wtMatrix = h.readlines()
planilha_final_media_energia = []
for i in wtMatrix:
    planilha_final_media_energia.append(float(i.split('\n')[0]))

h = open('temp_armazenamento_contador_envio.txt')
wtMatrix = h.readlines()
planilha_final_contador_envio = []
for i in wtMatrix:
    planilha_final_contador_envio.append(float(i.split('\n')[0]))

f = open('Resultados_Media_energia_rede_tradicional_ndispositivos_'+str(n)+'t_distribuicao_'+str(tipo_distribuicao)+'t_validade_'+str(tempo_validade_c)+'prob_falha'+str(prob_falha_links)+'_teste_1.txt','w')
f.write(str(planilha_final_media_energia))
f.close()
f = open('Resultados_contador_tradicional_ndispositivos_'+str(n)+'t_distribuicao_'+str(tipo_distribuicao)+'t_validade_'+str(tempo_validade_c)+'prob_falha'+str(prob_falha_links)+'_teste_1.txt','w')
f.write(str(planilha_final_contador_envio))
f.close()
f = open('Resultados_RMSD_respostas_clientes_tradicional_'+str(n)+'t_distribuicao_'+str(tipo_distribuicao)+'t_validade_'+str(tempo_validade_c)+'prob_falha'+str(prob_falha_links)+'_teste_1.txt','w')
f.write(str(planilha_final_Media_RMSD))
f.close()
f = open('Resultados_cache_hits_tradicional_'+str(n)+'t_distribuicao_'+str(tipo_distribuicao)+'t_validade_'+str(tempo_validade_c)+'prob_falha'+str(prob_falha_links)+'_teste_1.txt','w')
f.write(str(planilha_final_cache_hits))
f.close()
f = open('Resultados_qtde_nos_fontes_tradicional_'+str(n)+'t_distribuicao_'+str(tipo_distribuicao)+'t_validade_'+str(tempo_validade_c)+'prob_falha'+str(prob_falha_links)+'_teste_1.txt','w')
f.write(str(planilha_final_qtde_nos_fontes))
f.close()
