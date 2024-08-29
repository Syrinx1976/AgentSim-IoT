# -*- coding: cp1252 -*-
# Este arquivo contém os parâmetros de entrada das simulações
import os
import time

numero_rodadas = 10
tamanho_inicial_do_MA = 1024 # tamanho inicial do Mobile Agent em bytes
tempo_validade_cache = [30,60,90,120,180] # Lista com Tempos de validade do cache
numero_dispositivos = [100] # Quantidade de dispositivos distribuídos na AoI
distribuicao = [1,2] # Distribuição de probabilidade para resquisições entre os conteúdos distintos. 1 para distribuição Uniforme, 2 para distribuição Zipf
prob_falha_links = [0] # probabilidade de falha de comunicação em um enlace (Unidades percentuais)
# dimensoes da AoI (Area of Interest)
largura_top = 50
comprimento_top = 50
coord_gateway_x = 50
coord_gateway_y = 50
modo_debug = 'on' # liga o modo Debug. Colocar 'on' ou 'off'

for n_dispositivos in numero_dispositivos:
   for t_distribuicao in distribuicao:
       for tempo in tempo_validade_cache:
           for prob in prob_falha_links:
               os.system('python tradicional/main.py '+str(n_dispositivos)+' '+str(t_distribuicao)+' '+str(tempo)+' '+str(prob)+' '+str(largura_top)+' '+str(comprimento_top)+' '+str(tamanho_inicial_do_MA)+' '+modo_debug+' '+str(coord_gateway_x)+' '+str(coord_gateway_y)+' '+str(numero_rodadas))


for n_dispositivos in numero_dispositivos:
  for t_distribuicao in distribuicao:
      for tempo in tempo_validade_cache:
          for prob in prob_falha_links:
              os.system('python proposta/main.py '+str(n_dispositivos)+' '+str(t_distribuicao)+' '+str(tempo)+' '+str(prob)+' '+str(largura_top)+' '+str(comprimento_top)+' '+str(tamanho_inicial_do_MA)+' '+modo_debug+' '+str(coord_gateway_x)+' '+str(coord_gateway_y)+' '+str(numero_rodadas))
