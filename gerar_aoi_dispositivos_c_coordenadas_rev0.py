# -*- coding: cp1252 -*-
import random
random.seed()
from scipy.spatial import distance
import executar_selecao_dos_dispositivos
import gerar_grafo_bidirecional_coordenadas

class dispositivos:

    def __init__(self):
        self.nome = None
        self.coordenada_dispositivo = None
        self.conteudo_sensor_1 = None
        self.conteudo_sensor_2 = None
        self.conteudo_sensor_3 = None
        self.conteudo_sensor_4 = None
        self.conteudo_sensor_5 = None
        self.sensing_range_conteudo_1 = None
        self.sensing_range_conteudo_2 = None
        self.sensing_range_conteudo_3 = None
        self.sensing_range_conteudo_4 = None
        self.sensing_range_conteudo_5 = None
        self.communication_range = None
        self.tamanho_conteudo_1 = None
        self.tamanho_conteudo_2 = None
        self.tamanho_conteudo_3 = None
        self.tamanho_conteudo_4 = None
        self.tamanho_conteudo_5 = None
        self.prob_falha_comunicacao_dispositivo = None
        self.energia = None

    # Função que cria uma instância do dispositivo
    def cria_dispositivos(self,nome,coordenada_dispositivo):
        # nome do dispositivo
        self.nome = nome

        # coordenada do dispositivo na AoI
        self.coordenada_dispositivo = coordenada_dispositivo

        # conteúdos disponíveis no dispositivo
        self.conteudo_sensor_1 = 'k1'
        self.conteudo_sensor_2 = 'k2'
        self.conteudo_sensor_3 = 'k3'
        self.conteudo_sensor_4 = 'k4'
        self.conteudo_sensor_5 = 'k5'

        # sensing range para cada sensor_correspondente a cada conteúdo
        self.sensing_range_conteudo_1 = 12.55
        self.sensing_range_conteudo_2 = 12.55
        self.sensing_range_conteudo_3 = 12.55
        self.sensing_range_conteudo_4 = 12.55
        self.sensing_range_conteudo_5 = 12.55

        # alcance de comunicação do rádio do dispositivo
        self.communication_range = 15
        self.tamanho_conteudo_1 = 20
        self.tamanho_conteudo_2 = 20
        self.tamanho_conteudo_3 = 20
        self.tamanho_conteudo_4 = 20
        self.tamanho_conteudo_5 = 20

        # Valor numérico do conteúdo
        self.valor_conteudo_1 = None
        self.valor_conteudo_2 = None
        self.valor_conteudo_3 = None
        self.valor_conteudo_4 = None
        self.valor_conteudo_5 = None

        # probabilidade de falha de comunicação do rádio do dispositivo
        self.prob_falha_comunicacao_dispositivo = float(random.randrange(0,30,1))/1000

        # Energia inicial do dispositivo em Joules
        self.energia = 20

# Classe Pontos: Possui módulos que inicializam os pontos da AoI com seus atributos
class pontos:
    def __init__(self):
        self.nome = None
        self.coordenadas_pontos = None

    def criar_pontos(self,nome,coordenadas_pontos):
        self.nome = nome
        self.coordenadas_pontos = coordenadas_pontos

# Módulo que gera as instâncias dos dispositivo
def gera_aoi_dispositivos(plt, largura, comprimento, n, m, prob_falha_links):

    Lista_dispositivos = [j for j in range(0,n,1)]
    Dispositivos = []
    Dic_coordenadas_dispositivos = {}
    for i in Lista_dispositivos:
        Dic_coordenadas_dispositivos.update({'S'+str(i):None})

    # Adicionando o Gateway na lista de dispositivos
    Dic_coordenadas_dispositivos.update({'GW':None})

    for i in Dic_coordenadas_dispositivos:
        dispositivo = dispositivos()
        dispositivo.cria_dispositivos(i, Dic_coordenadas_dispositivos[i])
        Dispositivos.append(dispositivo)

    # Dicionário com os valores numéricos dos dispositivos
    Dic_valores_conteudos_dispositivos = {'S160': [7.0, 17.0, 11.0, 8.0, 14.0], 'S229': [3.0, 15.0, 2.0, 15.0, 2.0], 'S228': [18.0, 24.0, 13.0, 12.0, 1.0], 'S225': [29.0, 7.0, 13.0, 14.0, 9.0], 'S224': [6.0, 15.0, 13.0, 16.0, 17.0], 'S227': [2.0, 1.0, 20.0, 29.0, 25.0], 'S226': [10.0, 18.0, 25.0, 3.0, 23.0], 'S221': [19.0, 18.0, 5.0, 12.0, 16.0], 'S220': [21.0, 8.0, 4.0, 13.0, 21.0], 'S223': [14.0, 1.0, 3.0, 27.0, 8.0], 'S222': [26.0, 14.0, 17.0, 7.0, 0.0], 'S57': [5.0, 27.0, 27.0, 9.0, 15.0], 'S56': [7.0, 10.0, 9.0, 25.0, 18.0], 'S55': [24.0, 2.0, 0.0, 1.0, 4.0], 'S54': [28.0, 0.0, 11.0, 20.0, 20.0], 'S53': [4.0, 13.0, 9.0, 29.0, 11.0], 'S52': [10.0, 6.0, 20.0, 3.0, 13.0], 'S51': [16.0, 24.0, 3.0, 11.0, 1.0], 'S50': [8.0, 3.0, 10.0, 15.0, 5.0], 'S59': [13.0, 9.0, 21.0, 8.0, 26.0], 'S58': [12.0, 8.0, 14.0, 5.0, 29.0], 'S83': [13.0, 0.0, 19.0, 7.0, 1.0], 'S118': [27.0, 16.0, 11.0, 21.0, 16.0], 'S258': [28.0, 6.0, 11.0, 8.0, 10.0], 'S259': [8.0, 13.0, 5.0, 0.0, 4.0], 'S125': [26.0, 4.0, 23.0, 7.0, 16.0], 'S250': [20.0, 10.0, 27.0, 7.0, 10.0], 'S251': [16.0, 19.0, 29.0, 10.0, 29.0], 'S252': [11.0, 13.0, 7.0, 15.0, 25.0], 'S253': [20.0, 0.0, 18.0, 27.0, 16.0], 'S254': [14.0, 13.0, 22.0, 10.0, 8.0], 'S255': [10.0, 8.0, 16.0, 7.0, 1.0], 'S256': [17.0, 5.0, 20.0, 1.0, 17.0], 'S257': [8.0, 4.0, 29.0, 19.0, 3.0], 'S44': [14.0, 25.0, 16.0, 2.0, 3.0], 'S45': [25.0, 7.0, 3.0, 22.0, 5.0], 'S46': [19.0, 29.0, 5.0, 0.0, 22.0], 'S47': [27.0, 20.0, 6.0, 19.0, 3.0], 'S40': [15.0, 16.0, 0.0, 11.0, 7.0], 'S41': [13.0, 21.0, 10.0, 19.0, 7.0], 'S42': [19.0, 7.0, 21.0, 29.0, 12.0], 'S43': [19.0, 27.0, 12.0, 19.0, 15.0], 'S48': [28.0, 25.0, 25.0, 10.0, 16.0], 'S49': [8.0, 21.0, 14.0, 0.0, 10.0], 'S297': [23.0, 15.0, 17.0, 24.0, 9.0], 'S109': [7.0, 6.0, 5.0, 9.0, 13.0], 'S135': [25.0, 15.0, 0.0, 18.0, 5.0], 'S134': [25.0, 21.0, 6.0, 19.0, 8.0], 'S137': [11.0, 0.0, 22.0, 7.0, 16.0], 'S136': [1.0, 22.0, 12.0, 14.0, 2.0], 'S131': [16.0, 21.0, 26.0, 0.0, 21.0], 'S130': [24.0, 18.0, 29.0, 25.0, 8.0], 'S133': [16.0, 19.0, 11.0, 17.0, 8.0], 'S132': [0.0, 21.0, 9.0, 1.0, 9.0], 'S91': [11.0, 13.0, 2.0, 8.0, 6.0], 'S139': [24.0, 1.0, 14.0, 10.0, 23.0], 'S138': [9.0, 13.0, 28.0, 6.0, 3.0], 'S249': [6.0, 11.0, 0.0, 14.0, 28.0], 'S248': [19.0, 1.0, 4.0, 27.0, 16.0], 'S243': [22.0, 23.0, 21.0, 23.0, 13.0], 'S242': [11.0, 25.0, 5.0, 13.0, 7.0], 'S241': [23.0, 21.0, 28.0, 8.0, 0.0], 'S240': [0.0, 7.0, 15.0, 13.0, 17.0], 'S247': [20.0, 3.0, 14.0, 2.0, 19.0], 'S246': [20.0, 6.0, 24.0, 21.0, 19.0], 'S245': [7.0, 12.0, 20.0, 9.0, 26.0], 'S244': [23.0, 2.0, 19.0, 11.0, 9.0], 'S79': [17.0, 9.0, 12.0, 2.0, 8.0], 'S78': [4.0, 29.0, 3.0, 26.0, 24.0], 'S231': [12.0, 13.0, 20.0, 2.0, 20.0], 'S71': [18.0, 10.0, 4.0, 19.0, 19.0], 'S70': [11.0, 18.0, 13.0, 22.0, 22.0], 'S73': [20.0, 14.0, 23.0, 1.0, 17.0], 'S72': [29.0, 6.0, 6.0, 11.0, 12.0], 'S75': [10.0, 22.0, 27.0, 3.0, 8.0], 'S74': [15.0, 20.0, 20.0, 7.0, 28.0], 'S77': [28.0, 13.0, 9.0, 19.0, 2.0], 'S76': [27.0, 29.0, 7.0, 13.0, 3.0], 'S179': [2.0, 29.0, 6.0, 27.0, 9.0], 'S128': [4.0, 22.0, 7.0, 19.0, 29.0], 'S178': [25.0, 0.0, 18.0, 4.0, 3.0], 'S114': [12.0, 10.0, 26.0, 6.0, 24.0], 'S82': [10.0, 18.0, 13.0, 13.0, 26.0], 'S140': [23.0, 26.0, 14.0, 9.0, 7.0], 'S141': [15.0, 29.0, 28.0, 21.0, 19.0], 'S142': [2.0, 26.0, 24.0, 4.0, 17.0], 'S143': [2.0, 2.0, 15.0, 20.0, 21.0], 'S144': [29.0, 0.0, 0.0, 10.0, 12.0], 'S145': [24.0, 26.0, 8.0, 26.0, 3.0], 'S146': [2.0, 24.0, 17.0, 8.0, 3.0], 'S147': [11.0, 23.0, 10.0, 13.0, 21.0], 'S148': [17.0, 28.0, 15.0, 25.0, 5.0], 'S149': [17.0, 19.0, 9.0, 20.0, 8.0], 'S120': [13.0, 4.0, 4.0, 13.0, 5.0], 'S85': [6.0, 4.0, 29.0, 4.0, 24.0], 'S88': [2.0, 19.0, 14.0, 22.0, 1.0], 'S121': [26.0, 26.0, 25.0, 2.0, 1.0], 'S86': [20.0, 27.0, 25.0, 17.0, 3.0], 'S87': [22.0, 10.0, 15.0, 23.0, 19.0], 'S269': [20.0, 27.0, 23.0, 16.0, 28.0], 'S68': [23.0, 29.0, 3.0, 25.0, 21.0], 'S69': [1.0, 3.0, 21.0, 22.0, 23.0], 'S66': [1.0, 13.0, 9.0, 4.0, 0.0], 'S67': [12.0, 12.0, 19.0, 19.0, 19.0], 'S64': [15.0, 3.0, 10.0, 29.0, 4.0], 'S65': [15.0, 11.0, 28.0, 16.0, 27.0], 'S62': [23.0, 14.0, 1.0, 22.0, 2.0], 'S63': [5.0, 2.0, 7.0, 24.0, 22.0], 'S60': [3.0, 28.0, 26.0, 25.0, 17.0], 'S61': [9.0, 20.0, 7.0, 11.0, 8.0], 'S89': [6.0, 14.0, 14.0, 24.0, 10.0], 'S172': [14.0, 28.0, 16.0, 22.0, 25.0], 'S23': [28.0, 14.0, 15.0, 5.0, 28.0], 'S153': [22.0, 16.0, 9.0, 22.0, 3.0], 'S152': [8.0, 25.0, 0.0, 17.0, 23.0], 'S151': [19.0, 5.0, 20.0, 13.0, 12.0], 'S150': [7.0, 20.0, 1.0, 24.0, 13.0], 'S157': [7.0, 4.0, 0.0, 25.0, 4.0], 'S156': [15.0, 23.0, 23.0, 6.0, 7.0], 'S155': [23.0, 15.0, 13.0, 16.0, 13.0], 'S154': [6.0, 20.0, 28.0, 5.0, 0.0], 'S159': [16.0, 26.0, 0.0, 28.0, 8.0], 'S158': [2.0, 18.0, 23.0, 8.0, 18.0], 'S9': [28.0, 15.0, 1.0, 21.0, 3.0], 'S8': [10.0, 7.0, 13.0, 21.0, 3.0], 'S111': [23.0, 6.0, 12.0, 26.0, 22.0], 'S206': [15.0, 12.0, 0.0, 17.0, 16.0], 'S3': [1.0, 8.0, 23.0, 16.0, 11.0], 'S2': [22.0, 16.0, 29.0, 29.0, 27.0], 'S1': [23.0, 0.0, 19.0, 27.0, 20.0], 'S0': [25.0, 6.0, 11.0, 28.0, 28.0], 'S7': [5.0, 14.0, 11.0, 16.0, 8.0], 'S6': [7.0, 4.0, 7.0, 10.0, 23.0], 'S5': [7.0, 14.0, 1.0, 27.0, 7.0], 'S4': [15.0, 0.0, 4.0, 22.0, 3.0], 'S168': [28.0, 29.0, 6.0, 26.0, 5.0], 'S107': [6.0, 14.0, 5.0, 28.0, 28.0], 'S276': [22.0, 21.0, 2.0, 26.0, 16.0], 'S19': [18.0, 17.0, 9.0, 1.0, 4.0], 'S18': [1.0, 18.0, 27.0, 9.0, 24.0], 'S119': [6.0, 17.0, 18.0, 4.0, 12.0], 'S92': [11.0, 22.0, 11.0, 20.0, 16.0], 'S13': [14.0, 3.0, 23.0, 25.0, 21.0], 'S12': [10.0, 13.0, 18.0, 18.0, 28.0], 'S11': [9.0, 18.0, 0.0, 28.0, 7.0], 'S10': [7.0, 9.0, 4.0, 19.0, 27.0], 'S17': [9.0, 27.0, 4.0, 15.0, 27.0], 'S16': [24.0, 23.0, 9.0, 22.0, 0.0], 'S15': [21.0, 23.0, 17.0, 14.0, 22.0], 'S14': [3.0, 7.0, 17.0, 12.0, 19.0], 'S93': [9.0, 13.0, 5.0, 11.0, 0.0], 'S277': [14.0, 15.0, 27.0, 15.0, 7.0], 'S274': [4.0, 12.0, 0.0, 28.0, 17.0], 'S90': [27.0, 18.0, 5.0, 16.0, 16.0], 'S272': [19.0, 9.0, 9.0, 9.0, 18.0], 'S96': [13.0, 13.0, 4.0, 14.0, 16.0], 'S270': [17.0, 22.0, 16.0, 9.0, 19.0], 'S94': [17.0, 2.0, 6.0, 27.0, 9.0], 'S166': [18.0, 9.0, 6.0, 15.0, 19.0], 'S97': [9.0, 16.0, 6.0, 24.0, 2.0], 'S99': [18.0, 7.0, 10.0, 21.0, 28.0], 'S98': [25.0, 28.0, 12.0, 17.0, 29.0], 'S162': [6.0, 26.0, 26.0, 6.0, 9.0], 'S163': [1.0, 5.0, 28.0, 14.0, 19.0], 'S278': [5.0, 13.0, 22.0, 2.0, 17.0], 'S161': [4.0, 1.0, 10.0, 8.0, 13.0], 'S214': [8.0, 18.0, 18.0, 19.0, 21.0], 'S215': [16.0, 26.0, 0.0, 8.0, 15.0], 'S216': [25.0, 23.0, 7.0, 25.0, 17.0], 'S217': [5.0, 27.0, 1.0, 23.0, 21.0], 'S108': [11.0, 23.0, 20.0, 5.0, 16.0], 'S95': [19.0, 14.0, 17.0, 14.0, 4.0], 'S212': [18.0, 6.0, 4.0, 26.0, 7.0], 'S213': [11.0, 22.0, 19.0, 17.0, 28.0], 'S104': [16.0, 5.0, 17.0, 2.0, 8.0], 'S105': [9.0, 12.0, 18.0, 0.0, 2.0], 'S106': [2.0, 20.0, 20.0, 16.0, 12.0], 'S169': [15.0, 26.0, 6.0, 16.0, 18.0], 'S100': [3.0, 28.0, 22.0, 21.0, 18.0], 'S101': [2.0, 7.0, 3.0, 9.0, 16.0], 'S102': [9.0, 17.0, 27.0, 26.0, 7.0], 'S103': [1.0, 8.0, 11.0, 15.0, 27.0], 'S184': [1.0, 28.0, 17.0, 1.0, 29.0], 'S185': [13.0, 18.0, 19.0, 19.0, 23.0], 'S186': [0.0, 17.0, 5.0, 9.0, 5.0], 'S187': [25.0, 11.0, 0.0, 29.0, 17.0], 'S180': [4.0, 24.0, 25.0, 26.0, 11.0], 'S181': [7.0, 13.0, 10.0, 24.0, 7.0], 'S182': [28.0, 29.0, 1.0, 18.0, 12.0], 'S167': [14.0, 8.0, 25.0, 18.0, 23.0], 'S110': [18.0, 20.0, 2.0, 19.0, 21.0], 'S188': [17.0, 18.0, 4.0, 18.0, 21.0], 'S189': [3.0, 3.0, 0.0, 28.0, 26.0], 'S165': [20.0, 19.0, 23.0, 1.0, 21.0], 'S273': [26.0, 11.0, 11.0, 6.0, 4.0], 'S275': [16.0, 4.0, 26.0, 1.0, 0.0], 'S268': [25.0, 17.0, 28.0, 15.0, 13.0], 'S261': [20.0, 8.0, 6.0, 4.0, 26.0], 'S260': [9.0, 15.0, 14.0, 21.0, 22.0], 'S263': [1.0, 25.0, 1.0, 4.0, 22.0], 'S262': [8.0, 14.0, 24.0, 10.0, 12.0], 'S265': [28.0, 23.0, 1.0, 7.0, 8.0], 'S264': [28.0, 24.0, 10.0, 0.0, 26.0], 'S267': [28.0, 3.0, 15.0, 7.0, 13.0], 'S266': [17.0, 22.0, 1.0, 5.0, 4.0], 'S171': [5.0, 20.0, 15.0, 21.0, 0.0], 'S170': [9.0, 5.0, 7.0, 28.0, 16.0], 'S173': [11.0, 28.0, 3.0, 6.0, 15.0], 'S279': [18.0, 22.0, 16.0, 25.0, 28.0], 'S175': [27.0, 0.0, 26.0, 10.0, 3.0], 'S174': [8.0, 15.0, 24.0, 10.0, 19.0], 'S177': [10.0, 1.0, 22.0, 19.0, 0.0], 'S176': [24.0, 5.0, 11.0, 6.0, 13.0], 'S207': [0.0, 1.0, 11.0, 28.0, 4.0], 'GW': [27.0, 11.0, 24.0, 28.0, 9.0], 'S205': [28.0, 26.0, 4.0, 2.0, 14.0], 'S204': [17.0, 2.0, 20.0, 21.0, 8.0], 'S203': [22.0, 17.0, 8.0, 1.0, 10.0], 'S202': [20.0, 6.0, 26.0, 8.0, 29.0], 'S201': [22.0, 23.0, 28.0, 5.0, 22.0], 'S200': [6.0, 12.0, 1.0, 25.0, 25.0], 'S117': [0.0, 8.0, 27.0, 3.0, 4.0], 'S116': [28.0, 16.0, 4.0, 4.0, 16.0], 'S115': [17.0, 3.0, 5.0, 17.0, 0.0], 'S164': [3.0, 27.0, 0.0, 1.0, 26.0], 'S113': [3.0, 10.0, 26.0, 26.0, 13.0], 'S112': [7.0, 6.0, 4.0, 6.0, 24.0], 'S209': [22.0, 16.0, 26.0, 28.0, 1.0], 'S208': [13.0, 22.0, 23.0, 0.0, 6.0], 'S197': [14.0, 11.0, 28.0, 16.0, 22.0], 'S196': [2.0, 25.0, 12.0, 3.0, 22.0], 'S195': [16.0, 8.0, 17.0, 9.0, 27.0], 'S194': [2.0, 28.0, 27.0, 7.0, 12.0], 'S193': [13.0, 15.0, 4.0, 5.0, 19.0], 'S192': [22.0, 15.0, 7.0, 27.0, 29.0], 'S191': [16.0, 15.0, 20.0, 1.0, 20.0], 'S190': [14.0, 22.0, 10.0, 2.0, 28.0], 'S210': [22.0, 6.0, 28.0, 11.0, 4.0], 'S199': [13.0, 29.0, 20.0, 12.0, 15.0], 'S211': [19.0, 11.0, 12.0, 25.0, 15.0], 'S271': [24.0, 23.0, 28.0, 11.0, 9.0], 'S39': [17.0, 24.0, 27.0, 18.0, 16.0], 'S38': [22.0, 13.0, 10.0, 17.0, 14.0], 'S35': [2.0, 0.0, 16.0, 5.0, 4.0], 'S34': [21.0, 7.0, 25.0, 15.0, 19.0], 'S37': [7.0, 29.0, 24.0, 0.0, 5.0], 'S36': [7.0, 24.0, 21.0, 28.0, 26.0], 'S31': [6.0, 7.0, 3.0, 0.0, 0.0], 'S30': [24.0, 17.0, 18.0, 15.0, 10.0], 'S33': [20.0, 27.0, 5.0, 4.0, 14.0], 'S32': [29.0, 20.0, 23.0, 10.0, 23.0], 'S298': [19.0, 15.0, 8.0, 12.0, 25.0], 'S299': [29.0, 27.0, 23.0, 27.0, 4.0], 'S294': [12.0, 20.0, 13.0, 16.0, 15.0], 'S295': [20.0, 27.0, 7.0, 7.0, 17.0], 'S296': [18.0, 6.0, 27.0, 7.0, 21.0], 'S198': [11.0, 18.0, 4.0, 21.0, 1.0], 'S290': [28.0, 12.0, 3.0, 10.0, 26.0], 'S291': [26.0, 21.0, 26.0, 7.0, 17.0], 'S292': [28.0, 2.0, 7.0, 26.0, 13.0], 'S293': [20.0, 18.0, 5.0, 9.0, 18.0], 'S232': [11.0, 18.0, 20.0, 23.0, 5.0], 'S233': [23.0, 10.0, 10.0, 17.0, 9.0], 'S230': [4.0, 18.0, 26.0, 25.0, 4.0], 'S129': [4.0, 16.0, 27.0, 6.0, 8.0], 'S236': [8.0, 22.0, 26.0, 20.0, 6.0], 'S237': [28.0, 18.0, 25.0, 0.0, 12.0], 'S234': [24.0, 13.0, 18.0, 24.0, 2.0], 'S235': [0.0, 27.0, 2.0, 23.0, 13.0], 'S122': [9.0, 16.0, 22.0, 25.0, 12.0], 'S123': [8.0, 19.0, 25.0, 0.0, 10.0], 'S238': [26.0, 3.0, 5.0, 19.0, 26.0], 'S239': [3.0, 14.0, 21.0, 9.0, 24.0], 'S126': [14.0, 6.0, 1.0, 27.0, 25.0], 'S127': [11.0, 2.0, 13.0, 19.0, 25.0], 'S124': [29.0, 2.0, 6.0, 9.0, 15.0], 'S219': [4.0, 4.0, 1.0, 3.0, 7.0], 'S22': [10.0, 17.0, 27.0, 20.0, 25.0], 'S80': [1.0, 24.0, 3.0, 15.0, 21.0], 'S20': [15.0, 10.0, 0.0, 21.0, 4.0], 'S21': [0.0, 26.0, 7.0, 1.0, 17.0], 'S26': [12.0, 15.0, 22.0, 3.0, 10.0], 'S27': [14.0, 21.0, 4.0, 26.0, 9.0], 'S24': [15.0, 14.0, 4.0, 12.0, 15.0], 'S25': [28.0, 13.0, 6.0, 28.0, 7.0], 'S28': [14.0, 20.0, 25.0, 6.0, 4.0], 'S29': [19.0, 15.0, 10.0, 2.0, 0.0], 'S218': [11.0, 1.0, 25.0, 20.0, 25.0], 'S81': [22.0, 16.0, 10.0, 13.0, 17.0], 'S183': [6.0, 11.0, 27.0, 14.0, 25.0], 'S84': [1.0, 0.0, 13.0, 19.0, 15.0], 'S289': [21.0, 28.0, 21.0, 5.0, 28.0], 'S288': [16.0, 25.0, 20.0, 27.0, 28.0], 'S287': [14.0, 4.0, 1.0, 28.0, 5.0], 'S286': [10.0, 28.0, 2.0, 29.0, 25.0], 'S285': [29.0, 23.0, 7.0, 3.0, 18.0], 'S284': [23.0, 9.0, 29.0, 4.0, 3.0], 'S283': [10.0, 26.0, 12.0, 9.0, 22.0], 'S282': [7.0, 8.0, 26.0, 25.0, 1.0], 'S281': [6.0, 27.0, 9.0, 11.0, 25.0], 'S280': [0.0, 4.0, 19.0, 6.0, 19.0]}


    for dispositivo in Dic_valores_conteudos_dispositivos:
        for nome in Dispositivos:
            if nome.nome == dispositivo:
                nome.valor_conteudo_1 = Dic_valores_conteudos_dispositivos[dispositivo][0]
                nome.valor_conteudo_2 = Dic_valores_conteudos_dispositivos[dispositivo][1]
                nome.valor_conteudo_3 = Dic_valores_conteudos_dispositivos[dispositivo][2]
                nome.valor_conteudo_4 = Dic_valores_conteudos_dispositivos[dispositivo][3]
                nome.valor_conteudo_5 = Dic_valores_conteudos_dispositivos[dispositivo][4]

    Dic_coordenadas_dispositivos, G, Dic_registro_custos_originais = gerar_grafo_bidirecional_coordenadas.gera_grafo(n,m,plt,Dic_coordenadas_dispositivos,largura,comprimento,Dispositivos,prob_falha_links)

    # Criação da AoI: Uma Matriz com pontos equidistantes
    First_point = (0,0)
    Matrix = [[First_point for i in range (0,largura,1)] for i in range (0,comprimento,1)]

    for m in range(0,largura,1):
        for n in range(0,comprimento,1):
            Matrix[m][n] = (m,n)

    numero = 0
    coordenadas_pontos = []
    for i in Matrix:
        for j in i:
            numero += 1
            coordenadas_pontos.append(j)

    # Criaçao das instâncias dos pontos
    index = 0
    Lista_pontos = []

    for i in range(0,numero,1):
        nome = 'P'+str(i)
        ponto = pontos()
        ponto.criar_pontos(nome,coordenadas_pontos[index])
        index += 1
        Lista_pontos.append(ponto)


    Lista_dispositivos = []

    for i in Dic_coordenadas_dispositivos:
        if i not in Lista_dispositivos and i != 'GW':
            Lista_dispositivos.append(i)


    return Dispositivos, Lista_pontos, G, Dic_registro_custos_originais
