# -*- coding: cp1252 -*-
import sys
import random
random.seed()
from networkx import nx
from scipy.spatial import distance

Dic_registro_custos_originais = {}

def gera_grafo(n,m,plt,Dic_coordenadas_dispositivos,largura,comprimento,Dispositivos,prob_falha_links):

    G = nx.Graph()


    G.add_nodes_from([i.nome for i in Dispositivos])


# Para AoI 100 x 100

    if n == 180:
        Dic_coordenadas_dispositivos = {'S57': (41.24934626209886, 4.531536785242185), 'S56': (85.10191425046332, 75.04593306428688), 'S55': (11.566999328071148, 45.943351896323136), 'S54': (21.483856048043226, 52.59968457013153), 'S53': (66.89076001568739, 58.38821851521423), 'S52': (71.92217171946241, 96.10820790763412), 'S51': (37.088227034870336, 71.58963976716872), 'S50': (69.61878702466713, 29.86521682899036), 'S59': (20.171067607749038, 31.517885984675264), 'S58': (1.2448640014288137, 12.49753836409192), 'GW': (96.80513566985174, 89.03787327137519), 'S44': (24.336672351224465, 71.5085426805079), 'S45': (38.36620305376116, 82.83110437797326), 'S46': (44.98473641087345, 35.53963319022137), 'S47': (52.08193032051226, 70.41510277626071), 'S40': (74.06387010694642, 87.41573038129096), 'S41': (27.92512476667254, 33.03026697005572), 'S42': (16.537935000657356, 51.5268608885171), 'S43': (5.37404411525666, 13.193106733549143), 'S48': (25.18438911816643, 63.676373182067394), 'S49': (77.05472977601673, 41.07092722349647), 'S135': (4.9107982634443275, 23.748014296400054), 'S134': (38.86711577191667, 48.29722058940782), 'S137': (59.808680741798014, 84.28413462953232), 'S136': (44.24054899362619, 28.718155197744554), 'S131': (94.34843757711216, 61.55853688005111), 'S130': (46.99329446233366, 77.82508720480743), 'S133': (65.82467134927002, 19.62719376743699), 'S132': (86.46994057003799, 50.27203275949345), 'S139': (31.19884857194659, 75.91814325669655), 'S138': (30.075211273945268, 87.75085525502621), 'S79': (92.33356525527212, 28.12776834254852), 'S78': (37.97570759928538, 7.231314084846307), 'S129': (85.45717217741313, 24.96460436599356), 'S71': (3.03595461278533, 58.698180378346706), 'S70': (40.42028685872051, 5.171273594476123), 'S73': (19.139288431377956, 94.7270709192982), 'S72': (17.907268969598555, 58.46981968751118), 'S75': (68.45167103005797, 45.46928906714063), 'S74': (72.71246991746966, 19.11205604732181), 'S77': (33.33527169686421, 22.237958563509828), 'S76': (8.037402572793418, 0.9050543139843326), 'S179': (86.68743996725264, 55.19252084795371), 'S178': (76.28723120851963, 49.76703334583888), 'S140': (59.3108293496878, 32.162226568286826), 'S141': (46.18957115499745, 67.01288570652592), 'S142': (63.57754041901723, 65.46235849692003), 'S143': (36.849082241819865, 15.77829186856431), 'S144': (36.81250652751932, 48.81742893032327), 'S145': (75.10440789516917, 57.85695362312552), 'S146': (65.9834631495319, 39.18971108792625), 'S147': (59.236763957275464, 92.01600298785915), 'S148': (9.725251992617622, 16.336017483709988), 'S149': (29.766315129581823, 20.590508487454173), 'S171': (35.711656017373485, 72.65191037522848), 'S68': (50.78283116028184, 62.33521111271686), 'S69': (67.16276815110203, 99.17806906933781), 'S66': (8.030491743152524, 92.53963674279215), 'S67': (67.6896622852019, 38.599242384611124), 'S64': (30.56547060858923, 39.182249308081595), 'S65': (19.541537933363596, 60.34990518981089), 'S62': (79.49893652863933, 72.69949868093398), 'S63': (94.78429576292648, 56.28207938009322), 'S60': (90.02746142505791, 58.52678185437714), 'S61': (17.16132976782606, 36.984008760927665), 'S153': (11.896059784989754, 85.02035905178555), 'S152': (90.77800671667188, 29.79850971534166), 'S151': (70.43379253634453, 31.509810895555248), 'S150': (48.07605734591511, 12.714088915507771), 'S157': (92.59563859294565, 4.711032636372414), 'S156': (58.62393564926415, 2.2170505324910716), 'S155': (39.37940339388204, 33.887595453587316), 'S154': (4.256509367466766, 96.18640927131464), 'S159': (82.78790924932537, 50.217846318080305), 'S158': (68.11888518029417, 86.54407901268331), 'S170': (67.10386293694947, 25.1042964476471), 'S9': (52.76995557927728, 7.990119844832034), 'S8': (66.6239955161654, 16.170168921119043), 'S3': (97.29653621133633, 90.47061184036556), 'S2': (44.5757377007598, 90.70929975803878), 'S1': (90.89134886878404, 6.276235833215993), 'S0': (34.19048255331629, 86.47241952059376), 'S7': (80.98484725217254, 22.264677649826325), 'S6': (81.25372215484649, 98.54665404424937), 'S5': (26.789116779087752, 31.93411303941167), 'S4': (47.685270712848734, 16.031703992841216), 'S168': (81.52134937790835, 16.56747896047466), 'S19': (44.628473499923594, 51.541974756868335), 'S18': (68.79296658613285, 50.393328069943), 'S13': (6.225064440201478, 49.52075210829946), 'S12': (60.41494635686765, 27.254072531079178), 'S11': (24.181468405802352, 96.55493195791205), 'S10': (27.945690146708102, 3.4590112501207138), 'S17': (43.18253958564709, 83.41043534794505), 'S16': (60.33351984852714, 59.24291036199296), 'S15': (81.72116352030946, 34.3428563401461), 'S14': (47.126661603103706, 84.2190616050593), 'S93': (76.44846405549991, 66.73301801232806), 'S92': (50.62581898447371, 9.842807953537058), 'S91': (79.34558718945435, 11.346439646185491), 'S90': (75.18652685994628, 7.467031515758526), 'S97': (66.2098837339833, 61.02319569410661), 'S96': (5.170307150304998, 95.8233389460556), 'S95': (6.535224679071938, 50.37661662908764), 'S94': (45.07814362188455, 24.704059737992146), 'S166': (94.87374630124395, 62.4673005716416), 'S167': (86.35949529739547, 16.86683026015785), 'S99': (5.066261966971464, 20.56481430343313), 'S98': (49.21168175131487, 31.443982268157423), 'S162': (34.291511391549655, 33.240617887392546), 'S163': (61.513667493359236, 62.572562870905735), 'S160': (77.05451044112881, 64.13600468333162), 'S161': (90.6688834464057, 82.21978680966153), 'S108': (86.05712502185929, 63.05549484489962), 'S109': (78.44140166465431, 5.929914445502716), 'S104': (55.52954702288733, 86.80593785751603), 'S105': (61.86469725389391, 19.89053072131146), 'S106': (30.229772094569284, 4.671786083259377), 'S169': (8.185333521809602, 91.98524452507688), 'S100': (21.208074310806357, 98.99265160288223), 'S101': (4.309467205415373, 30.391142822803786), 'S102': (17.834174509060873, 37.13822264818193), 'S103': (54.540253210746606, 3.3486763119679908), 'S164': (86.03242437948035, 84.3200809171017), 'S165': (39.61114735980381, 32.37024992495539), 'S80': (98.8414917802042, 38.94474002628303), 'S81': (3.142282359651438, 38.14842211842332), 'S82': (91.65531274808136, 94.99928308162416), 'S83': (25.935272389738607, 21.463732258041823), 'S84': (13.546140873666479, 70.18615452072974), 'S85': (82.5800176230533, 71.75022628122086), 'S86': (25.07030329802078, 18.715822607584787), 'S87': (75.93304022535469, 13.999033680356154), 'S88': (80.22088737211132, 13.846771261094414), 'S89': (51.613212365131275, 59.51410079823114), 'S173': (78.35702444428823, 28.1701965767328), 'S172': (2.7552274440215463, 68.88984713046378), 'S175': (52.288727621267675, 55.4072386849096), 'S174': (85.34572377633786, 63.948890545212635), 'S177': (28.799452060376506, 98.63516420675481), 'S176': (52.29961164562128, 92.10259359967326), 'S119': (99.10522887566566, 84.86702197703565), 'S118': (66.35210052735174, 29.6575456532133), 'S117': (51.64020868001391, 57.59544674186587), 'S116': (84.49510244580482, 46.18533852386714), 'S115': (16.650421188592823, 69.19049525176794), 'S114': (70.70643679377878, 7.949710320677539), 'S113': (76.92677506373035, 85.64904949638989), 'S112': (98.23013643617855, 27.628406740669476), 'S111': (71.76157458532552, 41.383986818380414), 'S110': (19.810461213392006, 77.50434804949401), 'S39': (21.30979286248077, 14.333556913928835), 'S38': (53.97942929492251, 11.230794182940274), 'S35': (97.36464928170658, 16.17215633025648), 'S34': (94.92664480852535, 22.41261895486355), 'S37': (42.91384677134623, 59.08168947492057), 'S36': (1.2098509000173396, 9.713998163938397), 'S31': (82.91729201962006, 16.047493543109614), 'S30': (2.3432638457923205, 10.33835549093327), 'S33': (92.13587667229926, 26.445330685154055), 'S32': (53.69027620684634, 46.4638972687222), 'S128': (40.635850399488845, 10.995595120797141), 'S107': (69.05682629588146, 42.128773772081686), 'S122': (1.3380809605669008, 18.255504082367334), 'S123': (63.854495902032895, 76.07298455067263), 'S120': (41.44709349184011, 61.18134042387774), 'S121': (96.17245697999635, 90.32967150844495), 'S126': (75.1315852242099, 1.9088927212302598), 'S127': (27.15667038785188, 63.859226595232855), 'S124': (59.30608451063225, 53.7742071103763), 'S125': (69.52810135254224, 51.37003494841612), 'S22': (2.92445880081027, 9.338790681827547), 'S23': (7.7632404648681375, 66.96613805641151), 'S20': (26.156547986136935, 51.88229677389676), 'S21': (37.658438298268734, 63.6111221368371), 'S26': (23.741857819107715, 23.65137195689804), 'S27': (14.1504269405319, 18.993223386071634), 'S24': (47.70882555606543, 31.404712822865534), 'S25': (93.73959831712611, 3.6356907445584286), 'S28': (98.11585336384015, 56.977982127471435), 'S29': (30.719837984436094, 88.20368557411639)}

    if n == 300:
        Dic_coordenadas_dispositivos = {'S229': (31.222946582705667, 2.236134303115944), 'S228': (57.68748726784286, 31.99506958230518), 'S225': (82.9385456523633, 10.084637537687435), 'S224': (57.96296443434552, 98.54246735999762), 'S227': (15.44252668870586, 28.74869926489175), 'S226': (14.698169007352469, 21.67302785920777), 'S221': (22.353954884822326, 79.11076073694929), 'S220': (60.2092159732192, 43.077038540519354), 'S223': (78.12488644577469, 92.31090010402261), 'S222': (33.02498431989157, 16.18876606218187), 'S57': (25.22298582922974, 30.37630733179514), 'S56': (65.38306282344637, 90.51663397061026), 'S55': (56.4214425095931, 47.449490391523554), 'S54': (9.5124116482725, 8.052415467140062), 'S53': (79.64896518390704, 78.55596526784629), 'S52': (39.9851864123018, 95.13853228376848), 'S51': (81.75981607616941, 56.269774816678996), 'S50': (74.84372322661031, 60.894762371630875), 'S270': (89.61874471912887, 94.1035395386369), 'S59': (12.669148173500599, 78.33868569802107), 'S58': (55.632584024681364, 55.89966817117913), 'GW': (27.23960464515538, 2.880521735944619), 'S189': (9.583422651812501, 68.6946717623172), 'S211': (62.18806107766215, 81.43727727384653), 'S258': (87.98233290172779, 56.828494555861354), 'S259': (18.489417714061197, 44.61448857458746), 'S250': (5.264297266517426, 23.991395450794872), 'S251': (1.0721438916987736, 1.7218964960285743), 'S252': (91.73864493022565, 42.26417783111861), 'S253': (88.1020712426154, 15.792250578671874), 'S254': (7.507530114826222, 88.00351849125867), 'S255': (30.280182807913324, 69.44233858421629), 'S256': (32.53997701843166, 42.551239111644556), 'S257': (16.14339184233402, 68.33977588640792), 'S44': (80.53726359630014, 51.5089390647515), 'S45': (80.59323071102055, 76.41414305009636), 'S46': (62.029293255805875, 54.168103561465685), 'S47': (86.81263060653846, 52.55436668391555), 'S40': (19.96637127211962, 79.7844871877596), 'S41': (56.53908670398452, 42.863574487940994), 'S42': (36.42914771881849, 66.28322553154253), 'S43': (28.08238816679045, 66.66382160944544), 'S48': (78.41047185599278, 38.24827675380704), 'S49': (95.25347784869463, 91.99693737107786), 'S261': (98.58916478981253, 32.882748456479085), 'S292': (10.742887717493211, 38.779217370477134), 'S135': (35.292679061276, 78.59601977015332), 'S134': (77.30035176406092, 9.036184525285995), 'S137': (70.62554007560563, 70.0479908307626), 'S136': (88.33193298073093, 46.4138197894588), 'S131': (95.63082477169384, 72.2780981211722), 'S130': (50.35898160053735, 97.76265546597907), 'S133': (0.556944372413426, 61.25779188265216), 'S132': (8.593061582856798, 50.96984331674827), 'S139': (62.47909973946597, 91.18177297245312), 'S138': (20.545348518363603, 51.698104553687166), 'S260': (61.8949667403699, 50.474776571741664), 'S249': (82.67685910956651, 52.48125166031939), 'S248': (57.98150130627935, 50.39987555292515), 'S243': (97.41838908976864, 68.43963070774836), 'S242': (12.583275170626962, 26.423781153927507), 'S241': (70.46565140853053, 52.093306393064395), 'S240': (32.73237800270225, 40.70303730644882), 'S247': (62.04962440629185, 21.585030235216262), 'S230': (25.876752979594574, 39.01955212327582), 'S245': (68.10764568055666, 86.39377073585295), 'S244': (12.75999425550738, 93.30054145266685), 'S79': (60.49049961069588, 20.1151683225816), 'S78': (81.28532201801859, 68.63711953232539), 'S129': (91.97672159941615, 80.16865327395794), 'S265': (62.14223765854785, 50.69893564901706), 'S71': (73.56854236443529, 9.953581496868546), 'S70': (73.13960623518827, 92.68454898462983), 'S73': (59.73871809975376, 31.489433479019567), 'S72': (94.3319726713675, 25.51906810979029), 'S75': (64.57259880973788, 92.77421416385087), 'S74': (18.580619624823036, 54.70933517957464), 'S77': (18.564487946212417, 28.09541807266115), 'S76': (49.458875710580095, 87.64173243793675), 'S179': (16.09894388652606, 71.85250733261726), 'S200': (39.704750336442295, 28.23648387463571), 'S234': (60.810665726363546, 64.28605555492099), 'S178': (90.28541556811537, 68.65805384282797), 'S263': (42.27717101398677, 11.825041840461337), 'S140': (67.98021601416599, 6.9308217319302585), 'S141': (21.060501205771054, 26.507093367864698), 'S142': (63.05006521221696, 13.18257283810781), 'S143': (2.463478143266973, 61.14478446049559), 'S144': (48.12366218518258, 84.10469202746704), 'S145': (75.06259544519651, 58.5422601851837), 'S146': (95.60123599627036, 43.97913094387506), 'S147': (74.34189995036178, 11.609296488286681), 'S148': (17.965433635133788, 63.33085887385182), 'S149': (74.20615917766672, 65.52208095253286), 'S262': (0.17343289248021865, 63.576990068891824), 'S238': (49.634945625101466, 19.38133788874029), 'S264': (32.6301359275112, 85.47376716528478), 'S274': (11.49649732736292, 86.71935101342893), 'S239': (95.56447199625478, 43.19016940360852), 'S209': (99.07034642236238, 44.02999331761791), 'S278': (26.2882334721093, 27.00768113184161), 'S266': (47.45799142057356, 78.97836477308576), 'S171': (49.33176416048719, 13.826894194180694), 'S68': (98.64075447695006, 85.71996337730737), 'S69': (21.475953676402703, 85.55511980284926), 'S66': (83.39509040618753, 61.98761245501528), 'S67': (91.1298617434604, 36.882545561337224), 'S64': (13.357075118348993, 97.77590385786279), 'S65': (58.200149808166124, 73.73857123843815), 'S62': (49.47861482413184, 26.67960666521686), 'S63': (96.89136135865022, 90.90517540013927), 'S60': (50.58039245227736, 93.29533146306333), 'S61': (36.97902119286901, 26.348847122886276), 'S272': (21.517919253196172, 62.10060207286867), 'S269': (65.2459415416363, 26.649730909892355), 'S153': (33.205731674474514, 19.19467119476872), 'S152': (41.56854482864628, 8.788182988397642), 'S151': (99.85531601238509, 5.788182686333054), 'S150': (80.44891103617869, 82.86492971963898), 'S157': (43.60445504498435, 0.06302381724558881), 'S156': (77.02192850059313, 50.860139573832484), 'S155': (34.51545211595918, 89.48683975333155), 'S154': (34.30949728734555, 18.76012488542118), 'S159': (50.154732001274525, 82.6359813215888), 'S158': (60.29824376652011, 59.90749108927134), 'S170': (22.72446550166872, 53.5415928761904), 'S9': (21.163303506567733, 42.43380101324538), 'S8': (20.210938743669303, 40.66557294263432), 'S3': (0.14491401932313908, 70.36492151014374), 'S2': (38.15287561244697, 56.729583721593045), 'S1': (59.13356960974928, 39.14509325855236), 'S0': (26.729944483681678, 14.134304914666496), 'S7': (48.9434006416937, 94.03048737842148), 'S6': (91.96027749437499, 95.62210539981741), 'S5': (94.0688944543607, 51.33663763543501), 'S4': (79.39660116260534, 42.943778037517156), 'S168': (91.93446690052764, 81.29956867135725), 'S267': (88.9604753788259, 97.23167337322576), 'S276': (22.29943975857931, 37.81745390382597), 'S19': (84.28075211951912, 40.21336681734997), 'S18': (53.71168713096453, 99.63657149660354), 'S201': (88.84133024003029, 63.81643501431677), 'S277': (89.99749291110089, 86.0334443785486), 'S13': (34.981142177780214, 74.81206894541118), 'S12': (84.03252336380572, 3.3595264920939005), 'S11': (61.21050142053153, 4.369242050497634), 'S10': (69.29090378679851, 89.98566625457609), 'S17': (72.15979645823232, 71.84690508594642), 'S16': (5.100224118035945, 13.292674535206316), 'S15': (69.62351166401763, 46.47351840697582), 'S14': (61.63421248826003, 71.63670497685166), 'S93': (78.80995508373026, 39.54579817699093), 'S92': (9.236079575080158, 99.04362883674743), 'S91': (71.94234617711047, 75.94228839615441), 'S90': (99.35126307289607, 19.331790106228297), 'S97': (77.37971160480754, 7.864029312707322), 'S96': (88.20405202910466, 17.070589897656983), 'S95': (49.77850719408621, 76.78851760452037), 'S94': (65.29359300140439, 70.88154183647562), 'S166': (90.78248469452994, 22.1603668846306), 'S167': (94.54900666185341, 54.84654400048664), 'S99': (99.14798032506326, 23.457293001436664), 'S98': (37.44046872472639, 29.65836472381227), 'S162': (44.67839629786808, 79.76210668408679), 'S163': (78.52751650792852, 23.280428116425256), 'S160': (75.02607342681719, 81.57069276793965), 'S161': (87.28234346229239, 94.64820202027585), 'S214': (28.86783958735146, 85.00228575166169), 'S215': (13.7556746410879, 59.977572391357285), 'S216': (59.81991814453883, 4.3734004787040925), 'S217': (37.34015423506172, 67.38664199409477), 'S108': (68.40899164352085, 68.39532605677691), 'S109': (38.658083902539154, 15.85887865360791), 'S212': (90.55986961230742, 17.311958453282816), 'S213': (30.265383533839973, 25.47655732745944), 'S104': (78.12317453811363, 24.53395180202379), 'S105': (4.011973486367437, 55.33080294126247), 'S106': (88.91768442719442, 3.3510985845171315), 'S169': (87.37318754546962, 50.839104328098564), 'S100': (51.29915684577156, 98.13954734340109), 'S101': (8.616705312912142, 60.762407366932656), 'S102': (35.76087479683624, 63.09310321420391), 'S103': (61.723778159453346, 6.967869644978897), 'S184': (72.32319803250421, 56.221221791686624), 'S185': (14.101858332368334, 60.96224361906927), 'S186': (34.075256667049814, 61.68427100334601), 'S187': (34.57929201099807, 99.13465097066478), 'S180': (3.3106014246813475, 57.353616341082734), 'S181': (94.10695241319674, 4.713572910143949), 'S182': (36.88159322205622, 13.170667729005004), 'S183': (68.77812935068717, 42.653783011854195), 'S208': (62.77720601171729, 69.82257237863456), 'S188': (66.07890017404704, 68.85954452787384), 'S164': (67.58633155375884, 55.0113612540392), 'S279': (64.35768884594228, 49.305417354318045), 'S165': (87.58585822615478, 7.665241998809524), 'S273': (7.705035914422598, 40.405769573066706), 'S275': (54.917715141549216, 50.80269514648252), 'S268': (77.14347322535916, 55.660815921428494), 'S80': (38.306730524247556, 67.68628555126554), 'S81': (59.49657390568752, 53.61289589827749), 'S82': (63.68616411591316, 90.50618525737652), 'S83': (40.02744192795484, 30.417785969083944), 'S84': (43.11994553235533, 78.93506979482918), 'S85': (91.83297905666527, 71.18445910354343), 'S86': (63.925043707776354, 99.81062994309725), 'S87': (3.5648908898020304, 55.81377405453046), 'S88': (64.92400296407806, 10.100225390148388), 'S89': (62.102562153094944, 35.99971536121538), 'S173': (46.29872580581026, 71.56694236838162), 'S172': (84.76146653303718, 10.366915596357817), 'S175': (1.3943322272375491, 2.1753189702899633), 'S174': (36.859618444403374, 35.6756472928366), 'S177': (13.435434657319744, 41.009308649057076), 'S176': (26.970930587807562, 71.12035231893364), 'S207': (65.68690126587363, 83.85674176551096), 'S206': (68.90698215263795, 33.85237320201482), 'S205': (27.202871806669904, 66.85184110867478), 'S204': (86.1770823133681, 46.47558507524151), 'S203': (73.51406531690733, 21.60517864424447), 'S202': (26.450561887532174, 4.509150863123024), 'S119': (20.88249733900952, 88.1078190741769), 'S118': (10.873668250232193, 21.13924090545405), 'S117': (7.735412222696036, 0.9739917707678458), 'S116': (31.720028620040996, 95.96707218017069), 'S115': (75.68933707237895, 47.969815103550694), 'S114': (45.54317696667376, 64.49695436985988), 'S113': (37.16678864022261, 91.31872544191434), 'S112': (2.1868560679521365, 22.700845361956258), 'S111': (52.226369206606925, 53.435699305167496), 'S110': (80.53272628311603, 80.24426609870491), 'S197': (29.926593694227066, 55.61756573662415), 'S196': (3.8148087093794025, 30.73310086967933), 'S195': (2.643870269794457, 65.76208085266573), 'S194': (27.09619318538403, 79.04218403262668), 'S193': (52.02230856568691, 57.436786062338605), 'S192': (97.80537304392874, 95.30203976292297), 'S191': (88.61181631138577, 9.061529388241818), 'S190': (15.495199044602426, 53.749516366208994), 'S210': (81.63600048960691, 3.7074687842908016), 'S199': (20.55709036080683, 9.724571033498307), 'S198': (28.635283336572602, 87.13036196112148), 'S246': (9.843119800927013, 80.13069516024436), 'S271': (30.78886818916937, 71.47290438706167), 'S39': (18.121674887663154, 4.214884784219164), 'S38': (59.884782719009344, 43.71114951098841), 'S35': (21.890783931656145, 15.086374197394115), 'S34': (16.205582059296653, 33.813496050703264), 'S37': (39.2148681693284, 16.368419140056844), 'S36': (58.86522844330819, 23.782732018570975), 'S31': (17.642849400906957, 23.022275561496684), 'S30': (48.11762134405735, 80.54973071880937), 'S33': (65.53196002911957, 62.417541282265454), 'S32': (84.08870022217391, 63.69082448133657), 'S298': (64.99732916973987, 33.195484668562), 'S299': (20.100631791025148, 70.171067186232), 'S294': (4.57217308978678, 6.222511411320042), 'S295': (63.505447830507855, 52.06565899827545), 'S296': (99.89840144671983, 54.735651638139714), 'S297': (27.36187478547786, 8.782303280783898), 'S290': (50.03576834609247, 23.405909659392833), 'S291': (43.09723897353501, 52.41393383987266), 'S231': (21.0559162544109, 11.771687066162073), 'S293': (32.40557347100069, 27.281397420674747), 'S232': (14.765549287368984, 12.020693697832286), 'S233': (9.776944946879373, 35.60202917661612), 'S128': (77.7125587589755, 57.11441922452218), 'S107': (96.49104817815292, 34.2413210711034), 'S236': (51.74827308390489, 26.60654179091403), 'S237': (57.01899013091425, 82.46399747182068), 'S219': (27.76524478849789, 75.94249781290488), 'S235': (13.580624702745414, 28.999344904429336), 'S122': (34.92260078981402, 28.060196124297455), 'S123': (63.114201187434546, 73.50305151497886), 'S120': (30.597087542882505, 80.18811654520036), 'S121': (27.580900931989117, 43.50017295984827), 'S126': (28.690570068541177, 81.22386241507509), 'S127': (42.86806278412924, 30.22903147702236), 'S124': (40.53274847338899, 88.27070880008874), 'S125': (42.77928175112332, 95.93137185846892), 'S22': (18.59332927967423, 42.0962523577999), 'S23': (14.055188081132764, 73.07630009673636), 'S20': (27.750211788387414, 19.01513192794161), 'S21': (79.00350133547109, 86.27258130824123), 'S26': (25.891044857826685, 99.66729738282775), 'S27': (66.83092509235593, 5.014201503688276), 'S24': (56.59249067143898, 74.81652350140034), 'S25': (62.9042192132349, 11.863186067070707), 'S28': (12.776728028906692, 18.21805000392147), 'S29': (95.68528194280772, 59.459626331016814), 'S218': (16.962787233548624, 3.340739289822159), 'S289': (12.613380520742313, 31.68857359195949), 'S288': (8.237917669246396, 34.596600049629124), 'S287': (84.99301809960575, 70.79899971672494), 'S286': (15.289834372185629, 63.481840401881826), 'S285': (19.153505168257144, 41.3264656071364), 'S284': (17.728766093513983, 91.43570438405165), 'S283': (60.76165429801754, 95.11395877347076), 'S282': (75.61665446199792, 0.006106653117921024), 'S281': (8.74666124263287, 50.03829232121985), 'S280': (94.79008082581632, 46.475508928708265)}


    for i in Dic_coordenadas_dispositivos:
        for j in Dispositivos:
            if i == j.nome:
                j.coordenada_dispositivo = Dic_coordenadas_dispositivos[i]

    for i in Dispositivos:
        for j in Dispositivos:
            if (distance.euclidean(j.coordenada_dispositivo, i.coordenada_dispositivo)) <= j.communication_range:
                G.add_edge(i.nome, j.nome, weight = round(distance.euclidean(j.coordenada_dispositivo, i.coordenada_dispositivo),2), prob_falha = float(prob_falha_links)/100)
                G.add_edge(j.nome, i.nome, weight = round(distance.euclidean(i.coordenada_dispositivo, j.coordenada_dispositivo),2), prob_falha = float(prob_falha_links)/100)


    for i in G.edges():
        Dic_registro_custos_originais.update({i:G.edges[i[0],i[1]]['weight']})



    return Dic_coordenadas_dispositivos, G, Dic_registro_custos_originais
