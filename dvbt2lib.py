#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:02:19 2024

Convention des axes retard-Doppler
 Doppler = temps long  = symb    = x = shape[0]
 retard  = temps court = carrier = y = shape[1]

@author: jbaudais 

@source: ETSI TS 102 755 v1.1.1 2023/2
"""
# https://docs.python.org/fr/3/tutorial/classes.html
# https://matplotlib.org/stable/gallery/color/color_cycle_default.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from scipy import signal, special
import random as rd
import os

epsZF=1e-10

# combinaison GI-PP table 59 et PP-CP G.1
configIn=(
    ((1024,),(1/16,),("PP4","PP5"),("CP1",)),
    ((1024,),(1/8,),("PP2","PP3"),("CP1",)),
    ((1024,),(1/4,),("PP1",),("CP1",)),
    ((2048,),(1/32,),("PP4","PP7"),("CP1","CP2")),
    ((2048,),(1/16,),("PP4","PP5"),("CP1","CP2")),
    ((2048,),(1/8,),("PP2","PP3"),("CP1","CP2")),
    ((2048,),(1/4,),("PP1",),("CP1","CP2")),
    ((4096,),(1/32,),("PP4","PP7"),("CP1","CP2","CP3")),
    ((4096,),(1/16,),("PP4","PP5"),("CP1","CP2","CP3")),
    ((4096,),(1/8,),("PP2","PP3"),("CP1","CP2","CP3")),
    ((4096,),(1/4,),("PP1",),("CP1","CP2")),
    ((8192,),(1/128,),("PP7",),("CP1","CP2","CP3","CP4")),
    ((8192,),(1/32,),("PP4","PP7"),("CP1","CP2","CP3","CP4")),
    ((8192,),(1/16,19/256),("PP4","PP5"),("CP1","CP2","CP3","CP4")),
    ((8192,),(1/16,19/256),("PP8",),("CP4",)),
    ((8192,),(1/8,19/128),("PP2",),("CP1","CP2","CP3","CP4")),
    ((8192,),(1/8,19/128),("PP3",),("CP1","CP2","CP3")),
    ((8192,),(1/8,19/128),("PP8",),("CP4",)),
    ((8192,),(1/4,),("PP1",),("CP1","CP2")),
    ((8192,),(1/4,),("PP8",),("CP4",)),
    ((16384,),(1/128,),("PP7",),("CP1","CP2","CP3","CP4","CP5")),
    ((16384,),(1/32,),("PP4","PP7"),("CP1","CP2","CP3","CP4","CP5")),
    ((16384,),(1/32,),("PP6",),("CP5",)),
    ((16384,),(1/16,19/256),("PP2","PP4","PP5"),("CP1","CP2","CP3","CP4","CP5")),
    ((16384,),(1/16,19/256),("PP8",),("CP4","CP5")),
    ((16384,),(1/8,19/128),("PP2",),("CP1","CP2","CP3","CP4","CP5")),
    ((16384,),(1/8,19/128),("PP3",),("CP1","CP2","CP3","CP5")),
    ((16384,),(1/8,19/128),("PP8",),("CP4","CP5")),
    ((16384,),(1/4,),("PP1",),("CP1","CP2","CP5")),
    ((16384,),(1/4,),("PP8",),("CP4","CP5")),
    ((32768,),(1/128,),("PP7",),("CP1","CP2","CP3","CP4","CP5","CP6")),
    ((32768,),(1/32,),("PP4",),("CP1","CP2","CP3","CP4","CP5","CP6")),
    ((32768,),(1/32,),("PP6",),("CP5","CP6")),
    ((32768,),(1/16,19/256),("PP2","PP4"),("CP1","CP2","CP3","CP4","CP5","CP6")),
    ((32768,),(1/16,19/256),("PP8",),("CP4","CP5","CP6")),
    ((32768,),(1/8,19/128),("PP2",),("CP1","CP2","CP3","CP4","CP5","CP6")),
    ((32768,),(1/8,19/128),("PP8",),("CP4","CP5","CP6"))
    )

configFull=(
    ((1024,),(0,1/16,1/8,1/4),("PP0",),("CP0",)),
    ((2048,4096),(0,1/32,1/16,1/8,1/4),("PP0",),("CP0",)),
    ((8192,16384),(0,1/128,1/32,1/16,19/256,1/8,19/128,1/4),("PP0",),("CP0",)),
    ((32768,),(0,1/128,1/32,1/16,19/256,1/8,19/128,1/4),("PP0",),("CP0",)),
    )

def all_config(config=configIn):
    """
    @brief Génère la liste de toutes les 924 configurations (FFT,GI,PP,CP,MOD)
    @details Appel : config=all_config(). Pour extraire une sous-liste z=[i for i in config if i[4]=='QPSK']
    @param configIn le format 'compressé' des configurations
    @return out l'ensemble des configurations
    """
    out=[]
    for i in range(0,len(config)):
        a,b,c,d=config[i]
        x=[(j,k,l,m,n) for j in a for k in b for l in c for m in d for n in ("QPSK","16QAM","64QAM","256QAM")]
        out=out+x
    return out

# [(i,all_config()[i]) for i in range(len(all_config()))]

class Param():
    """
    @brief Les paramètres temps, fréquence, mode de l'émetteur DVBT2 (ETSI TS 102 755 v1.1.1), plus quelques paramètres de flexibilité
    @param _TU  time unit, la période d'échantillonnage en s., table 66 de la spec.
    """
    _TU=7e-6/64
    def __init__(self,FFT=1024,GI=1/4,PP='PP1',CP='CP1',MOD='QPSK',boostP=1,boostD=1,seed=-1):
        """
        @brief Initialisation des paramètres et mode DVBT2
        @param FFT FFT size, le mode de la spec., table 56
        @param GI guard interval, la taille du préfixe cyclique relatif à la taille FFT, nombre < 1, table 68
        @param PP pilots pattern, motif des pilotes dispersés, chaine de caractères, par défaut 'PP1', table 59 & 61
        @param CP continual pilots, motif des pilotes continus, chaine de caractères, par défaut 'CP1', 62 & 63 
        @param MOD modulation des datas, chaine de caractères, par défaut 'QPSK', table 12
        @param boostP coefficient d'amplitude de tous les pilotes, 1 par défaut, paramètre hors spec.
        @param boostD coefficient d'amplitude des datas, 1 par défaut, paramètre hors spec.
        @param seed nouvelle séquence pseudo si seed=-1 (par défaut), initialisation des géné. pseudo si seed>=0, paramèter hors spec.
        """
        #self.TU=TU
        self.FFT=FFT
        self.GI=GI
        self.PP=PP
        self.CP=CP
        self.MOD=MOD
        self.boostP=boostP
        self.boostD=boostD
        self.seed=seed
        
    def update(self,F=550e6):
        """
        @brief Calcul des autres paramètres
        @return B attribut fréquence d'échantillonnage en Hz
        @param F attribut fréquence porteuse en Hz, de 470 à 694 MHz, défaut 550 MHz
        @return NP2 attribut nombre de symboles P2 en début de trame, table 52
        @return DeltaF attribut espace interporteuse en Hz
        @return LF attribut frame length, nombre de symboles OFDM dans la trame, table 47
        @return Dx, Dy attributs en fonction de PP, table 58
        @return Asp attribut amplitude des pilotes dispersés, table 61
        @return Kmod attribut pour les pilotes continus, table 62
        @return k32K attribut pour les pilotes continus, fonction de PP et CP, table G.1
        @return Acp attribut amplitude des pilotes continus, table 63
        @return Kmin, Kmax attribut table 67
        @return Aep attribut amplitude des pilotes de bord
        @return Aall attribut amplitude du signal, eq. p.120
        """
        self.F=F
        self.B=1/self._TU
        self.DeltaF=self.FFT/self._TU
        self.LF=int((250e-3-2*1024*self._TU)/(self._TU*self.FFT*(1+self.GI)))
        #if self.FFT==32768:
        #    self.LF=2*int(self.LF/2)
        #    self.GI=1/8
        self.NP2=np.max([int(16*1024/self.FFT),1])
        # Mis à zero en attendant la génération du P2
        self.NP2=0
        if self.PP=="PP0":
            self.Dx=self.FFT
            self.Dy=self.LF
            self.Asp=1
            self.CP="CP0"
        elif self.PP=="PP1":
            self.Dx=3
            self.Dy=4
            self.Asp=4/3
        elif self.PP=="PP2":
            self.Dx=6
            self.Dy=2
            self.Asp=4/3
        elif self.PP=="PP3":
            self.Dx=6
            self.Dy=4
            self.Asp=7/4
        elif self.PP=="PP4":
            self.Dx=12
            self.Dy=2
            self.Asp=7/4
        elif self.PP=="PP5":
            self.Dx=12
            self.Dy=4
            self.Asp=7/3
        elif self.PP=="PP6":
            self.Dx=24
            self.Dy=2
            self.Asp=7/3
        elif self.PP=="PP7":
            self.Dx=24
            self.Dy=4
            self.Asp=7/3
        elif self.PP=="PP8":
            self.Dx=6
            self.Dy=16
            self.Asp=7/3
        self.Kmod=np.min([1632,int(1632*self.FFT/2048)])
        self.k32K=[]
        if self.CP=="CP1":
            if self.PP=="PP1":
                self.k32K=[116,255,285,430,518,546,601,646,744,1662,1893,1995,2322,3309,3351,3567,3813,4032,5568,5706]
            elif self.PP=="PP2":
                self.k32K=[116,318,390,430,474,518,601,646,708,726,1752,1758,1944,2100,2208,2466,3792,5322,5454,5640]
            elif self.PP=="PP3":
                self.k32K=[116,318,342,426,430,518,582,601,646,816,1758,1764,2400,3450,3504,3888,4020,4932,5154,5250,5292,5334]
            elif self.PP=="PP4":
                self.k32K=[108,116,144,264,288,430,518,564,636,646,828,2184,3360,3396,3912,4032,4932,5220,5676,5688]
            elif self.PP=="PP5":
                self.k32K=[108,116,228,430,518,601,646,804,1644,1680,1752,1800,1836,3288,3660,4080,4932,4968,5472]
            elif self.PP=="PP7":
                self.k32K=[264,360,1848,2088,2112,2160,2256,2280,3936,3960,3984,5016,5136,5208,5664]
        elif self.CP=="CP2":
            if self.PP=="PP1":
                self.k32K=[1022,1224,1302,1371,1495,2261,2551,2583,2649,2833,2925,3192,4266,5395,5710,5881,8164,10568,11069,11560,12631,12946,13954,16745,21494]
            elif self.PP=="PP2":
                self.k32K=[1022,1092,1369,1416,1446,1495,2598,2833,2928,3144,4410,4800,5710,5881,6018,6126,10568,11515,12946,13954,15559,16681]
            elif self.PP=="PP3":
                self.k32K=[1022,1495,2261,2551,2802,2820,2833,2922,4422,4752,4884,5710,8164,10568,11069,11560,12631,12946,16745,21494]
            elif self.PP=="PP4":
                self.k32K=[601,1022,1092,1164,1369,1392,1452,1495,2261,2580,2833,3072,4320,4452,5710,5881,6048,10568,11515,12946,13954,15559,16681]
            elif self.PP=="PP5":
                self.k32K=[52,1022,1495,2508,2551,2604,2664,2736,2833,3120,4248,4512,4836,5710,5940,6108,8164,10568,11069,11560,12946,13954,21494]
            elif self.PP=="PP7":
                self.k32K=[116,430,518,601,646,1022,1296,1368,1369,1495,2833,3024,4416,4608,4776,5710,5881,6168,7013,8164,10568,10709,11515,12946,15559,23239,24934,25879,26308,26674]
        elif self.CP=="CP3":
            if self.PP=="PP2":
                self.k32K=[2261,8164]
            elif self.PP=="PP3":
                self.k32K=[13954]
            elif self.PP=="PP4":
                self.k32K=[8164]
            elif self.PP=="PP5":
                self.k32K=[648,4644,16745]
            elif self.PP=="PP7":
                self.k32K=[456,480,2261,6072,17500]
        elif self.CP=="CP4":
            if self.PP in ("PP2","PP4"):
                self.k32K=[10709,19930]
            elif self.PP=="PP5":
                self.k32K=[12631]
            elif self.PP=="PP7":
                self.k32K=[1008,6120,13954]
            elif self.PP=="PP8":
                self.k32K=[116,132,180,430,518,601,646,1022,1266,1369,1495,2261,2490,2551,2712,2833,3372,3438,4086,4098,4368,4572,4614,4746,4830,4968,5395,5710,5881,7649,8164,10568,11069,11560,12631,12946,13954,15760,16612,16745,17500,19078,19930,21494,22867,25879,26308]
        elif self.CP=="CP5":
            if self.PP=="PP1":
                self.k32K=[1369,7013,7215,7284,7649,7818,8025,8382,8733,8880,9249,9432,9771,10107,10110,10398,10659,10709,10785,10872,11115,11373,11515,11649,11652,12594,12627,12822,12984,15760,16612,17500,18358,19078,19930,20261,20422,22124,22867,23239,24934,25879,26308,26674]
            elif self.PP=="PP2":
                self.k32K=[6744,7013,7020,7122,7308,7649,7674,7752,7764,8154,8190,8856,8922,9504,9702,9882,9924,10032,10092,10266,10302,10494,10530,10716,11016,11076,11160,11286,11436,11586,12582,13002,17500,18358,19078,22124,23239,24073,24934,25879,26308]
            elif self.PP=="PP3":
                self.k32K=[1369,5395,5881,6564,6684,7013,7649,8376,8544,8718,8856,9024,9132,9498,9774,9840,10302,10512,10566,10770,10914,11340,11418,11730,11742,12180,12276,12474,12486,15760,16612,17500,18358,19078,19930,20261,20422,22124,22867,23239,24934,25879,26308,26674]
            elif self.PP=="PP4":
                self.k32K=[6612,6708,7013,7068,7164,7224,7308,7464,7649,7656,7716,7752,7812,7860,8568,8808,8880,9072,9228,9516,9696,9996,10560,10608,10728,11148,11232,11244,11496,11520,11664,11676,11724,11916,17500,18358,19078,21284,22124,23239,24073,24934,25879,26308]
            elif self.PP=="PP5":
                self.k32K=[1369,2261,5395,5881,6552,6636,6744,6900,7032,7296,7344,7464,7644,7649,7668,7956,8124,8244,8904,8940,8976,9216,9672,9780,10224,10332,10709,10776,10944,11100,11292,11364,11496,11532,11904,12228,12372,12816,15760,16612,17500,19078,22867,25879]
            elif self.PP=="PP6":
                self.k32K=[116,384,408,518,601,646,672,960,1022,1272,1344,1369,1495,1800,2040,2261,2833,3192,3240,3768,3864,3984,4104,4632,4728,4752,4944,5184,5232,5256,5376,5592,5616,5710,5808,5881,6360,6792,6960,7013,7272,7344,7392,7536,7649,7680,7800,8064,8160,8164,8184,8400,8808,8832,9144,9648,9696,9912,10008,10200,10488,10568,10656,10709,11088,11160,11515,11592,12048,12264,12288,12312,12552,12672,12946,13954,15559,16681,17500,19078,20422,21284,22124,23239,24934,25879,26308,26674]
            elif self.PP=="PP7":
                self.k32K=[6984,7032,7056,7080,7152,7320,7392,7536,7649,7704,7728,7752,8088,8952,9240,9288,9312,9480,9504,9840,9960,10320,10368,10728,10752,11448,11640,11688,11808,12192,12240,12480,12816,16681,22124]
            elif self.PP=="PP8":
                self.k32K=[6720,6954,7013,7026,7092,7512,7536,7596,7746,7758,7818,7986,8160,8628,9054,9096,9852,9924,10146,10254,10428,10704,11418,11436,11496,11550,11766,11862,12006,12132,12216,12486,12762,18358,20261,20422,22124,23239,24934]
        elif self.CP=="CP6":
            if self.PP=="PP2":
                self.k23k=[13164,13206,13476,13530,13536,13764,13848,13938,13968,14028,14190,14316,14526,14556,14562,14658,14910,14946,15048,15186,15252,15468,15540,15576,15630,15738,15840,16350,16572,16806,17028,17064,17250,17472,17784,17838,18180,18246,18480,18900,18960,19254,19482,19638,19680,20082,20310,20422,20454,20682,20874,21240,21284,21444,21450,21522,21594,21648,21696,21738,22416,22824,23016,23124,23196,23238,23316,23418,23922,23940,24090,24168,24222,24324,24342,24378,24384,24540,24744,24894,24990,25002,25194,25218,25260,25566,26674,26944]
            if self.PP=="PP4":
                self.k23k=[13080,13152,13260,13380,13428,13572,13884,13956,14004,14016,14088,14232,14304,14532,14568,14760,14940,15168,15288,15612,15684,15888,16236,16320,16428,16680,16812,16908,17184,17472,17508,17580,17892,17988,18000,18336,18480,18516,19020,19176,19188,19320,19776,19848,20112,20124,20184,20388,20532,20556,20676,20772,21156,21240,21276,21336,21384,21816,21888,22068,22092,22512,22680,22740,22800,22836,22884,23304,23496,23568,23640,24120,24168,24420,24444,24456,24492,24708,24864,25332,25536,25764,25992,26004,26674,26944]
            if self.PP=="PP6":
                self.k23k=[13080,13368,13464,13536,13656,13728,13824,14112,14232,14448,14472,14712,14808,14952,15000,15336,15360,15408,15600,15624,15648,16128,16296,16320,16416,16536,16632,16824,16848,17184,17208,17280,17352,17520,17664,17736,17784,18048,18768,18816,18840,19296,19392,19584,19728,19752,19776,20136,20184,20208,20256,21096,21216,21360,21408,21744,21768,22200,22224,22320,22344,22416,22848,22968,23016,23040,23496,23688,23904,24048,24168,24360,24408,24984,25152,25176,25224,25272,25344,25416,25488,25512,25536,25656,25680,25752,25992,26016]
            if self.PP=="PP7":
                self.k23k=[13416,13440,13536,13608,13704,13752,14016,14040,14112,14208,14304,14376,14448,14616,14712,14760,14832,14976,15096,15312,15336,15552,15816,15984,16224,16464,16560,17088,17136,17256,17352,17400,17448,17544,17928,18048,18336,18456,18576,18864,19032,19078,19104,19320,19344,19416,19488,19920,19930,19992,20424,20664,20808,21168,21284,21360,21456,21816,22128,22200,22584,22608,22824,22848,22944,22992,23016,23064,23424,23448,23472,23592,24192,24312,24360,24504,24552,24624,24648,24672,24768,24792,25080,25176,25224,25320,25344,25584,25680,25824,26064,26944]
            if self.PP=="PP8":
                self.k23k=[10709,11515,13254,13440,13614,13818,14166,14274,14304,14364,14586,14664,15030,15300,15468,15474,15559,15732,15774,16272,16302,16428,16500,16662,16681,16872,17112,17208,17862,18036,18282,18342,18396,18420,18426,18732,19050,19296,19434,19602,19668,19686,19728,19938,20034,21042,21120,21168,21258,21284,21528,21594,21678,21930,21936,21990,22290,22632,22788,23052,23358,23448,23454,23706,23772,24048,24072,24073,24222,24384,24402,24444,24462,24600,24738,24804,24840,24918,24996,25038,25164,25314,25380,25470,25974,26076,26674,26753,26944]
        if self.FFT in (1024,2048):
            self.Acp=4/3
        elif self.FFT==4096:
            self.Acp=4*np.sqrt(2)/3
        else:
            self.Acp=8/3
        self.Aep=self.Asp
        self.Kmin=0
        self.Kmax=int(852*self.FFT/1024)
        if self.PP=="PP0":
            self.Kmax=self.FFT            
        self.Aall=5/np.sqrt(27*(self.Kmax+1)/self.FFT)
            
    def show(self):
        """Affichage des paramètres du signal OFDM"""
        print('Période échant.\t\t%e s\nNb. sous-porteuses\t%d\nIntervalle de garde\t%d' % (self._TU,self.FFT,int(self.GI*self.FFT)))
        print(f'Motif. pilot \t\t{self.PP}\nPilots continus\t\t{self.CP}\nModulation\t\t\t{self.MOD}')
        if hasattr(self,'LF'):
            print(f'Nb. symb. OFDM/T2 \t{self.LF}')
            print('Durée trame T2\t\t%f s' % (self._TU*self.LF*self.FFT*(1+self.GI)))

def fmod(mod):
    """
    @brief Génération de la constellation
    @details Génération des points de la constellation mod
    @param mod La modulation parmi BPSK, QAM, 16QAM, 64QAM, 256QAM
    @return symb Le tableau de tous les symboles de modulation
    """
    choose_mod={'BPSK': 2,'QPSK':4, '16QAM': 16, '64QAM': 64, '256QAM': 256}
    if mod in choose_mod:
        M=choose_mod[mod]
        if M==2:
            A=2.5
            symb=np.array([-1,1])
        else:
            A=M
            n=int(np.sqrt(M))
            vn=np.arange(n)*2+1-n
            symb=(np.array([np.ones(n)]).T*vn+1j*np.array([vn]).T*np.ones(n))
        symb=symb*np.sqrt(3/2/(A-1))
    else:
        print("La modulation *%s* n'est pas implémentée" % mod)
    return symb.flatten()

class Grid():
    """
    @brief Les positions des pilotes et des datas de la trame T2 dans la grille temps-fréquence (ETSI TS 102 755 v1.1.1)
    """
    
    def __init__(self,p):
        """
        @brief Génération et affectation de la grille temps-fréquence : données, pilotes dispersés, continus, de bord et porteuses nulles
        @param p passage d'une instance de la classe Param
        @return mat attribut grille temps-fréquence avec identificateur données, pilotes...
        @return nbD attribut nombre de données
        @return nbSP attribut nombre de pilotes dispersés
        @return nbCP attribut nombre de pilotes continus
        @return nbEP attribut nombre de pilotes de bord
        @return nbNP attribut nombre de sous-porteuses nulles
        """
        self.p=p
        self.mat=4*np.ones((self.p.LF,self.p.FFT))
        # scattered
        for l in range(0,self.p.LF):
            for k in range(self.p.Kmin+1,self.p.Kmax):
                if k%(self.p.Dx*self.p.Dy)==self.p.Dx*(l%self.p.Dy):
                    self.mat[l,k]=3
        # continual
        for i in self.p.k32K:
            self.mat[:,i%self.p.Kmod]=2
        # edge
        if p.CP!="CP0":
            self.mat[:,self.p.Kmin]=1
            self.mat[:,self.p.Kmax]=1
        # null
        self.mat[:,:self.p.Kmin]=0
        self.mat[:,self.p.Kmax+1:]=0
        # décommenter pour avoir un signal en bande de base, sinon analytique
        #self.mat=np.fft.fftshift(self.mat)
        self.nbD=np.count_nonzero(self.mat==4)
        self.nbSP=np.count_nonzero(self.mat==3)
        self.nbCP=np.count_nonzero(self.mat==2) # len(self.p.k32K)*self.p.LF
        self.nbEP=np.count_nonzero(self.mat==1) # 2*self.p.LF
        self.nbNP=np.count_nonzero(self.mat==0) # (self.p.Kmin+self.p.FFT-self.p.Kmax-1)*self.p.LF
    
    def test():
        # exécuter Grid.test() mais pas très orthodoxe !
        a=Param(FFT=32768,PP="PP0")
        a.update()
        b=Grid(a);
        print(f"level: {np.unique(b.mat)}\nSP: {b.nbSP} \nCP: {b.nbCP} ({len(a.k32K)*a.LF})\nEP: {b.nbEP} ({2*a.LF*sum(np.unique(b.mat)==1)})\nNP: {b.nbNP} ({(a.Kmin+a.FFT-a.Kmax-1)*a.LF})\nAll: {b.nbSP+b.nbCP+b.nbEP+b.nbNP+b.nbD}/{b.mat.shape[0]*b.mat.shape[1]}")
        
    def verif(self):
        print(f"level: {np.unique(self.mat)}\nSP: {self.nbSP} \nCP: {self.nbCP} ({len(self.p.k32K)*self.p.LF})\nEP: {self.nbEP} ({2*self.p.LF*sum(np.unique(self.mat)==1)})\nNP: {self.nbNP} ({(self.p.Kmin+self.p.FFT-self.p.Kmax-1)*self.p.LF})\nAll: {self.nbSP+self.nbCP+self.nbEP+self.nbNP+self.nbD}/{self.mat.shape[0]*self.mat.shape[1]}")

    def show(self,carrierMin=0,carrierMax=0,symbMin=0,symbMax=0):
        """@brief Affichage de la grille temps-fréquence sur les coordonnées (carrierMin,carrierMax,symbMin,symbMax), par défaut toute la matrice"""
        plt.figure(dpi=300)
        #plt.rcParams["figure.dpi"]=300 # pour le prompt
        if carrierMax==0:
            carrierMax=self.mat.shape[1]
        if symbMax==0:
            symbMax=self.mat.shape[0]
        plt.imshow(self.mat[symbMin:symbMax,carrierMin:carrierMax].T,origin='lower',aspect='auto',norm=colors.BoundaryNorm(range(0,6),20),cmap=plt.cm.tab20c,extent=(symbMin,symbMax,carrierMin,carrierMax))
        plt.xlabel("symbole OFDM")
        plt.ylabel("sous-porteuse")
        cb=plt.colorbar()
        cb.set_ticks(ticks=[0.5,1.5,2.5,3.5,4.5], labels=["porteuse nulle","pilote de bord","pilote continu","pilot dispersé","donnée"])

class Signal():
    """@brief Le signal DVBT2 en fréquence (symbole de modulation sur la grille) et en temps (ETSI TS 102 755 v1.1.1)"""

    def __init__(self,g):
        """
        @brief Génération du signal en fréquence et en temps
        @param g passage d'une instance de la classe Grid
        @return mat attribut matrice temps-fréquence des sous-porteuses modulées, signal émis
        @return t attribut signal temporel reçu
        """
        if g.p.seed != -1:
            rd.seed(g.p.seed)
            np.random.seed(g.p.seed)
        self.mat=np.zeros((g.p.LF,g.p.FFT),dtype=complex)
        self.mat[g.mat==4]=g.p.Aall*g.p.boostD*np.random.choice(fmod(g.p.MOD),size=(g.nbD,))
        self.mat[g.mat==3]=g.p.Aall*g.p.boostP*g.p.Asp*np.random.choice(fmod("BPSK"),size=(g.nbSP,))
        self.mat[g.mat==2]=g.p.Aall*g.p.boostP*g.p.Acp*np.random.choice(fmod("BPSK"),size=(g.nbCP,))
        self.mat[g.mat==1]=g.p.Aall*g.p.boostP*g.p.Aep*np.random.choice(fmod("BPSK"),size=(g.nbEP,))
        self.t=np.fft.ifft(self.mat)*np.sqrt(g.p.FFT)
        self.t=np.append(self.t[:,g.p.FFT-int(g.p.GI*g.p.FFT):],self.t,axis=1).reshape(-1,)
        self.p=g.p
        
    def test(self):
        pass
        
    def show(self,T,xmin=0,xmax=0,ymin=0,ymax=0,dB=False):
        """
        @brief Affichage en temps-fréquence ou en temps
        @param T type d'affichage : 'F' temps-fréquence (attribut self.mat), 'T' temporel (attribut self.t)
        """
        if T=="F":
            if xmax==0:
                xmax=self.mat.shape[0]
            if ymax==0:
                ymax=self.mat.shape[1]
            plt.figure(dpi=300)
            plt.imshow(np.abs(self.mat[xmin:xmax,ymin:ymax].T)**2,origin="lower",aspect="auto",extent=(xmin,xmax,ymin,ymax),vmin=0,vmax=np.max(np.abs(self.mat)**2))
            plt.xlabel("symbole OFDM")
            plt.ylabel("sous-porteuse")
            plt.colorbar()
        elif T=="T":
            if xmax==0:
                xmax=self.t.shape[0]
            plt.figure(dpi=300)
            t_y=np.abs(self.t[xmin:xmax])**2
            if dB==True:
                t_y=10*np.log10(t_y)
            plt.plot(range(xmin,xmax),t_y)
            plt.grid(which='major')
            plt.xlabel("Échantillon")
            plt.ylabel("Amplitude")
        
    def add_target(self,d=0,D=0,SER=0):
        """
        @brief Ajout d'une cible 
        @param d le retard introduit par la cible, en case distance
        @param D le Doppler introduit par la cible, en case Doppler
        @param SER la RCS de la cible en dB, défaut 0
        """
        zM=self.p.FFT*(1+self.p.GI)*self.p.LF
        z=self.t*np.exp(2*1j*np.pi*D*np.arange(0,len(self.t))/zM)
        self.t=np.fft.ifft(np.fft.fft(z)*np.exp(-2*1j*np.pi*d*np.arange(0,len(self.t))/zM))*10**(SER/10)
        self.t[:int(np.ceil(d))]=0
        
    def add_noise(self,SNR=np.Inf):
        """
        @brief Ajout du bruit
        @param SNR le snr en dB
        """
        tmp=np.random.normal(0,1,self.t.shape[0])+1j*np.random.normal(0,1,self.t.shape[0])
        self.t=self.t+tmp*np.sqrt(10**(-SNR/10)/2*np.mean(np.abs(self.t)**2))

class Rdm():
    """@brief Range-Doppler map. Construction de la carte et affichage"""
    
    def __init__(self,s,symbMin=0,symbMax=0,carrierMin=0,carrierMax=0,wind="KAI",beta=0,ZP=False,FA='MF',OFDM=True,symbSize=0,carrierSize=0,SNR=-np.Inf):
        """
        @brief Construction de la RDM à partir d'une instance Signal et d'une instance Grid
        @param s instance Signal (le signal émis et reçu)
        @param symbMin,symbMax,carrierMin,carrierMax zone de calcul, LFxFFT par défaut en OFDM, LFxFFTx(1+GI) sinon
        @param wind type de fenêtre, KAI pour Kaiser, KAI2 pour 2D, CHE pour Chebyshev, CHE2, KAI par défaut
        @param beta paramètre beta de la fenêtre KAI, ou at en dB pour la fenêtre CHE, 0 par défaut
        @param ZP pour la normalisation de l'amplitude des pilotes à 1 si traitement OFDM, False par défaut
        @param FA pour le filtre adapté ou non, MF pour matched filter, ZF zero forcing, WF pour MMSE, MF par défaut
        @param OFDM traitement OFDM ou non, True par défaut
        @param symbSize,carrierSize nbre d'échantillon short et long, défaut LF x {FFT ou (1+GI)xFFT} cas OFDM ou radar à bruit
        @param SNR Le SNR en dB
        @return mat la RDM
        @return p les attributs p (param) de l'instance signal
        @return symbSize,carrierSize les dimensions temps long et court utilisé
        @return min la valeur min des symboles de modulation (utile pour les perf. "pratique" du filtre ZF)
        """
        self.p=s.p
        self.symbSize=int(symbSize)
        self.carrierSize=int(carrierSize)
        # choix des dimensions short-symb-x et long-carrier-y
        if OFDM==True:
            self.symbSize=self.p.LF
            self.carrierSize=self.p.FFT
        else:
            if self.symbSize==0:
                self.symbSize=self.p.LF
            if self.carrierSize==0:
                self.carrierSize=int(self.p.FFT*(1+self.p.GI))
         # re-génération du signal tps émis
        faMat=np.fft.ifft(s.mat)*np.sqrt(self.p.FFT)
        faMat=np.append(faMat[:,self.p.FFT-int(self.p.GI*self.p.FFT):],faMat,axis=1).reshape(-1,)
        # mise en matrice du signal reçu et du filtre
        if OFDM==True:
            t_a=s.t.reshape(self.p.LF,int(self.p.FFT*(1+self.p.GI)))[:,int(self.p.FFT*self.p.GI):]
            faMat=faMat.reshape(self.p.LF,int(self.p.FFT*(1+self.p.GI)))[:,int(self.p.FFT*self.p.GI):]
        else:
            t_a=s.t[:self.symbSize*self.carrierSize].reshape(self.symbSize,self.carrierSize)
            faMat=faMat[:self.symbSize*self.carrierSize].reshape(self.symbSize,self.carrierSize)
        t_a=np.fft.fft(t_a)/np.sqrt(self.carrierSize)
        faMat=np.fft.fft(faMat)/np.sqrt(self.carrierSize)
        # traitement ZF des pilotes
        if OFDM==True and ZP==True:
            g=Grid(s.p)
            faMat[g.mat==3]=1/self.p.Asp
            faMat[g.mat==2]=1/self.p.Acp
            faMat[g.mat==1]=1/self.p.Aep
        # pavé de corrélation
        if carrierMax==0:
            carrierMax=t_a.shape[1]
        if symbMax==0:
            symbMax=t_a.shape[0]
        # fenêtre
        if wind=="KAI":
            w=np.kaiser(carrierMax-carrierMin+1,beta)[:-1]
        elif wind=="KAI2":
            w=np.array([np.kaiser(symbMax-symbMin+1,beta)[:-1]]).T*np.kaiser(carrierMax-carrierMin+1,beta)[:-1]
        elif wind=="CHE":
            w=signal.windows.chebwin(carrierMax-carrierMin,at=beta)
        elif wind=="CHE2":
            w=np.array([signal.windows.chebwin(symbMax-symbMin,at=beta)]).T*signal.windows.chebwin(carrierMax-carrierMin,at=beta)
        else:
            w=1
        # filtre et FFT2D
        self.min=0
        if FA=='MF':
            faMat=np.conjugate(faMat)
        elif FA=='ZF':
            self.min=np.min(abs(faMat))
            tmp=np.copy(faMat)
            faMat[abs(faMat)<=np.sqrt(epsZF)]=1
            faMat=1/faMat
            faMat[abs(tmp)<=np.sqrt(epsZF)]=0
        elif FA=='WF':
            faMat=np.conjugate(faMat)/(np.abs(faMat)**2+10**(-SNR/10)*np.mean(np.abs(faMat)**2))
        self.mat=np.fft.ifft2(w*t_a[symbMin:symbMax,carrierMin:carrierMax]*faMat[symbMin:symbMax,carrierMin:carrierMax])*np.sqrt((symbMax-symbMin)*(carrierMax-carrierMin))
        self.mat=np.concatenate((self.mat[:1,:],self.mat[-1:0:-1,:]),axis=0)
        
    def test(D=0,d=0,SNR=3,FA='MF',GI=0,OFDM=True):
        # exécuter Rmd.test() mais pas très orthodoxe !
        FFT,GI,PP,CP,MOD=all_config()[18]
        a=Param(FFT=FFT,GI=GI,PP=PP,CP=CP,MOD=MOD)
        a.update();
        #a.LF=
        a.show()
        t=Signal(Grid(a))
        t.add_target(d=d,D=D)
        t.add_noise(SNR=SNR)
        x,y=noise_mode(a)#;x=2230;y=1024
        print(f'noise radar {x} x {y}')
        b=Rdm(t,OFDM=OFDM,wind="CHE",beta=80,FA=FA,symbSize=x,carrierSize=y,SNR=SNR)
        z=np.where(np.abs(b.mat)==np.max(np.abs(b.mat)));print(f'estim max D={z[0]}, d={z[1]}')
        z=b.pislr(dB=True,dx=5,dy=5,x0=D,y0=d)[:2];print(f'PISLR simu {z[0]}, {z[1]}')
        z=b.pislrTheo(SNR=SNR,FA=FA,D=D,d=d,OFDM=OFDM,dB=True,bmin=b.min);print(f'PISLR theo {z[0]}, {z[1]}')
        #b.show(shift=True,view="3D",dBmin=-20,xmin=100,xmax=200,ymin=0,ymax=50)
        b.show(shift=True,view="1Dx",dBmin=-20,ymin=0) # 1K, 16QAM PP0 2048x1115
        #return t

    def show(self,xmin=0,xmax=0,ymin=0,ymax=0,dBmin=-np.Inf,unit="I",shift=False,view="2D"):
        """
        @brief Affichage de la carte retard-Doppler
        @param xmin,xmax,ymin,ymax bornes de l'affichage en échantillons, x : temps long, y : temps court
        @param dBmin la valeur min. de l'axe z
        @param unit le mode des axes, I pour indice, RD pour retard-Doppler, DV pour distance-vitesse
        @param shift fftshift sur le temps long, False par défaut
        @param view 1Dx, 1Dy, 2D ou 3D, 2D par défaut, 1Dx coupe x en ymin
        """
        if xmax==0:
            xmax=self.mat.shape[0]
        else:
            xmax=min(xmax,self.mat.shape[0])
        if ymax==0:
            ymax=self.mat.shape[1]
        else:
            ymax=min(ymax,self.mat.shape[1])
        t_a=np.abs(self.mat[:,ymin:ymax])**2
        if shift==True and xmin==0:
            t_a=np.concatenate((t_a[-int(xmax/2):,:],t_a[:int(xmax/2),:]),axis=0)
            xmin,xmax=tuple(i-int(xmax/2) for i in (xmin,xmax))
        else:
            t_a=t_a[xmin:xmax,:]
        if dBmin != -np.Inf:
            t_a[t_a<10**(dBmin/10)]=10**(dBmin/10)
            t_a=10*np.log10(t_a)
            zTxt="dB"
        else:
            dBmin=np.min(t_a)
            zTxt=""
        fig=plt.figure(dpi=300)
        if unit=="I":
            xTxt,yTxt="Doppler","retard"
        elif unit=="RD":
            xmin,xmax=tuple(i/((1+self.p.GI)*self.p._TU*self.p.FFT*self.p.LF) for i in (xmin,xmax))
            ymin,ymax=tuple(i*self.p._TU for i in (ymin,ymax))
            xTxt,yTxt="Doppler [Hz]","retard [s]"
        elif unit=="DV":
            xmin,xmax=tuple(3e8/2*self.p._TU*i/((1+self.p.GI)*self.p._TU*self.p.FFT*self.p.LF) for i in (xmin,xmax))
            ymin,ymax=tuple(3e8/2*i*self.p._TU for i in (ymin,ymax))
            xTxt,yTxt="vitesse max. [m/s]","distance [m]"
        if view=="1Dx":
            plt.plot(np.arange(xmin,xmax),t_a[:,0])
            plt.grid()
            plt.xlabel(xTxt)
        elif view=="1Dy":
            plt.plot(np.arange(ymin,ymax),t_a[0,:])
            plt.grid()
            plt.xlabel(yTxt)
        elif view=="2D":
            plt.xlabel(xTxt)
            plt.ylabel(yTxt)
            plt.imshow(t_a.T,origin="lower",aspect="auto",extent=(xmin,xmax,ymin,ymax),vmin=np.min(dBmin),vmax=np.max(t_a))
            plt.colorbar()
        elif view=="3D":
            ax=fig.add_subplot(projection="3d")
            X=np.arange(xmin,xmax,(xmax-xmin)/t_a.shape[0])
            Y=np.arange(ymin,ymax,(ymax-ymin)/t_a.shape[1])
            X,Y=np.meshgrid(Y,X)
            ax.plot_surface(X,Y,t_a,vmin=t_a.min(),vmax=t_a.max(),rstride=1,cstride=1,cmap=cm.hot,linewidth=0,antialiased=True)
            ax.set_xlabel(yTxt)
            ax.set_ylabel(xTxt)
            ax.set_zlabel(zTxt)
        plt.show()
        
    def pislr(self,x0=np.Inf,y0=np.Inf,dx=0,dy=0,dB=False):
        """
        @brief Calcul le PSLR et l'ISLR pour une cible connue (x0,y0) ou pour le 1er max.
        @param x0,y0 coordonnées de la cible, en indice et si connues, mettre à np.Inf si inconnues (défaut)
        @param dx,dy zone en indice du lobe principal, x0+-dx,y0+-dy, x temps long, y temps court
        @param dB à True si résultats en dB
        @return prls,islr normaux
        @return pslrm,islrm sans prendre en compte la réponse temps court en x0
        """
        x0=x0%self.symbSize
        y0=y0%self.carrierSize
        t_a=np.abs(self.mat)**2
        a=np.where(t_a==t_a.max())
        if len(a[0])==2:
            a=a[0]
        if x0==np.Inf:
            x0=int(a[0])
        if y0==np.Inf:
            y0=int(a[1])
        # PSLR et ISLR classique
        # taille lobe principal
        z0=t_a[x0,y0]
        # énergie lobe principal et suppression
        z0s=0
        for i in np.arange(-dx,dx+1):
            for j in np.arange(-dy,dy+1):
                z0s=z0s+t_a[(x0+i)%self.symbSize,(y0+j)%self.carrierSize]
                t_a[(x0+i)%self.symbSize,(y0+j)%self.carrierSize]=0
        # calcul PSLR et ISLR
        zM=t_a.max()
        zS=np.sum(t_a)
        pslr=z0/zM
        islr=z0s/zS
        # PSLR et ISLR sans la réponse en x0 
        t_a[x0,:]=0
        zM=t_a.max()
        zS=np.sum(t_a)
        pslrm=z0/zM
        islrm=z0s/zS
        if dB==True:
            pslr,islr,pslrm,islrm=10*np.log10((pslr,islr,pslrm,islrm))
        return pslr,islr,pslrm,islrm
    
    def pislrTheo(self,SNR=np.Inf,FA="MF",D=0,d=0,dB=False,OFDM=True,bmin=0):
        """
        @brief Calcul des perf. théorique, OFDM random sans interférence
        @param SNR Le SNR en dB, défaut infini
        @param FA Le filtre MF ou ZF, défaut MF
        @param D Le Doppler en nombre d'échantillon, défaut 0
        @param d Le retard en nombre d'échantillon, défaut 0
        @param dB à True si résultats en dB, défaut False
        @param OFDM à True pour les perf. OFDM
        @param bmin pour paser la valeur min des coeff du filtre
        @return pslr,islr
        """
        sacmac=1
        scsamac=1
        a=np.abs(fmod(self.p.MOD))
        D=D/self.symbSize
        s2=10**(-SNR/10)
        if OFDM==True:
            D=D/(1+self.p.GI)
            d=(d-self.p.GI*self.p.FFT+np.abs(d-self.p.GI*self.p.FFT))/2
            if FA=='MF':
                sacmac=np.mean(a**4)
            elif FA=='ZF':
                scsamac=np.mean(1/a**2)
            elif FA=='WF':
                sacmac=np.mean(a**4/(a**2+s2)**2)/np.mean(a**2/(a**2+s2))**2
                scsamac=np.mean(a**2/(a**2+s2)**2)/np.mean(a**2/(a**2+s2))**2
        else:
            if FA=='MF':
                sacmac=2
            elif FA=='ZF':
                if bmin==0:
                    scsamac=-special.expi(-epsZF)/np.exp(-2*epsZF)
                else:   
                    sacmac=1/np.exp(-bmin**2)
                    scsamac=-special.expi(-bmin**2)/np.exp(-2*bmin**2)
            elif FA=='WF':
                sacmac=(1+s2+s2*np.exp(s2)*special.expi(-s2)*(2+s2))/(1+s2*np.exp(s2)*special.expi(-s2))**2
                scsamac=(-1-(1+s2)*np.exp(s2)*special.expi(-s2))/(1+s2*np.exp(s2)*special.expi(-s2))**2
        tmp=1+self.symbSize*(self.carrierSize-d)/(sacmac-1+scsamac*(1-np.sinc(D)**2+s2/(1-d/self.carrierSize)+d/(self.carrierSize-d))/np.sinc(D)**2) # OK OFDM avec et sans interf.
        Hn=np.log(self.symbSize*self.carrierSize-1)+0.5772156649+1/2/(self.symbSize*self.carrierSize-1) # approx. Euler-Mascheroni
        pslr=tmp/Hn
        islr=tmp/(self.symbSize*self.carrierSize-1)
        if dB==True:
            pslr,islr=10*np.log10((pslr,islr))
        return pslr,islr

def noise_mode(p,nbCarrier=1000):
    """
    @brief Génération de la configuration tps-longx tps-court pour le traitement radar à bruit
    @param p instance Param mise à jour avec FFT, LF et GI
    @param nbCarrier nbre d'échantillons min. temps court, max. 2*nbCarrier, 1000 par défaut
    @return nbre d'échantillons tps-long x tps-court
    """
    l=p.FFT*(1+p.GI)*p.LF
    t=np.log2(nbCarrier/p.LF)
    if t>0:
        y=p.LF*2**(np.ceil(t))
    else:
        y=p.LF
    if p.LF>2*nbCarrier:
        y=p.LF/2
    return int(l/y), int(y)

def simul(sim=-1,config=all_config(),fichOut="test.txt",SNR=[3],D=[0],d=[0],newFile=False):
    # 0 configuration OFDM et NOISE avec paramètres et gain de traitement
    # 1 affichage des grilles temps-fréquence
    # 2 affichage RDM de base
    # 3 calcul PISLR pour config
    # 4 calcul PISLR vs retard et Doppler, config
    
    # toutes les config
    #config=all_config()
    # les config full-OFDM
    #config=all_config(config=configFull)
    # sélection des config. avec QPSK parmi toutes les config.
    #config=all_config()[0:None:4]
    # génération des config. FFTxGI
    #config=[(configIn[i][0],configIn[i][1][j],configIn[i][2][0],configIn[i][3][0],'QPSK') for i in range(len(configIn)) for j in range(len(configIn[i][1]))]
    # sélection de config. avec critères parmi toutes les config.
    #config=[i for i in all_config() if i[2] in ("PP7","PP8") and i[3] in 'CP4' and i[4] in "QPSK"]
    # configuration 32K

    n=0
    if sim==0:
        out=[np.Inf,-np.Inf,np.Inf,-np.Inf,np.Inf,-np.Inf]
    if newFile==True:
        if os.path.isfile(fichOut):
            os.remove(fichOut)
    for data in config:
        FFT,GI,PP,CP,MOD=data
        print(f'{n} FFT={FFT} GI={GI} {PP} {CP} {MOD} ',end='')
        p=Param(FFT,GI,PP,CP,MOD)
        p.update()
        if sim==0:
            x,y=noise_mode(p)
            print(f'LF={p.LF}, {x}x{y} {10*np.log10(p.FFT*p.LF):.2f} {10*np.log10(x*y):.2f}',end='')
            out=[min(out[0],10*np.log10(p.FFT*p.LF)),max(out[1],10*np.log10(p.FFT*p.LF)),min(out[2],10*np.log10(x*y)),max(out[3],10*np.log10(x*y)),min(out[4],y),max(out[5],y)]
        if sim==2:
            PARAM=np.array([np.kron(SNR,np.kron(np.ones(len(D)),np.ones(len(d)))),
                            np.kron(np.ones(len(SNR)),np.kron(D,np.ones(len(d)))),
                            np.kron(np.ones(len(SNR)),np.kron(np.ones(len(D)),d))])
            print('')
            for var in PARAM.T:
                SNRi,Di,di=var
                b=Signal(Grid(p))
                x=2230
                y=1024
                b.add_target(d=di,D=Di)
                b.add_noise(SNR=SNRi)
                lconf=((True,'MF'),
                       (True,'ZF'),
                       (True,'WF'),
                       (False,'MF'),
                       (False,'ZF'),
                       (False,'WF'))
                out=()
                for c in lconf:
                    OFDM=c[0]
                    FA=c[1]
                    print(f'SNR={SNRi} D={Di} d={di} OFDM={OFDM} {FA}',end='')
                    r=Rdm(b,OFDM=OFDM,wind="NULL",beta=80,FA=FA,symbSize=x,carrierSize=y,SNR=SNRi)
                    pslr=r.pislr(dB=True,dx=5,dy=5,x0=int(Di),y0=int(di))[0]
                    pslrTheo=r.pislrTheo(SNR=SNRi,FA=FA,D=Di,d=di,OFDM=OFDM,dB=True)[0]
                    print(f' {pslr:.2f} {pslrTheo:.2f}')
                    out=out+(pslr,pslrTheo)
                with open(fichOut,"a") as f:
                    f.write(f'{p.LF} {FFT} {GI} {PP} {CP} {MOD} {x} {y} {SNRi} {Di} {di}')
                    for i in out:
                        f.write(f' {i}')
                    f.write('\n')
        if sim==3:
            b=Signal(Grid(p))
            x,y=noise_mode(p)
            b.add_target(d=0,D=0);
            b.add_noise(SNR=3)
            lconf=(('NUL',True,'MF',False),
                  ('CHE',True,'MF',False),
                  ('CHE',True,'MF',True),
                  ('CHE',True,'ZF',False),
                  ('CHE',False,'MF',False),
                  ('CHE',False,'ZF',False))
            out=()
            for c in lconf:
                r=Rdm(b,wind=c[0],beta=80,OFDM=c[1],carrierSize=y,symbSize=x,FA=c[2],ZP=c[3])
                pislr=r.pislr(dB=True,dx=5,dy=5,x0=0,y0=0)
                out=out+pislr
            with open(fichOut,"a") as f:
                f.write(f"{p.LF} {FFT} {GI} {PP} {CP} {MOD} {x} {y}")
                for i in out:
                    f.write(f' {i}')
                f.write('\n')
        if sim==4:
            PARAM=np.array([np.kron(SNR,np.kron(np.ones(len(D)),np.ones(len(d)))),
                            np.kron(np.ones(len(SNR)),np.kron(D,np.ones(len(d)))),
                            np.kron(np.ones(len(SNR)),np.kron(np.ones(len(D)),d))])
            print('')
            for var in PARAM.T:
                SNRi,Di,di=var
                b=Signal(Grid(p))
                x,y=noise_mode(p)
                b.add_target(d=di,D=Di)
                b.add_noise(SNR=SNRi)
                lconf=((True,'MF'),
                       (True,'ZF'),
                       (True,'WF'),
                       (False,'MF'),
                       (False,'ZF'),
                       (False,'WF'))
                out=()
                for c in lconf:
                    OFDM=c[0]
                    FA=c[1]
                    print(f'SNR={SNRi} D={Di} d={di} OFDM={OFDM} {FA}',end='')
                    r=Rdm(b,OFDM=OFDM,wind="CHE",beta=80,FA=FA,symbSize=x,carrierSize=y,SNR=SNRi)
                    pslr=r.pislr(dB=True,dx=5,dy=5,x0=int(Di),y0=int(di))[0]
                    pslrTheo=r.pislrTheo(SNR=SNRi,FA=FA,D=Di,d=di,OFDM=OFDM,dB=True)[0]
                    print(f' {pslr:.2f} {pslrTheo:.2f}')
                    out=out+(pslr,pslrTheo)
                with open(fichOut,"a") as f:
                    f.write(f'{p.LF} {FFT} {GI} {PP} {CP} {MOD} {x} {y} {SNRi} {Di} {di}')
                    for i in out:
                        f.write(f' {i}')
                    f.write('\n')
        if sim==5:
            pass
        n=n+1
        print('')
    if sim==0:
        print(f"\n{out[0]:.1f} <= gain OFDM <= {out[1]:.1f}\n{out[2]:.1f} <= gain random <= {out[3]:.1f}\n{out[4]:d} <= m' <= {out[5]:d}")
    #return

def result(res=0,fichIn="",fichOut=""):
    if res==1:
        for MOD in ('QPSK','16QAM','64QAM','256QAM'):
            a=np.abs(fmod(MOD))
            for FA in  ('MF','ZF'):
                if FA=='MF':
                    c=np.conjugate(a)
                elif FA=='ZF':
                    c=1/a
                sac2=np.mean(np.abs(a*c)**2)
                muac=np.mean(a*c)
                sc2=np.mean(np.abs(c)**2)
                sa2=np.mean(np.abs(a)**2)
                print(f'{MOD:6} {FA} s_ac^2/mu_ac^2={sac2/muac**2:.2f} ; s_a^2s_c^2/mu_ac^2={sa2*sc2/muac**2:.2f}')
                
    if res==2:
        D=100;d=300
        a=Param();a.update();a.show()
        t=Signal(Grid(a));t.add_target(d=d,D=D);t.add_noise(SNR=3)
        b=Rdm(t,wind="CHE",beta=80,OFDM=True,FA='MF')
        c,d,e,f=b.pislr(dB=True,dx=5,dy=5,x0=D,y0=d)
        print(f'[dB] PSLR={c:.2f} ISLR={d:.2f} (out of d0: PSLR={e:.2f} ISLR={f:.2f})')
        b.show(dBmin=-20,unit="I",shift=True,view="3D")
        plt.show()
    if np.floor(res/10)==1:
        c0=8
        if fichIn=="":
            print("Error fichIn empty abort")
            return
        with open(fichIn) as f:
            # suppression des lignes False
            data=list(filter(None,f.read().split('\n')[0:]))
        dataMod=[data[i] for i in range(len(data)) if data[i].split()[5]=='QPSK']
        #y=[i.split()[c0+j] for i in data for j in np.arange(0,24,4)] # all PSLR 1
        #y=[i.split()[7] for i in data if i.split()[1] in ("0.25","0.125")]
        y=np.array([i.split()[c0+0] for i in dataMod],dtype=np.float32)
        z=np.array([i.split()[c0+4] for i in dataMod],dtype=np.float32)
        t=np.array([i.split()[c0+8] for i in dataMod],dtype=np.float32)
        u=np.array([i.split()[c0+12] for i in dataMod],dtype=np.float32)
        v=np.array([i.split()[c0+16] for i in dataMod],dtype=np.float32)
        w=np.array([i.split()[c0+20] for i in dataMod],dtype=np.float32)
        plt.figure(dpi=300)
        if res==11:
            plt.plot(y,z,'y.',label='OFDM MF w/o CHE vs w/ CHE')
            plt.plot(z,t,'r.',label='OFDM CHE MF w/o ZP vs w/ ZP')
            plt.plot(z,u,'b.',label='OFDM CHE w/ MF vs w/ ZF')
            plt.plot(z,v,'g.',label='CHE MF w/ OFDM vs w/ randon')
            vm=max(np.min((y,z)),np.min((z,t,u,v)));vM=min(np.max((y,z)),np.max((z,t,u,v)))
        elif res==14:
            dataQPSK=[i.split()[c0:] for i in data if i.split()[5]=='QPSK']
            data16QAM=[data[i].split()[c0:] for i in range(len(data)) if data[i].split()[5]=='16QAM']
            data64QAM=[data[i].split()[c0:] for i in range(len(data)) if data[i].split()[5]=='64QAM']
            data256QAM=[data[i].split()[c0:] for i in range(len(data)) if data[i].split()[5]=='256QAM']
            #vv=np.array([0,4,8,12,16,20],dtype=int) # tout
            vv=np.array([0,4,8,12,16],dtype=int) # sauf ZF nOFDM
            y=np.array(dataQPSK,dtype=np.float32)[:,vv].reshape(-1)
            z=np.array(data16QAM,dtype=np.float32)[:,vv].reshape(-1)
            t=np.array(data64QAM,dtype=np.float32)[:,vv].reshape(-1)
            u=np.array(data256QAM,dtype=np.float32)[:,vv].reshape(-1)
            plt.plot(y,z,'b.',label='QPSK vs. 16QAM')
            plt.plot(y,t,'r.',label='QPSK vs. 64QAM')
            plt.plot(y,u,'g.',label='QPSK vs. 256QAM')
            vm=max(np.min(y),np.min((z,t,u)));vM=min(np.max(y),np.max((z,t,u)))
        plt.plot([vm,vM],[vm,vM],'k--')
        plt.grid(which='major')
        plt.xlabel("PSLR [dB]")
        plt.ylabel("PSLR [dB]")
        plt.legend()
        plt.show()
    if res==20:
        c0=8
        vec=0 # colonne 0-SNR, 1-D, 2-d
        with open(fichIn) as f:
            # suppression des lignes False
            data=list(filter(None,f.read().split('\n')[0:]))
        z=np.array([d.split()[c0:] for d in data],dtype=np.float32)
        plt.figure(dpi=300)
        pltLegend=plt.plot(z[:,vec],z[:,3:15:2],'.') # simu 3,5,7/9,11,13 ; all 3:15:2
        plt.gca().set_prop_cycle(None)
        pltLegend=pltLegend+plt.plot(z[:,vec],z[:,4:15:2],'--') # theo 4,6,8/10,12,14 ; all 4:15:2
        plt.legend(pltLegend,('OFDM MF','OFDM ZF','OFDM WF','random MF','random ZF','random WF'),shadow=False,framealpha=.8)
        plt.grid(which='major')
        plt.xlabel("SNR [dB]")
        plt.ylabel("PSLR [dB]")
    if res==21:
        vec=2 # colonne 0-SNR, 1-D, 2-d
        with open(fichIn) as f:
            # suppression des lignes False
            data=list(filter(None,f.read().split('\n')[0:]))
        z=np.array([d.split() for d in data],dtype=np.float32)
        plt.figure(dpi=300)
        plt.gca().set_prop_cycle(color=['tab:blue','tab:green','tab:red','tab:brown'])
        pltLegend=[]
        pltLegend=plt.plot(z[:,vec],z[:,[3,7,9,13]],'.') # simu 3,5,7/9,11,13 ; all 3:15:2
        plt.gca().set_prop_cycle(color=['tab:blue','tab:green','tab:red','tab:brown'])
        #plt.gca().set_prop_cycle(None)
        pltLegend=pltLegend+plt.plot(z[:,vec],z[:,[4,8,10,14]],'--') # theo 4,6,8/10,12,14 ; all 4:15:2
        plt.legend(pltLegend,('OFDM MF','OFDM ZF','OFDM WF','random MF','random ZF','random WF'),shadow=False,framealpha=.8)
        plt.grid(which='major')
        plt.xlabel("SNR [dB]")
        plt.ylabel("PSLR [dB]")
    if res==23:
        c0=11-2
        if fichIn=="":
            print("Error fichIn empty abort")
            return
        with open(fichIn) as f:
            # suppression des lignes False
            data=list(filter(None,f.read().split('\n')[0:]))
        z=np.array([d.split()[c0:] for d in data],dtype=np.float32)
        plt.figure(dpi=300)
        plt.subplot(1,2,1)
        plt.gca().set_prop_cycle(color=['tab:blue','tab:green','tab:red','tab:brown'])
        plt.plot(z[0:21,0],z[0:21,[2,6,8,12]],'.-')
        plt.gca().set_prop_cycle(color=['tab:blue','tab:green','tab:red','tab:brown'])
        plt.plot(z[0:21,0],z[0:21,[3,7,9,13]],'--')
        plt.grid(which='major')
        plt.xlabel("Doppler [échant.]")
        plt.ylabel("PSLR [dB]")
        plt.ylim(11,54)
        plt.subplot(1,2,2)
        plt.gca().set_prop_cycle(color=['tab:blue','tab:green','tab:red','tab:brown'])
        plt.plot(z[21:42,1],z[21:42,[2,6,8,12]],'.-')
        plt.gca().set_prop_cycle(color=['tab:blue','tab:green','tab:red','tab:brown'])
        plt.plot(z[21:42,1],z[21:42,[3,7,9,13]],'--')
        plt.grid(which='major')
        plt.xlabel("retard [échant.]")
        plt.legend(('OFDM MF','OFDM WF','random MF','random WF'))
        plt.ylim(11,54)
    if fichOut!="":
        plt.savefig(fichOut)
    #return v
