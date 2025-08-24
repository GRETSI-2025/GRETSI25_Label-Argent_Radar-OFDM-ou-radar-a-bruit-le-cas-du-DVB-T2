# DVBT2

<hr>

**_Dépôt labelisé dans le cadre du [Label Reproductible du GRESTI'25](https://gretsi.fr/colloque2025/recherche-reproductible/)_**

| Label décerné | Auteur | Rapporteur | Éléments reproduits | Liens |
|:-------------:|:------:|:----------:|:-------------------:|:------|
| ![](label_argent.png) | Jean-Yves BAUDAIS<br>[@jbaudais](https://github.com/jbaudais) | Jean-Baptiste COURBOT<br>[@courbot](https://github.com/courbot) |  Figures 1, 2, 3 et 4<br>Table 1 | 📌&nbsp;[Dépôt&nbsp;original](https://framagit.org/jbaudais/dvbt2)<br>⚙️&nbsp;[Issue](https://github.com/GRETSI-2025/Label-Reproductible/issues/4)<br>📝&nbsp;[Rapport](https://github.com/akrah/test/tree/main/rapports/Rapport_issue_04) |

<hr>

Dépôt pour le label science reproductible, colloque GRETSI 2025

* Identifiant de la communication : ID1427
* Titre de la communication : Radar OFDM ou radar à bruit : le cas du DVB-T2
* Liste des auteurs : Jean-Yves Baudais <jean-yves.baudais@insa-rennes.fr>

## Outils et version

* Python 3.10.12

## Remarque

Je n'ai pas une grande expérience en Python, ni en langage objet, alors
toute remarque constructive sur le code est bienvenue.

## Reproduction des résultats

La branche `main` peut être clonée ou seul le fichier `dvbt2lib.py` peut être téléchargé en local et les
codes qui suivent exécutés.

L'ordre de reproduction est celui d'apparition dans la
publication. Les codes qui suivent sont à exécuter dans un prompt
python3 qui est initialisé avec

``` python3
from dvbt2lib import *
```

### Figure 1

La figure 1 est tracée avec

``` python3
result(res=2,fichOut="fig1.png")
```

La sortie est enregistrée dans le fichier `fig1.png` qui est la figure 1.

### Table 1

Les données de la table 1 sont obtenues avec

``` python3
result(res=1)
```

La sortie est

``` python3
QPSK   MF s_ac^2/mu_ac^2=1.00 ; s_a^2s_c^2/mu_ac^2=1.00
QPSK   ZF s_ac^2/mu_ac^2=1.00 ; s_a^2s_c^2/mu_ac^2=1.00
16QAM  MF s_ac^2/mu_ac^2=1.32 ; s_a^2s_c^2/mu_ac^2=1.00
16QAM  ZF s_ac^2/mu_ac^2=1.00 ; s_a^2s_c^2/mu_ac^2=1.89
64QAM  MF s_ac^2/mu_ac^2=1.38 ; s_a^2s_c^2/mu_ac^2=1.00
64QAM  ZF s_ac^2/mu_ac^2=1.00 ; s_a^2s_c^2/mu_ac^2=2.69
256QAM MF s_ac^2/mu_ac^2=1.40 ; s_a^2s_c^2/mu_ac^2=1.00
256QAM ZF s_ac^2/mu_ac^2=1.00 ; s_a^2s_c^2/mu_ac^2=3.44
```

avec `s_ac^2/mu_ac^2`$=\frac{\sigma_{ac}^2}{\mu_{ac}^2}$ et
`s_a^2s_c^2/mu_ac^2`$=\frac{\sigma_a^2\sigma_c^2}{\mu_{ac}^2}$.  Les
valeurs numériques sont celles de la table 1.

### Affirmation sur les extremums des gains de traitement (section 5)

Validation de l'affirmation : "Les gains de traitement varient entre
62,6 dB et 63,5 dB pour le radar OFDM et entre 63,5 dB et 63,6 dB pour
le radar à bruit [...] avec $m'\in[1024,1982]$".

``` python3
simul(sim=0)
```

La sortie est 908 lignes listant les paramètres des configurations,
avec en dernière colonne le gain du radar à bruit et en avant dernière
colonne de gain du radar OFDM. Suivent des lignes

``` python3
62.6 <= gain OFDM <= 63.5
63.5 <= gain random <= 63.6
1024 <= m' <= 1982
```

qui donnent les encadrements des gains et de m'.

### Figure 2

La figure 2 est tracée à partir du fichier de données `gretsi1.txt`

``` python3
result(res=11,fichIn="gretsi1.txt",fichOut="out.png")
```

La sortie est le fichier `out.png`.

Les données sont reproduites avec (temps de simulation d'environ 30
min)

``` python3
simul(sim=3,config=all_config(),fichOut="out1.txt",newFile=True)
```

La sortie est le fichier `out1.txt` qui, s'il existe, est écrasé. Il a
la forme du fichier `gretsi1.txt`. Les résultats étant issus de
processus aléatoires, les données peuvent être différentes mais pas
les tendances. La figure 2 est tracée à partir de ce nouveau fichier
de données avec

``` python3
result(res=11,fichIn="out1.txt",fichOut="fig2.png")
```

La sortie est le fichier `fig2.png`.

### Affirmation sur la comparaison des modulations (section 5)

Validation de l'affirmation : "Seule la modulation QPSK est présentée
car la modulation influence peu le PSLR."

``` python3
result(res=14,fichIn="gretsi1.txt",fichOut="out.png")
```
ou, si le fichier `out1.txt` a été généré précédemment,

``` python3
result(res=14,fichIn="out1.txt",fichOut="out.png")
```

La sortie est le fichier `out.png` avec des points très proches de la
première bissectrice.

### Figure 3

La figure 3 est tracée à partir du fichier de données `gretsi2.txt`

``` python3
result(res=20,fichIn='gretsi2.txt',fichOut="out.png")
```

La sortie est le fichier `out.png`.

Les données sont reproduites avec (temps de simulation environ 5 min)

``` python3
simul(sim=2,config=(all_config(config=configFull)[19],),fichOut="out2.txt",SNR=list(np.linspace(-10,20,21)),newFile=True)
```

La sortie est le fichier `out2.txt` qui, s'il existe, est écrasé. Il
est similaire au fichier `gretsi2.txt`. (Cf. le commentaire de la
figure 2 sur la variabilité des résultats.) La figure 3 peut-être
tracée à partir de ce nouveau fichier de données avec

``` python3
result(res=20,fichIn='out2.txt',fichOut="fig3.png")
```

La sortie est le fichier `fig3.png`.

### Figure 4

La figure 4 est tracée à partir du fichier de données `gretsi3.txt`

``` python3
result(res=23,fichIn='gretsi3.txt',fichOut="out.png")
```

La sortie est le fichier `out.png`.

Les données sont reproduites avec (temps de simulation environ 10 min)

``` python3
simul(sim=4,config=(all_config()[799],),D=list(np.linspace(0,100,21)),fichOut="out3.txt",newFile=True)
simul(sim=4,config=(all_config()[799],),d=list(np.linspace(0,900,21)),fichOut="out3.txt",newFile=False)
```

La sortie est le fichier `out3.txt` qui, s'il existe, est écrasé. Il
est similaire au fichier `gretsi3.txt`. (Cf. le commentaire de la
figure 2 sur la variabilité des résultats.) La figure 4 peut-être
tracée à partir de ce nouveau fichier de données avec

``` python3
result(res=23,fichIn='out3.txt',fichOut="fig4.png")
```

La sortie est le fichier `fig4.png`
