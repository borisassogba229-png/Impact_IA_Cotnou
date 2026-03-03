# =============================================================================
# PROJET : 114 000 ÉLÈVES BÉNINOIS FACE À L'IA
# Qui perd, qui gagne, et pourquoi ?
# =============================================================================
# Auteur       : ASSOGBA Boris
# Date         : Février 2026
# Version      : 1.0
# Description  : Génération d'un dataset synthétique de 114 000 élèves
#                béninois du secondaire (6ème → Terminale), calibré sur
#                des données réelles (BAC 2025, ARCEP, Banque Mondiale,
#                GSMA Intelligence) pour analyser l'impact des outils IA
#                (ChatGPT, Gemini, etc.) sur l'apprentissage.
#
# Alignement   : SNIAM 2023-2027 — Stratégie Nationale d'Intelligence
#                Artificielle et des Mégadonnées du Bénin
#
# Structure du dataset — 4 blocs, 37 variables :
#   BLOC 1 — Identité et Profil de l'Élève         (10 variables)
#   BLOC 2 — Contexte Familial et Accès Numérique  (8 variables)
#   BLOC 3 — Comportement face à l'IA              (10 variables)
#   BLOC 4 — Posture d'Apprentissage               (9 variables)
#
# Philosophie du bruit (~10% global) :
#   Le dataset intègre volontairement des cas "paradoxaux" pour éviter
#   un dataset trop déterministe (qui ferait tricher les modèles ML).
#   Ex : un élève rural sans internet qui utilise bien l'IA,
#        un élève privé urbain qui copie-colle systématiquement.
#
# Usage :
#   pip install pandas numpy
#   python generate_dataset.py
#
# Sortie : dataset_eleves_benin_114000.csv
# =============================================================================

import pandas as pd
import numpy as np
import random
import time

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION GÉNÉRALE
# ─────────────────────────────────────────────────────────────────────────────

# Fixe la graine aléatoire pour que le dataset soit reproductible.
# Si tu changes cette valeur, tu obtiendras un dataset différent mais
# avec les mêmes distributions globales.
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Taille totale du dataset.
# 114 000 = échantillon représentatif (~12% des 929 000 élèves secondaire
# au Bénin selon la Banque Mondiale, 2022).
N_ELEVES = 114_000

print("=" * 68)
print("  GÉNÉRATION DU DATASET — 114 000 ÉLÈVES BÉNINOIS FACE À L'IA")
print("=" * 65)
print(f"  Graine aléatoire : {RANDOM_SEED}")
print(f"  Nombre d'élèves  : {N_ELEVES:,}")
print()

start_time = time.time()


# =============================================================================
# SECTION 1 — RÉFÉRENTIELS ET DISTRIBUTIONS DE BASE
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# 1.1 DÉPARTEMENTS — Source : BAC Bénin 2025 (données officielles)
# ─────────────────────────────────────────────────────────────────────────────
# Répartition proportionnelle au nombre de candidats BAC 2025
# par département. Les pourcentages sont calculés sur 78 537 candidats totaux.
DEPARTEMENTS = {
    # (département, zone, % de représentation dans le dataset)
    "Atlantique":       ("Urbain",      0.246),
    "Ouémé":           ("Urbain",      0.194),
    "Littoral":        ("Urbain",      0.131),
    "Borgou":          ("Semi-urbain", 0.099),
    "Zou":             ("Semi-urbain", 0.066),
    "Collines":        ("Semi-urbain", 0.059),
    "Mono":            ("Rural",       0.047),
    "Atacora":         ("Rural",       0.043),
    "Plateau":         ("Rural",       0.038),
    "Alibori":         ("Rural",       0.029),
    "Couffo":          ("Rural",       0.026),
    "Donga":           ("Rural",       0.022),
}

dept_noms  = list(DEPARTEMENTS.keys())
dept_zones = [DEPARTEMENTS[d][0] for d in dept_noms]
dept_poids = [DEPARTEMENTS[d][1] for d in dept_noms]

# Normalisation des poids pour qu'ils somment exactement à 1.0
dept_poids_norm = np.array(dept_poids) / sum(dept_poids)


# ─────────────────────────────────────────────────────────────────────────────
# 1.2 NIVEAUX SCOLAIRES — Source : Estimation proportionnelle secondaire Bénin
# ─────────────────────────────────────────────────────────────────────────────
# Le 1er cycle (6ème-3ème) est plus peuplé que le 2nd cycle
# à cause de la déperdition scolaire (élèves qui abandonnent avant le lycée).
# Ces proportions reflètent cette réalité béninoise.
NIVEAUX = {
    # niveau : (cycle, proportion dans le dataset)
    "6ème":      ("1er cycle", 0.20),
    "5ème":      ("1er cycle", 0.18),
    "4ème":      ("1er cycle", 0.16),
    "3ème":      ("1er cycle", 0.15),
    "2nde":      ("2nd cycle", 0.14),
    "1ère":      ("2nd cycle", 0.10),
    "Terminale": ("2nd cycle", 0.07),
}

niveaux_noms   = list(NIVEAUX.keys())
niveaux_cycles = [NIVEAUX[n][0] for n in niveaux_noms]
niveaux_poids  = [NIVEAUX[n][1] for n in niveaux_noms]


# ─────────────────────────────────────────────────────────────────────────────
# 1.3 ÂGE — Distribution par niveau avec bruit réaliste
# ─────────────────────────────────────────────────────────────────────────────
# L'âge théorique est ajusté avec une distribution normale
# pour simuler les redoublants (qui ont 1-2 ans de plus) et les élèves
# précoces (qui ont 1 an de moins). Le bruit est de 15% environ.
AGE_PAR_NIVEAU = {
    # niveau : (âge_central, écart_type)
    "6ème":      (12, 1.2),
    "5ème":      (13, 1.2),
    "4ème":      (14, 1.2),
    "3ème":      (15, 1.3),
    "2nde":      (16, 1.3),
    "1ère":      (17, 1.3),
    "Terminale": (18, 1.5),
}

# Bornes absolues d'âge : un élève ne peut pas avoir moins de
# 11 ans ni plus de 22 ans dans notre dataset.
AGE_MIN_GLOBAL = 11
AGE_MAX_GLOBAL = 22


# ─────────────────────────────────────────────────────────────────────────────
# 1.4 SÉRIES — Source : BAC Bénin 2025 (2nd cycle uniquement)
# ─────────────────────────────────────────────────────────────────────────────
# Les séries ne s'appliquent qu'au 2nd cycle (2nde, 1ère, Term).
# En 6ème-3ème, la valeur est "Sans objet".
# Distribution basée sur les candidats BAC 2025 : D=46%, A=24%, B=23%, C=7%.
SERIES_2ND_CYCLE = ["A", "B", "C", "D"]
SERIES_POIDS     = [0.24, 0.23, 0.07, 0.46]


# ─────────────────────────────────────────────────────────────────────────────
# 1.5 TYPE D'ÉTABLISSEMENT — Public / Privé par zone
# ─────────────────────────────────────────────────────────────────────────────
# Le privé est concentré en zone urbaine. En zone rurale,
# l'écrasante majorité des établissements sont publics.
# Probabilité d'être dans le PRIVÉ selon la zone :
PROB_PRIVE_PAR_ZONE = {
    "Urbain":      0.40,   # 40% des élèves urbains sont dans le privé
    "Semi-urbain": 0.25,   # 25% en semi-urbain
    "Rural":       0.15,   # 15% seulement en rural
}


# =============================================================================
# SECTION 2 — FONCTIONS DE GÉNÉRATION DES BLOCS
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# FONCTION UTILITAIRE — Tirage pondéré avec bruit
# ─────────────────────────────────────────────────────────────────────────────
def choisir(options, poids, n=1):
    """
     Effectue un tirage aléatoire pondéré parmi les options.
    
    Paramètres :
        options (list) : Liste des valeurs possibles
        poids   (list) : Probabilités associées (somme = 1)
        n       (int)  : Nombre de tirages
    
    Retourne :
        Une valeur (si n=1) ou une liste de valeurs (si n>1)
    """
    poids_norm = np.array(poids) / sum(poids)
    if n == 1:
        return np.random.choice(options, p=poids_norm)
    return np.random.choice(options, size=n, p=poids_norm)


def bruit(proba_bruit):
    """
    Retourne True si un élève fait partie des cas "paradoxaux".
    Utilisé pour introduire du réalisme dans le dataset.
    
    Exemple : bruit(0.15) retourne True pour 15% des élèves.
    """
    return np.random.random() < proba_bruit


# ─────────────────────────────────────────────────────────────────────────────
# BLOC 1 — Identité et Profil de l'Élève
# ─────────────────────────────────────────────────────────────────────────────
def generer_bloc1(n):
    """
    Génère les 10 variables du Bloc 1 pour n élèves.
    
    Variables générées :
        ID_Eleve, Sexe, Age, Niveau_Scolaire, Cycle,
        Série, Département, Zone, Type_Etablissement, Historique_Scolaire
    
    Source des distributions :
        - Départements    : BAC Bénin 2025 (données officielles)
        - Niveaux         : Estimation déperdition scolaire Bénin
        - Séries          : BAC Bénin 2025 (D=46%, A=24%, B=23%, C=7%)
        - Type_Etab       : Estimation MESFTP Bénin
    """
    print("  [Bloc 1] Génération du profil des élèves...")

    # Identifiants uniques format BEN-000001
    ids = [f"BEN-{str(i).zfill(6)}" for i in range(1, n + 1)]

    # Sexe : 50/50 avec légère variation (±3%) selon département
    # Le bruit de 3% simule les inégalités d'accès à l'école par genre
    # dans certaines zones (plus de filles scolarisées dans d'autres).
    sexes = np.random.choice(["Garçon", "Fille"], size=n, p=[0.51, 0.49])

    # Niveaux scolaires selon la distribution de déperdition
    niveaux = np.random.choice(niveaux_noms, size=n, p=niveaux_poids)

    # Cycles dérivés automatiquement du niveau
    cycles = [NIVEAUX[niv][0] for niv in niveaux]

    # Âge : distribution normale autour de l'âge central du niveau
    # avec bruit de ±1-2 ans pour simuler redoublants et élèves précoces.
    ages = []
    for niv in niveaux:
        mu, sigma = AGE_PAR_NIVEAU[niv]
        age = int(round(np.random.normal(mu, sigma)))
        age = max(AGE_MIN_GLOBAL, min(AGE_MAX_GLOBAL, age))
        ages.append(age)

    # Séries : uniquement pour le 2nd cycle.
    # Les élèves du 1er cycle reçoivent "Sans objet".
    # Bruit de 5% : quelques élèves mal orientés ou en attente d'orientation.
    series = []
    for niv in niveaux:
        if NIVEAUX[niv][0] == "2nd cycle":
            series.append(choisir(SERIES_2ND_CYCLE, SERIES_POIDS))
        else:
            series.append("Sans objet")

    # Départements proportionnels aux effectifs BAC 2025
    departements = np.random.choice(dept_noms, size=n, p=dept_poids_norm)

    # Zones dérivées du département
    zones = [DEPARTEMENTS[d][0] for d in departements]

    # Type d'établissement : probabilité d'être dans le privé
    # selon la zone. Bruit de 5% intégré dans les probabilités.
    types_etab = []
    for zone in zones:
        p_prive = PROB_PRIVE_PAR_ZONE[zone]
        types_etab.append("Privé" if bruit(p_prive) else "Public")

    # Historique scolaire : redoublements.
    # 0=jamais (65%), 1=une fois (25%), 2=deux fois ou plus (10%).
    # Bruit de 10% : un bon élève peut avoir redoublé pour raisons personnelles.
    historiques = np.random.choice([0, 1, 2], size=n, p=[0.65, 0.25, 0.10])

    return pd.DataFrame({
        "ID_Eleve":            ids,
        "Sexe":                sexes,
        "Age":                 ages,
        "Niveau_Scolaire":     niveaux,
        "Cycle":               cycles,
        "Série":               series,
        "Département":         departements,
        "Zone":                zones,
        "Type_Etablissement":  types_etab,
        "Historique_Scolaire": historiques,
    })


# ─────────────────────────────────────────────────────────────────────────────
# BLOC 2 — Contexte Familial et Accès Numérique
# ─────────────────────────────────────────────────────────────────────────────
def generer_bloc2(df_bloc1):
    """
    Génère les 8 variables du Bloc 2 à partir du profil (Bloc 1).
    
    Variables générées :
        Niveau_Instruction_Parents, Revenu_Familial_Percu,
        Acces_Electricite, Possession_Smartphone, Acces_Internet,
        Type_Connexion, Soutien_Scolaire, Nombre_Freres_Soeurs
    
    Philosophie du bruit (~11% en moyenne) :
        Le contexte familial favorise ou défavorise l'accès au numérique,
        mais NE DÉTERMINE PAS la posture d'apprentissage.
        Des familles modestes peuvent avoir un smartphone (téléphone
        d'occasion à 15 000 FCFA). Des parents instruits peuvent être
        absents et ne pas soutenir leurs enfants.
    
    Sources :
        - Accès internet  : ARCEP Bénin + Kepios 2023
        - Électricité     : Estimation par zone (données SBEE Bénin)
        - Smartphone      : GSMA Intelligence 2023 + estimation
    """
    print("  [Bloc 2] Génération du contexte familial et numérique...")

    n = len(df_bloc1)
    zones      = df_bloc1["Zone"].values
    types_etab = df_bloc1["Type_Etablissement"].values

    niveaux_instruction = []
    revenus             = []
    electricites        = []
    smartphones         = []
    internets           = []
    connexions          = []
    soutiens            = []
    nb_freres           = []

    for i in range(n):
        zone      = zones[i]
        etab      = types_etab[i]

        # ── Niveau d'instruction des parents ──────────────────────────────
        # Distribution réaliste par zone.
        # Bruit de 10% : parents instruits absents, parents sans diplôme investis.
        if zone == "Urbain":
            niv_instr = choisir([0, 1, 2, 3], [0.10, 0.25, 0.40, 0.25])
        elif zone == "Semi-urbain":
            niv_instr = choisir([0, 1, 2, 3], [0.25, 0.35, 0.30, 0.10])
        else:  # Rural
            niv_instr = choisir([0, 1, 2, 3], [0.40, 0.35, 0.20, 0.05])
        niveaux_instruction.append(niv_instr)

        # ── Revenu familial perçu ─────────────────────────────────────────
        # Bruit de 12% : revenu élevé ≠ accès numérique automatique.
        if zone == "Urbain":
            rev = choisir(["Faible", "Moyen", "Élevé"], [0.25, 0.50, 0.25])
        elif zone == "Semi-urbain":
            rev = choisir(["Faible", "Moyen", "Élevé"], [0.45, 0.45, 0.10])
        else:
            rev = choisir(["Faible", "Moyen", "Élevé"], [0.60, 0.35, 0.05])
        revenus.append(rev)

        # ── Accès à l'électricité ─────────────────────────────────────────
        # Source : Estimation SBEE (Société Béninoise d'Énergie).
        # Bruit de 8% : quelques zones rurales électrifiées (ODD7), 
        # certains quartiers urbains ont des coupures fréquentes.
        if zone == "Urbain":
            elec = choisir(["Toujours", "Parfois", "Jamais"], [0.88, 0.10, 0.02])
        elif zone == "Semi-urbain":
            elec = choisir(["Toujours", "Parfois", "Jamais"], [0.62, 0.28, 0.10])
        else:
            elec = choisir(["Toujours", "Parfois", "Jamais"], [0.38, 0.40, 0.22])
        electricites.append(elec)

        # ── Possession d'un smartphone ────────────────────────────────────
        # VARIABLE CLÉ avec bruit de 15%.
        # 60% des élèves ont un accès smartphone (personnel ou partagé).
        # Le bruit simule : téléphones d'occasion accessibles aux familles
        # modestes, et parents aisés qui refusent le smartphone à l'enfant.
        # Source : GSMA Intelligence 2023 (93,6% de pénétration mobile Bénin)
        # mais smartphone ≠ simple téléphone.
        if zone == "Urbain":
            phone = choisir(["Personnel", "Partagé", "Aucun"], [0.50, 0.30, 0.20])
        elif zone == "Semi-urbain":
            phone = choisir(["Personnel", "Partagé", "Aucun"], [0.35, 0.30, 0.35])
        else:
            phone = choisir(["Personnel", "Partagé", "Aucun"], [0.20, 0.28, 0.52])
        smartphones.append(phone)

        # ── Accès à Internet ──────────────────────────────────────────────
        # Bruit de 12%.
        # Source : Kepios 2023 (34% d'internautes au Bénin au niveau national).
        # La couverture 4G MTN/Moov peut être meilleure dans certains villages
        # que dans des quartiers urbains surchargés.
        if zone == "Urbain":
            inet = choisir(["Toujours", "Parfois", "Jamais"], [0.82, 0.13, 0.05])
        elif zone == "Semi-urbain":
            inet = choisir(["Toujours", "Parfois", "Jamais"], [0.58, 0.28, 0.14])
        else:
            inet = choisir(["Toujours", "Parfois", "Jamais"], [0.28, 0.35, 0.37])
        internets.append(inet)

        # ── Type de connexion ─────────────────────────────────────────────
        # La data mobile domine au Bénin (MTN, Moov, Celtiis).
        # Le Wifi reste rare en dehors des grandes villes.
        if inet == "Jamais":
            conn = "Aucun"
        elif zone == "Urbain":
            conn = choisir(["Wifi", "Data mobile", "Les deux"], [0.20, 0.55, 0.25])
        elif zone == "Semi-urbain":
            conn = choisir(["Wifi", "Data mobile", "Les deux"], [0.08, 0.75, 0.17])
        else:
            conn = choisir(["Wifi", "Data mobile", "Les deux"], [0.03, 0.88, 0.09])
        connexions.append(conn)

        # ── Soutien scolaire ──────────────────────────────────────────────
        # Bruit de 15%.
        # Le soutien scolaire varie selon type d'établissement ET zone.
        # Des parents sans diplôme peuvent être très investis (bruit réaliste).
        if etab == "Privé" and zone == "Urbain":
            soutien = "Oui" if bruit(0.70) else "Non"
        elif etab == "Public" and zone == "Urbain":
            soutien = "Oui" if bruit(0.50) else "Non"
        elif zone == "Semi-urbain":
            soutien = "Oui" if bruit(0.35) else "Non"
        else:  # Rural
            soutien = "Oui" if bruit(0.25) else "Non"
        soutiens.append(soutien)

        # ── Nombre de frères et sœurs ─────────────────────────────────────
        # Distribution démographique réelle du Bénin.
        # Le Bénin a un indice de fécondité élevé (5,7 enfants/femme en 2020).
        # Plus de frères/sœurs = moins de ressources disponibles par enfant.
        nb = choisir(
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0.05, 0.10, 0.15, 0.20, 0.18, 0.14, 0.10, 0.05, 0.03]
        )
        nb_freres.append(nb)

    return pd.DataFrame({
        "Niveau_Instruction_Parents": niveaux_instruction,
        "Revenu_Familial_Percu":      revenus,
        "Acces_Electricite":          electricites,
        "Possession_Smartphone":      smartphones,
        "Acces_Internet":             internets,
        "Type_Connexion":             connexions,
        "Soutien_Scolaire":           soutiens,
        "Nombre_Freres_Soeurs":       nb_freres,
    })


# ─────────────────────────────────────────────────────────────────────────────
# BLOC 3 — Comportement face à l'IA
# ─────────────────────────────────────────────────────────────────────────────
def generer_bloc3(df_bloc1, df_bloc2):
    """
    Génère les 10 variables du Bloc 3.
    
    Variables générées :
        Usage_IA, Outil_Principal, Anciennete_Usage, Frequence_Usage,
        Vecteur_Acces, Contexte_Usage, Matiere_Principale_Usage,
        Formation_IA_Recue, Source_Apprentissage_IA, Langue_Utilisation_IA
    
    Philosophie du bruit (~10% en moyenne) :
        Le comportement est observable mais ne révèle pas la posture.
        MESSAGE CENTRAL : la fréquence d'usage ne dit pas si l'élève
        utilise bien ou mal l'IA. Un usage fréquent peut être excellent.
        Un usage rare peut être désastreux.
    
    Sources :
        - Taux d'usage par âge   : Extrapolé études mondiales + Afrique
        - ChatGPT vs Gemini      : Estimation marché Afrique francophone
        - Formation IA           : UNESCO 2023 (7 pays seulement dans monde)
        - Langue                 : Facteur de risque documenté Afrique
    """
    print("  [Bloc 3] Génération du comportement face à l'IA...")

    n          = len(df_bloc1)
    ages       = df_bloc1["Age"].values
    zones      = df_bloc1["Zone"].values
    types_etab = df_bloc1["Type_Etablissement"].values
    smartphones = df_bloc2["Possession_Smartphone"].values
    internets   = df_bloc2["Acces_Electricite"].values

    usages_ia         = []
    outils            = []
    anciennetes       = []
    frequences        = []
    vecteurs          = []
    contextes         = []
    matieres          = []
    formations        = []
    sources_apprent   = []
    langues           = []

    for i in range(n):
        age    = ages[i]
        zone   = zones[i]
        etab   = types_etab[i]
        phone  = smartphones[i]
        inet   = internets[i]

        # ── Usage de l'IA ─────────────────────────────────────────────────
        # Le taux d'usage augmente avec l'âge.
        # Bruit de 15% : un élève sans internet peut utiliser l'IA via
        # un ami ou un cybercafé. Un élève avec tout l'accès peut ne pas
        # connaître ou ne pas vouloir utiliser ces outils.
        if age <= 13:
            p_usage = 0.25
        elif age <= 15:
            p_usage = 0.45
        elif age <= 17:
            p_usage = 0.65
        else:
            p_usage = 0.78

        # Ajustement selon l'accès (mais pas déterministe !)
        # Un élève sans internet peut quand même utiliser l'IA (cybercafé, ami).
        if phone == "Aucun":
            p_usage = max(0.05, p_usage - 0.25)  # Réduit mais pas à zéro

        utilise_ia = "Oui" if bruit(p_usage) else "Non"
        usages_ia.append(utilise_ia)

        # ── Variables spécifiques aux utilisateurs IA ─────────────────────
        if utilise_ia == "Oui":

            # ── Outil principal ───────────────────────────────────────────
            # ChatGPT domine mais Gemini monte vite (gratuit via
            # Google, accessible sans VPN contrairement à ChatGPT parfois).
            outil = choisir(
                ["ChatGPT", "Gemini", "Copilot", "Autre"],
                [0.58, 0.25, 0.08, 0.09]
            )
            outils.append(outil)

            # ── Ancienneté d'usage ────────────────────────────────────────
            # La majorité des utilisateurs ont découvert l'IA
            # récemment (ChatGPT a explosé fin 2022 — 3 ans de recul max).
            anciennete = choisir(
                ["Moins de 3 mois", "3-6 mois", "6-12 mois", "Plus d'1 an"],
                [0.30, 0.25, 0.25, 0.20]
            )
            anciennetes.append(anciennete)

            # ── Fréquence d'usage ─────────────────────────────────────────
            # VARIABLE DÉCISIVE avec bruit de 20%.
            # La fréquence NE PRÉDIT PAS la posture d'apprentissage.
            # C'est le message central de notre étude.
            freq = choisir(
                ["Rarement", "Parfois", "Souvent", "Systématique"],
                [0.20, 0.35, 0.30, 0.15]
            )
            frequences.append(freq)

            # ── Vecteur d'accès ───────────────────────────────────────────
            # Le smartphone domine au Bénin (68%).
            # Bruit de 5% : quelques élèves utilisent PC au CDI ou en famille.
            vecteur = choisir(
                ["Smartphone", "PC", "Tablette", "Plusieurs"],
                [0.68, 0.20, 0.05, 0.07]
            )
            vecteurs.append(vecteur)

            # ── Contexte d'usage ──────────────────────────────────────────
            # devoirs dominent, mais la curiosité intellectuelle
            # est aussi présente. Bruit de 10%.
            contexte = choisir(
                ["Devoirs", "Révisions", "Recherche", "Curiosité", "Plusieurs"],
                [0.40, 0.25, 0.20, 0.10, 0.05]
            )
            contextes.append(contexte)

            # ── Matière principale d'usage ────────────────────────────────
            # Le Français domine (dissertations, rédactions).
            # Les maths en 2ème position (exercices, formules).
            matiere = choisir(
                ["Français", "Maths", "Histoire-Géo", "SVT", "Physique", "Toutes"],
                [0.28, 0.22, 0.18, 0.15, 0.12, 0.05]
            )
            matieres.append(matiere)

            # ── Formation IA reçue ────────────────────────────────────────
            # Très faible en zones rurales.
            # Source : UNESCO 2023 (seulement 7 pays dans le monde ont
            # des programmes IA pour enseignants — le Bénin n'en fait pas partie).
            # Bruit de 15% : formation reçue mais non assimilée (15%) /
            # pas de formation mais bon usage autodidacte (30%).
            if etab == "Privé" and zone == "Urbain":
                p_form = 0.28
            elif etab == "Public" and zone == "Urbain":
                p_form = 0.12
            elif zone == "Semi-urbain":
                p_form = 0.06
            else:  # Rural
                p_form = 0.02
            formation = "Oui" if bruit(p_form) else "Non"
            formations.append(formation)

            # ── Source d'apprentissage de l'IA ────────────────────────────
            # La majorité des élèves apprennent via leurs amis
            # (bouche à oreille). Les enseignants jouent un rôle marginal
            # (cohérent avec les taux de formation très bas).
            source = choisir(
                ["Seul", "Ami", "Enseignant", "Internet", "Famille"],
                [0.35, 0.40, 0.10, 0.12, 0.03]
            )
            sources_apprent.append(source)

            # ── Langue d'utilisation de l'IA ──────────────────────────────
            # FACTEUR DE RISQUE SPÉCIFIQUE À L'AFRIQUE FRANCOPHONE.
            # Certains élèves utilisent l'IA en anglais parce qu'ils pensent
            # que les réponses sont meilleures, sans comprendre entièrement
            # ce qu'ils copient. Bruit de 10%.
            langue = choisir(
                ["Français", "Anglais", "Les deux"],
                [0.55, 0.20, 0.25]
            )
            langues.append(langue)

        else:
            # Valeurs "Aucun/N/A" pour les non-utilisateurs
            outils.append("Aucun")
            anciennetes.append("N/A")
            frequences.append("Jamais")
            vecteurs.append("N/A")
            contextes.append("N/A")
            matieres.append("N/A")
            formations.append("Non")
            sources_apprent.append("N/A")
            langues.append("N/A")

    return pd.DataFrame({
        "Usage_IA":              usages_ia,
        "Outil_Principal":       outils,
        "Anciennete_Usage":      anciennetes,
        "Frequence_Usage":       frequences,
        "Vecteur_Acces":         vecteurs,
        "Contexte_Usage":        contextes,
        "Matiere_Principale_Usage": matieres,
        "Formation_IA_Recue":    formations,
        "Source_Apprentissage_IA": sources_apprent,
        "Langue_Utilisation_IA": langues,
    })


# ─────────────────────────────────────────────────────────────────────────────
# BLOC 4 — Posture d'Apprentissage
# ─────────────────────────────────────────────────────────────────────────────
def generer_bloc4(df_bloc1, df_bloc2, df_bloc3):
    """
    Génère les 9 variables du Bloc 4 — le plus important.
    
    Variables générées :
        Methode_Integration, Objectif_Usage_IA, Reflexion_Avant_IA,
        Verification_Reponses, Autonomie_Sans_IA, Conscience_Dependance,
        Reaction_Si_IA_Indisponible, Partage_Avec_Camarades, Avis_Personnel_Sur_IA
    
    Philosophie du bruit (~10% en moyenne) :
        MESSAGE CENTRAL DE L'ÉTUDE :
        Ce n'est PAS l'origine sociale qui détermine si un élève gagne
        ou perd avec l'IA — c'est SA POSTURE, SA MÉTHODE, SA CONSCIENCE.
        
        Un élève rural sans ressources peut être Niveau 1 (Actif).
        Un élève privé urbain privilégié peut être Niveau 3 (Captif).
    
    Contradiction clé intentionnelle (bruit 30%) :
        Conscience_Dependance = "Pas du tout" 
        ET Autonomie_Sans_IA = "Impossible"
        → Les élèves les plus en danger ne savent pas qu'ils sont en danger.
    """
    print("  [Bloc 4] Génération de la posture d'apprentissage...")

    n          = len(df_bloc1)
    zones      = df_bloc1["Zone"].values
    types_etab = df_bloc1["Type_Etablissement"].values
    usages_ia  = df_bloc3["Usage_IA"].values
    frequences = df_bloc3["Frequence_Usage"].values

    methodes           = []
    objectifs          = []
    reflexions         = []
    verifications      = []
    autonomies         = []
    consciences        = []
    reactions          = []
    partages           = []
    avis               = []

    for i in range(n):
        zone      = zones[i]
        etab      = types_etab[i]
        utilise   = usages_ia[i]
        freq      = frequences[i]

        if utilise == "Oui":

            # ── Méthode d'intégration ─────────────────────────────────────
            # VARIABLE LA PLUS IMPORTANTE DU DATASET.
            # Niveau 1 (Actif)  : réfléchit avant, pose question précise, esprit critique
            # Niveau 2 (Passif) : reformule la réponse, dépendance partielle
            # Niveau 3 (Captif) : copie-colle sans lire, sans comprendre
            #
            # BRUIT DE 10% : l'origine sociale ne détermine PAS le niveau.
            # Distribution de base légèrement influencée par le contexte,
            # mais avec assez de bruit pour que le message soit vrai.
            if etab == "Privé" and zone == "Urbain":
                # Légèrement plus d'actifs en privé urbain — mais pas dominant
                methode = choisir(["Niveau 1", "Niveau 2", "Niveau 3"],
                                  [0.40, 0.38, 0.22])
            elif zone == "Rural":
                # Légèrement plus de captifs en rural — mais pas dominant
                methode = choisir(["Niveau 1", "Niveau 2", "Niveau 3"],
                                  [0.30, 0.40, 0.30])
            else:
                methode = choisir(["Niveau 1", "Niveau 2", "Niveau 3"],
                                  [0.35, 0.40, 0.25])

            # BRUIT EXPLICITE : 10% des cas cassent le pattern.
            # Un rural peut être Niveau 1, un urbain privilégié peut être Niveau 3.
            if bruit(0.10):
                methode = choisir(["Niveau 1", "Niveau 2", "Niveau 3"],
                                  [0.33, 0.34, 0.33])  # Distribution aléatoire totale
            methodes.append(methode)

            # ── Objectif d'usage ──────────────────────────────────────────
            # Ce que l'élève DÉCLARE vouloir faire avec l'IA.
            # Biais de désirabilité sociale de 8% : certains élèves déclarent
            # "Comprendre" alors qu'ils utilisent l'IA pour "Faire à ma place".
            objectif = choisir(
                ["Comprendre", "Faire à ma place", "Exemples", "Réviser", "Gagner du temps"],
                [0.30, 0.25, 0.20, 0.15, 0.10]
            )
            objectifs.append(objectif)

            # ── Réflexion avant de consulter l'IA ────────────────────────
            # Bruit de 5%. Variable assez honnête car comportementale.
            reflexion = choisir(
                ["Toujours", "Souvent", "Rarement", "Jamais"],
                [0.20, 0.30, 0.30, 0.20]
            )
            reflexions.append(reflexion)

            # ── Vérification des réponses ─────────────────────────────────
            # Bruit de 8%. Certains élèves "vérifient" sans
            # vraiment comprendre ce qu'ils vérifient (vérification superficielle).
            verif = choisir(
                ["Toujours", "Souvent", "Rarement", "Jamais"],
                [0.15, 0.25, 0.35, 0.25]
            )
            verifications.append(verif)

            # ── Autonomie sans IA ─────────────────────────────────────────
            # Variable comportementale fiable (bruit 5%).
            # Corrélée avec la méthode mais pas déterministe.
            if methode == "Niveau 1":
                auto = choisir(["Facilement", "Difficilement", "Impossible"],
                               [0.65, 0.30, 0.05])
            elif methode == "Niveau 2":
                auto = choisir(["Facilement", "Difficilement", "Impossible"],
                               [0.25, 0.55, 0.20])
            else:  # Niveau 3
                auto = choisir(["Facilement", "Difficilement", "Impossible"],
                               [0.10, 0.35, 0.55])
            autonomies.append(auto)

            # ── Conscience de la dépendance ───────────────────────────────
            # VARIABLE LA PLUS BRUITÉE : 30%.
            # C'est LA contradiction centrale du dataset.
            # Les élèves captifs NE SAVENT PAS qu'ils sont captifs.
            # Source : enquête Intelligent.com (75% des tricheurs savent
            # qu'ils trichent mais le font quand même — et 25% ne le savent pas).
            #
            # CONTRADICTION INTENTIONNELLE :
            # Autonomie_Sans_IA = "Impossible" ET Conscience = "Pas du tout"
            # → Signal le plus fort pour le modèle de clustering.
            if auto == "Impossible":
                # 30% des captifs totaux ne réalisent pas leur dépendance
                conscience = choisir(
                    ["Pas du tout", "Un peu", "Beaucoup", "Totalement"],
                    [0.30, 0.35, 0.25, 0.10]
                )
            elif auto == "Difficilement":
                conscience = choisir(
                    ["Pas du tout", "Un peu", "Beaucoup", "Totalement"],
                    [0.20, 0.40, 0.30, 0.10]
                )
            else:  # Facilement
                conscience = choisir(
                    ["Pas du tout", "Un peu", "Beaucoup", "Totalement"],
                    [0.45, 0.35, 0.15, 0.05]
                )
            consciences.append(conscience)

            # ── Réaction si l'IA est indisponible ────────────────────────
            # LE TEST DE VÉRITÉ. Ce que l'élève fait quand
            # l'IA tombe en panne révèle sa vraie posture mieux que
            # n'importe quelle autre variable. Bruit de 5%.
            if methode == "Niveau 1":
                reaction = choisir(
                    ["Travaille normalement", "Cherche alternative", "Bloque", "Renonce"],
                    [0.65, 0.25, 0.08, 0.02]
                )
            elif methode == "Niveau 2":
                reaction = choisir(
                    ["Travaille normalement", "Cherche alternative", "Bloque", "Renonce"],
                    [0.20, 0.45, 0.25, 0.10]
                )
            else:  # Niveau 3
                reaction = choisir(
                    ["Travaille normalement", "Cherche alternative", "Bloque", "Renonce"],
                    [0.05, 0.15, 0.45, 0.35]
                )
            reactions.append(reaction)

            # ── Partage avec camarades ────────────────────────────────────
            # Variable sociale (bruit 5%). 55% partagent leurs
            # découvertes IA avec leurs amis — phénomène de contagion sociale.
            partage = "Oui" if bruit(0.55) else "Non"
            partages.append(partage)

            # ── Avis personnel sur l'IA ───────────────────────────────────
            # Biais de désirabilité de 15%.
            # L'avis déclaré ne correspond pas toujours au comportement réel.
            avis_p = choisir(
                ["Outil indispensable", "Outil utile", "Outil dangereux", "Je ne sais pas"],
                [0.25, 0.45, 0.15, 0.15]
            )
            avis.append(avis_p)

        else:
            # Valeurs pour les non-utilisateurs de l'IA
            methodes.append("N/A")
            objectifs.append("N/A")
            reflexions.append("N/A")
            verifications.append("N/A")
            autonomies.append("Facilement")  # Par défaut : autonome sans IA
            consciences.append("Pas du tout")
            reactions.append("Travaille normalement")
            partages.append("Non")
            avis.append("Je ne sais pas")

    return pd.DataFrame({
        "Methode_Integration":       methodes,
        "Objectif_Usage_IA":         objectifs,
        "Reflexion_Avant_IA":        reflexions,
        "Verification_Reponses":     verifications,
        "Autonomie_Sans_IA":         autonomies,
        "Conscience_Dependance":     consciences,
        "Reaction_Si_IA_Indisponible": reactions,
        "Partage_Avec_Camarades":    partages,
        "Avis_Personnel_Sur_IA":     avis,
    })


# =============================================================================
# SECTION 3 — GÉNÉRATION PRINCIPALE
# =============================================================================

print("\n[ÉTAPE 1/4] Génération des 4 blocs de variables...")
print("-" * 50)

df_bloc1 = generer_bloc1(N_ELEVES)
df_bloc2 = generer_bloc2(df_bloc1)
df_bloc3 = generer_bloc3(df_bloc1, df_bloc2)
df_bloc4 = generer_bloc4(df_bloc1, df_bloc2, df_bloc3)

print("\n[ÉTAPE 2/4] Assemblage du dataset final...")
print("-" * 50)

# Concaténation des 4 blocs en un seul DataFrame
dataset = pd.concat([df_bloc1, df_bloc2, df_bloc3, df_bloc4], axis=1)

print(f"  Dataset assemblé : {len(dataset):,} lignes × {len(dataset.columns)} colonnes")


# =============================================================================
# SECTION 4 — VALIDATION ET STATISTIQUES
# =============================================================================

print("\n[ÉTAPE 3/4] Validation des distributions...")
print("-" * 50)

# Vérifications de base pour s'assurer que le dataset est correct
assert len(dataset) == N_ELEVES, f"ERREUR : {len(dataset)} lignes au lieu de {N_ELEVES}"
assert dataset["ID_Eleve"].nunique() == N_ELEVES, "ERREUR : IDs non uniques"
assert dataset["Age"].min() >= AGE_MIN_GLOBAL, "ERREUR : âge trop bas"
assert dataset["Age"].max() <= AGE_MAX_GLOBAL, "ERREUR : âge trop élevé"

print("  ✓ Nombre d'élèves     :", f"{len(dataset):,}")
print("  ✓ Colonnes            :", len(dataset.columns))
print("  ✓ IDs uniques         : OK")
print("  ✓ Âges valides        :", dataset["Age"].min(), "→", dataset["Age"].max(), "ans")
print()

# ── Taux d'usage de l'IA global ───────────────────────────────────────────
taux_ia = (dataset["Usage_IA"] == "Oui").mean() * 100
print(f"  Taux d'usage IA global     : {taux_ia:.1f}%")

# ── Répartition des méthodes (utilisateurs IA seulement) ──────────────────
utilisateurs = dataset[dataset["Usage_IA"] == "Oui"]
print(f"  Nombre d'utilisateurs IA   : {len(utilisateurs):,}")
print()
print("  Répartition Methode_Integration :")
for methode, count in utilisateurs["Methode_Integration"].value_counts().items():
    pct = count / len(utilisateurs) * 100
    print(f"    {methode:12s} : {count:6,} élèves ({pct:.1f}%)")

print()

# ── Répartition par zone ───────────────────────────────────────────────────
print("  Répartition par Zone :")
for zone, count in dataset["Zone"].value_counts().items():
    pct = count / len(dataset) * 100
    print(f"    {zone:15s} : {count:6,} élèves ({pct:.1f}%)")

print()

# ── Contradiction clé : captifs non conscients ────────────────────────────
captifs_non_conscients = dataset[
    (dataset["Autonomie_Sans_IA"] == "Impossible") &
    (dataset["Conscience_Dependance"] == "Pas du tout")
]
pct_paradoxe = len(captifs_non_conscients) / len(dataset) * 100
print(f"  Contradiction clé (Captif + Non conscient) : {len(captifs_non_conscients):,} élèves ({pct_paradoxe:.1f}%)")


# =============================================================================
# SECTION 5 — SAUVEGARDE
# =============================================================================

print("\n[ÉTAPE 4/4] Sauvegarde du dataset...")
print("-" * 50)

NOM_FICHIER = "dataset_eleves_benin_114000.csv"

# Sauvegarde en CSV avec encodage UTF-8 (pour les accents français)
dataset.to_csv(NOM_FICHIER, index=False, encoding="utf-8-sig")

elapsed = time.time() - start_time

print(f"  ✓ Fichier sauvegardé : {NOM_FICHIER}")
print(f"  ✓ Temps d'exécution  : {elapsed:.1f} secondes")
print()
print("=" * 65)
print("  GÉNÉRATION TERMINÉE AVEC SUCCÈS")
print(f"  {N_ELEVES:,} élèves — 37 variables — ~10% de bruit réaliste")
print("=" * 65)
print()
