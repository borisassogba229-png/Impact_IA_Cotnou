# Impact de l'IA sur l'Apprentissage des Élèves au Bénin
## Analyse de 114 000 élèves béninois face à l'Intelligence Artificielle

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Clustering%20%7C%20Classification-green.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

---

## Table des Matières

- [Vue d'ensemble](#vue-densemble)
- [Contexte](#contexte)
- [Dataset](#dataset)
- [Méthodologie](#méthodologie)
- [Résultats](#résultats)
- [Technologies](#technologies)
- [Auteur](#auteur)

---

##  Vue d'ensemble

Ce projet analyse l'impact de l'Intelligence Artificielle (ChatGPT, Gemini) sur les méthodes d'apprentissage de **114 000 élèves béninois** du secondaire (6ème à Terminale).

**Objectif principal :** Identifier les différents profils d'utilisation de l'IA et comprendre si l'origine sociale (zone urbaine/rurale, école publique/privée) détermine la posture d'un élève face à l'IA.

**Message central :** Ce n'est **PAS** l'origine sociale qui détermine si un élève gagne ou perd avec l'IA, mais **sa posture, sa méthode et sa conscience** de ce qu'il fait.

---

##  Contexte

### Alignement stratégique

Ce projet s'inscrit dans la **SNIAM 2023-2027** (Stratégie Nationale d'Intelligence Artificielle et des Mégadonnées du Bénin), notamment sur les cas d'usage :
- Prédiction comportementale des apprenants
- Détection de comportements à risque

### Problématique

Au Bénin aujourd'hui, des élèves de 12 ans utilisent déjà ChatGPT et Gemini :
- Certains s'en servent pour **comprendre** → Posture Active
- D'autres pour **copier** → Posture Passive ou Captive
- Certains progressent, d'autres deviennent dépendants sans s'en rendre compte

**Question de recherche :** La zone géographique (urbaine/rurale) détermine-t-elle la posture face à l'IA ?

---

##  Dataset

### Caractéristiques

- **114 000 élèves** synthétiques du secondaire béninois
- **Du collège au lycée** : 6ème → Terminale (12 à 22 ans)
- **12 départements** du Bénin
- **37 variables** organisées en 4 blocs
- **~10% de bruit réaliste** intégré volontairement

### Pourquoi synthétique ?

Il n'existe pas de données officielles béninoises sur l'usage de l'IA par les élèves. Chaque paramètre est **calibré sur des sources réelles** :

| Source | Donnée utilisée |
|--------|-----------------|
| BAC Bénin 2025 | 78 537 candidats, taux par série et département |
| Banque Mondiale 2022 | 929 000 élèves secondaire au Bénin |
| ARCEP Bénin 2023 | 93,6% pénétration mobile |
| Kepios 2023 | 34% d'internautes au Bénin |
| UNESCO 2023 | Formation enseignants |
| 20+ études académiques | Usage ChatGPT et éducation mondiale |

### Les 4 blocs de variables

| Bloc | Variables | Rôle |
|------|-----------|------|
| **Bloc 1 - Profil** | 10 variables | Décrire l'élève (âge, sexe, zone, établissement) |
| **Bloc 2 - Contexte numérique** | 8 variables | Expliquer l'accès (smartphone, internet, électricité) |
| **Bloc 3 - Comportement IA** | 10 variables | Mesurer ce qui est fait (outil, fréquence, formation) |
| **Bloc 4 - Posture** | 9 variables | Révéler comment l'élève pense (méthode, autonomie, conscience) |

### Variable clé : Methode_Integration

**3 niveaux de posture :**

1. **Niveau 1 (Actif)** : Réfléchit d'abord, pose des questions précises, esprit critique
2. **Niveau 2 (Passif)** : Reformule la réponse, dépendance partielle
3. **Niveau 3 (Captif)** : Copie-colle sans lire, ne peut plus travailler seul

### Signal fort du dataset

**3 078 élèves** sont captifs (`Autonomie_Sans_IA = "Impossible"`) mais ne savent pas qu'ils le sont (`Conscience_Dependance = "Pas du tout"`).

---

##  Méthodologie

### Approche Machine Learning

**Phase 1 : Clustering (Non Supervisé)**

Objectif : Laisser les modèles découvrir les groupes naturels

| Modèle | Résultat | Conclusion |
|--------|----------|------------|
| **K-Means (k=3)** | 3 clusters | Sépare selon contexte (zone, école), pas comportement |
| **DBSCAN** | 1 cluster | Données trop homogènes |
| **Hierarchical** | Abandonné | Trop lent (43 608 élèves) |

**Phase 2 : Classification (Supervisé)**

Objectif : Prédire la posture (Actif/Passif/Captif) à partir des autres variables

| Modèle | Test Accuracy | Overfitting | Conclusion |
|--------|---------------|-------------|------------|
| **Random Forest** | 56.45% | 43% | Overfitting sévère |
| **Logistic Regression** | 50.77% | 0% | Stable mais faible |
| **XGBoost** | **55.63%** | 14% | **Meilleur compromis** |

### Pipeline complet


Dataset (114 000 élèves, 37 variables)
    ↓
Filtrage (43 608 utilisateurs IA)
    ↓
Sélection variables (21 features pertinentes)
    ↓
Preprocessing (encodage, normalisation)
    ↓
Split Train/Test (70/30)
    ↓
Modélisation (5 modèles testés)
    ↓
Évaluation (accuracy, F1-score, feature importance)
    ↓
Interprétation


---

##  Résultats

### Statistiques clés

| Indicateur | Chiffre |
|-----------|---------|
| Élèves utilisant l'IA | **43 608 (38,3%)** |
| Sans formation IA | **108 425 (95,1%)** |
| Captifs non conscients | **3 078 élèves** |
| Bloquent sans l'IA | **16 617 élèves** |
| Apprennent via leurs amis | **17 382 (15,2%)** |
| Utilisent l'IA en anglais sans maîtrise | **8 727 (7,7%)** |

### Répartition des postures (43 608 utilisateurs)

- **Niveau 1 (Actif)** : 35,0%
- **Niveau 2 (Passif)** : 39,0%
- **Niveau 3 (Captif)** : 26,0%

### Validation du message central

**Observation clé :** Les 3 niveaux de posture sont présents dans TOUTES les zones (urbaine, semi-urbaine, rurale) avec des proportions similaires.

**Conclusion :** La zone géographique **ne détermine PAS** la posture face à l'IA. Un élève rural peut être Actif, un élève urbain peut être Captif.

### Modèle retenu : XGBoost

**Performance :**
- Test Accuracy : **55,63%**
- F1-Macro : 0,5547
- Overfitting : 14% (acceptable)

**Prédictions par classe :**
- Actif : 63,9% correctement prédits
- Passif : 52,8%
- Captif : 48,6%

**Top 5 variables les plus importantes :**
1. Autonomie_Sans_IA
2. Conscience_Dependance
3. Age
4. Nombre_Freres_Soeurs
5. Objectif_Usage_IA

---

## Technologies

### Langages et Frameworks

- **Python 3.8+**
- **pandas** : Manipulation de données
- **numpy** : Calculs numériques
- **scikit-learn** : Machine Learning
- **xgboost** : Gradient Boosting
- **matplotlib** / **seaborn** : Visualisations

### Bibliothèques ML utilisées

python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
import xgboost as xgb



---

## Visualisations

### Distribution Usage IA par Zone


Urbain       : 40,6% utilisent l'IA
Semi-urbain  : 37,0% utilisent l'IA
Rural        : 33,0% utilisent l'IA


**Interprétation :** Différence modeste (7,6 points entre urbain et rural).

### Méthode d'Intégration par Zone

Les 3 postures (Actif/Passif/Captif) sont présentes dans toutes les zones avec des proportions similaires → **Validation du message central**.

---

##  Leçons Apprises

### Ce qui a fonctionné

1. Dataset bien construit (114 000 élèves, calibré sur données réelles)
2. Preprocessing rigoureux (encodage, normalisation)
3. Plusieurs modèles testés (clustering + classification)
4. Data leakage détecté et corrigé
5. Meilleur modèle identifié (XGBoost 55,63%)

### Ce qui n'a pas fonctionné

1. Clustering inefficace (sépare selon contexte, pas comportement)
2. Performances modestes (55% loin de 80-90% espérés)
3. Overfitting important sur Random Forest (43%)
4. Variables contextuelles dominent variables comportementales

### Améliorations futures

1. **Refaire le ML avec SEULEMENT les variables comportementales** (10-11 variables au lieu de 21)
2. **Feature engineering** : Créer des variables combinées (ex: "Formation reçue ET Autonomie facile = Actif probable")
3. **Augmenter la variance** des profils extrêmes dans le dataset
4. **Tester Neural Networks** pour capturer des relations non-linéaires complexes

---

## Conclusion

**Message principal validé :** La zone géographique ne détermine PAS la posture face à l'IA. Des élèves Actifs existent partout, des élèves Captifs aussi.

**Performance ML :** 55,63% avec XGBoost (mieux que hasard 33%, mais modeste). Indique que prédire le comportement humain à partir de variables déclaratives reste un problème difficile.

**Impact potentiel :** Ce projet peut servir de base pour :
- Identifier les élèves à risque de dépendance à l'IA
- Cibler les zones/écoles nécessitant une formation IA
- Éclairer les décideurs politiques (alignement SNIAM 2023-2027)

---

##  Auteur

**ASSOGBA BORIS**  
Étudiant en Data Science  
[data-python-boris-aa  ](https://www.linkedin.com/in/data-python-boris-assogba/) 


---

## Remerciements

- **Ministère du Numérique du Bénin** pour la SNIAM 2023-2027
- **Données officielles** : BAC Bénin 2025, Banque Mondiale, ARCEP, UNESCO
- **Études scientifiques** : 20+ publications sur ChatGPT et éducation

---

##  Références

- SNIAM 2023-2027 (Stratégie Nationale IA Bénin)
- BAC Bénin 2025 : 78 537 candidats
- Banque Mondiale 2022 : 929 000 élèves secondaire Bénin
- ARCEP Bénin 2023 : Pénétration mobile et internet
- UNESCO 2023 : Formation enseignants Afrique subsaharienne
- Ravšelj et al. 2024 : Étude mondiale ChatGPT éducation (23 218 étudiants, 109 pays)
