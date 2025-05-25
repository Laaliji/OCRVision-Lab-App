# Application OCR pour Images à Faible Résolution

Cette application web permet d'extraire du texte à partir d'images à faible résolution en utilisant des techniques avancées de traitement d'images et l'OCR.

## Fonctionnalités

- Upload d'images (JPEG/PNG/TIFF/BMP)
- Dégradation contrôlée de l'image (redimensionnement, bruit, flou, compression)
- Prétraitement intelligent de l'image (conversion en niveaux de gris, seuillage adaptatif, débruitage, amélioration de la netteté)
- Extraction de texte avec Tesseract OCR et deep learning
- Affichage interactif des résultats (image originale, dégradée, prétraitée et texte extrait)
- Interface utilisateur moderne et responsive

## Architecture du Projet

- **Frontend** : React.js avec Tailwind CSS
- **Backend** : Python avec Flask
- **Traitement d'image** : OpenCV (cv2)
- **OCR** : Tesseract via pytesseract et modèle de deep learning personnalisé
- **API REST** : Communication entre frontend et backend

## Prérequis Système

- **Python** : Version 3.7 ou supérieure
- **Node.js** : Version 16.x ou supérieure (pour le frontend React)
- **Tesseract OCR** : Installé sur votre système
- **RAM** : 4 Go minimum recommandé (8 Go pour de meilleures performances)

## Guide Rapide d'Installation

Pour des instructions d'installation complètes, veuillez consulter le fichier [DEPLOYMENT.md](DEPLOYMENT.md).

1. Installez Tesseract OCR sur votre système
2. Clonez ce dépôt : `git clone https://github.com/Laaliji/OCRVision-Lab-App.git`
3. Configurez l'environnement Python et installez les dépendances : `pip install -r requirements.txt`
4. Configurez et construisez le frontend React
5. Lancez l'application

## Démarrage Rapide

### Mode Développement (deux serveurs)

```bash
# Terminal 1 - Backend
python app.py

# Terminal 2 - Frontend
cd frontend/ocr-vision-app
npm start
```

### Mode Production (serveur unique)

```bash
# Construire le frontend
cd frontend/ocr-vision-app
npm run build

# Lancer le serveur backend
cd ../..
python app.py
```

Accédez à l'application dans votre navigateur : http://localhost:8080

## Processus de Traitement d'Image

1. **Dégradation d'Image** :
   - Redimensionnement à basse résolution (320x240)
   - Ajout de bruit gaussien ou sel-et-poivre
   - Application de flou
   - Compression JPEG forte

2. **Prétraitement Intelligent** :
   - Analyse d'histogramme et correction gamma
   - Égalisation de contraste adaptative (CLAHE)
   - Seuillage adaptatif avec algorithmes gaussien et moyen
   - Filtres bilatéraux et médians pour la réduction du bruit
   - Amélioration de netteté avec filtres de convolution
   - Opérations morphologiques pour nettoyer le texte

3. **Extraction de Texte** :
   - Tesseract OCR avec configuration optimisée
   - Modèle de deep learning pour reconnaissance de caractères spécifiques
   - Segmentation de caractères pour les cas difficiles

## Sécurité

- Validation des types de fichiers acceptés (JPEG/PNG/TIFF/BMP uniquement)
- Utilisation de noms de fichiers générés aléatoirement pour éviter les collisions
- Sanitisation des entrées utilisateur
- Limitation de taille de fichier (16 MB maximum)

## Déploiement Complet

Pour un guide complet de déploiement étape par étape, consultez le fichier [DEPLOYMENT.md](DEPLOYMENT.md) qui détaille :

- Installation complète sur différents systèmes d'exploitation
- Configuration du backend et du frontend
- Options de lancement
- Dépannage et solutions aux problèmes courants

## Licence

Ce projet est distribué sous licence MIT. Voir le fichier LICENSE pour plus de détails. 