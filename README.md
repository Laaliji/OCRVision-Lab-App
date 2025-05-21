# Application OCR pour Images à Faible Résolution

Cette application web permet d'extraire du texte à partir d'images à faible résolution en utilisant des techniques de traitement d'images et l'OCR.

## Fonctionnalités

- Upload d'image (JPEG/PNG)
- Dégradation contrôlée de l'image (redimensionnement, bruit, flou)
- Prétraitement de l'image (conversion en niveaux de gris, seuillage adaptatif, débruitage, amélioration de la netteté)
- Extraction de texte avec Tesseract OCR
- Affichage des résultats (image originale, dégradée, prétraitée et texte extrait)

## Prérequis

- Python 3.7 ou supérieur
- Tesseract OCR installé sur votre système

## Installation

1. Clonez ce dépôt :
```bash
git clone <https://github.com/Laaliji/LowResTextOCR.git>

```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

3. Assurez-vous que Tesseract OCR est installé sur votre système :
   - Sur Windows : Téléchargez et installez depuis https://github.com/UB-Mannheim/tesseract/wiki
   - Sur MacOS : `brew install tesseract`
   - Sur Ubuntu : `sudo apt install tesseract-ocr`

## Utilisation

1. Lancez l'application Flask :
```bash
python app.py
```

2. Ouvrez votre navigateur et accédez à `http://localhost:5000`

3. Uploadez une image, lancez le traitement et visualisez les résultats


## Aspects techniques

- **Frontend** : HTML, JavaScript, CSS avec Tailwind
- **Backend** : Python avec Flask
- **Traitement d'image** : OpenCV (cv2)
- **OCR** : Tesseract via pytesseract

## Processus de traitement d'image

1. **Dégradation** :
   - Redimensionnement à 320x240
   - Ajout de bruit gaussien
   - Application d'un flou gaussien
   - Compression JPEG forte

2. **Prétraitement** :
   - Conversion en niveaux de gris
   - Seuillage adaptatif
   - Filtrage bilatéral pour éliminer le bruit
   - Amélioration de la netteté avec un filtre de convolution

## Sécurité

- Validation des types de fichiers acceptés (JPEG/PNG uniquement)
- Utilisation de noms de fichiers générés pour éviter les collisions
- Sanitisation des entrées utilisateur 