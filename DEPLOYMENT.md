# Guide de Déploiement - Application OCR pour Images à Faible Résolution

Ce guide détaille la procédure complète pour déployer et exécuter l'application OCR sur n'importe quelle machine, du début à la fin.

## Prérequis Système

- **Système d'exploitation**: Windows, macOS ou Linux
- **Python**: Version 3.7 ou supérieure
- **Node.js**: Version 16.x ou supérieure (pour le frontend React)
- **NPM**: Version 8.x ou supérieure
- **Git**: Pour cloner le dépôt (facultatif)
- **Espace disque**: Minimum 1 Go (principalement pour les modèles et dépendances)
- **RAM**: Recommandé 4 Go minimum (8 Go pour de meilleures performances)

## Étape 1: Installation de Tesseract OCR

Tesseract OCR est un logiciel externe requis pour le fonctionnement de l'application.

### Windows
1. Téléchargez l'installateur depuis [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. Exécutez l'installation avec les options par défaut
3. Notez le chemin d'installation (par défaut `C:\Program Files\Tesseract-OCR\`)

### macOS
```bash
brew install tesseract
```

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install tesseract-ocr
```

## Étape 2: Récupération du Code Source

### Option 1: Via Git
```bash
git clone https://github.com/Laaliji/OCRVision-Lab-App.git
cd OCRVision-Lab-App
```

### Option 2: Téléchargement direct
Téléchargez et extrayez l'archive ZIP du projet.

## Étape 3: Configuration du Backend (Python/Flask)

1. Créez un environnement virtuel Python (recommandé):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

2. Installez les dépendances Python:
```bash
pip install -r requirements.txt
```

3. Configuration de Tesseract:
   - Ouvrez le fichier `app.py` dans un éditeur de texte
   - Localisez la ligne:
     ```python
     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
     ```
   - Modifiez ce chemin si nécessaire pour correspondre à l'emplacement réel de Tesseract sur votre système:
     - Windows: généralement `C:\Program Files\Tesseract-OCR\tesseract.exe`
     - macOS/Linux: laissez cette ligne en commentaire, car Tesseract est généralement dans le PATH

## Étape 4: Configuration du Frontend (React)

1. Accédez au répertoire du frontend:
```bash
cd frontend/ocr-vision-app
```

2. Installez les dépendances Node.js:
```bash
npm install
```

3. Construisez l'application frontend:
```bash
npm run build
```

## Étape 5: Lancement de l'Application

Deux options sont possibles: lancer le backend et le frontend séparément (développement) ou uniquement le backend avec le frontend précompilé (production).

### Option 1: Mode Développement (deux serveurs)

1. Terminal 1 - Backend:
```bash
# Dans le répertoire racine du projet, avec l'environnement virtuel activé
# Windows
python app.py

# macOS/Linux
python3 app.py
```

2. Terminal 2 - Frontend:
```bash
# Dans le répertoire frontend/ocr-vision-app
npm start
```

3. Accédez à l'application dans votre navigateur:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8080

### Option 2: Mode Production (serveur unique)

Si vous avez déjà construit le frontend (étape 4.3), vous pouvez simplement lancer le backend qui servira également les fichiers statiques du frontend:

```bash
# Dans le répertoire racine du projet, avec l'environnement virtuel activé
# Windows
python app.py

# macOS/Linux
python3 app.py
```

Accédez à l'application dans votre navigateur: http://localhost:8080

## Étape 6: Vérification du Fonctionnement

1. Ouvrez l'application dans votre navigateur
2. Téléchargez une image contenant du texte
3. Utilisez les options de dégradation si souhaité
4. Cliquez sur "Traiter l'image"
5. Vérifiez que le texte est correctement extrait et affiché

## Dépannage

### Problèmes avec Tesseract

- **Erreur "tesseract is not recognized"**: Vérifiez le chemin dans app.py
- **Windows**: Assurez-vous que le chemin dans app.py correspond à votre installation
- **macOS/Linux**: Vérifiez que tesseract est accessible via `which tesseract`

### Problèmes avec le Backend

- **Port déjà utilisé**: Modifiez le port dans app.py (cherchez `app.run(host='0.0.0.0', port=8080)`)
- **Erreur de module manquant**: Exécutez `pip install -r requirements.txt` à nouveau

### Problèmes avec le Frontend

- **Erreurs de compilation**: Vérifiez les versions de Node.js et NPM
- **Erreurs d'API**: Assurez-vous que le backend est en cours d'exécution et accessible

## Ressources Supplémentaires

- Documentation Tesseract: https://github.com/tesseract-ocr/tesseract
- Documentation Flask: https://flask.palletsprojects.com/
- Documentation React: https://react.dev/
- Documentation OpenCV: https://docs.opencv.org/

## Notes sur les Performances

- Les performances d'extraction de texte dépendent de la qualité des images d'entrée
- L'application peut être plus lente sur des machines avec des ressources limitées
- Le modèle de deep learning (final_best_model.h5) nécessite environ 500 Mo de RAM lors de l'exécution 