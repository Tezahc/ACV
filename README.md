# ACV
projet ACV partie 2

## Objectif de l'application
Cette application permet de détecter et d'alerter via un signal sonore en temps réel des évènements de chutes via un flux vidéo (non pré-enregistré ou pré-enregistré).

## Installation
Dans un terminal :
```
git clone https://github.com/Tezahc/ACV.git

conda create -n acv python=3.10

conda activate acv

pip install -r requirements.txt
```
## Lancement
Depuis un terminal :
python script.py [-i <input_video_file>] [-o <output_video_file>] [-p <show_landmarks_plots>]

Options :
- input_video_file : nom du fichier vidéo à analyser au format mp4 (optionnel). Si non renseigné, l'application utilise par défaut le flux vidéo de la webcam.
- output_video_file : nom du fichier vidéo analysé par l'application (optionnel).


