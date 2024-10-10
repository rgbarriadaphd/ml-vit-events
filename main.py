"""
# Author = ruben
# Date: 2/10/24
# Project: ml-vit-events
# File: main.py

Description: Entry point for ml-vit-events
"""
from classification.predict_agent import PredictImageFeaturesAgent

if __name__ == '__main__':
    agent = PredictImageFeaturesAgent('config/config.json')
    agent.run()
