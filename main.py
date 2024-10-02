"""
# Author = ruben
# Date: 2/10/24
# Project: ml-vit-events
# File: main.py

Description: Entry point for ml-vit-events
"""
from dataset.loader import EventsFeaturesDataLoader
from settings.settings_loader import Settings
from classification.predict_agent import PredictAgent


if __name__ == '__main__':
    # Import configuration
    settings = Settings('config/config.json')

    # Load dataset and apply transforms if defined
    dataframe = EventsFeaturesDataLoader(settings).get()

    # Run predictions
    agent = PredictAgent(settings, dataframe)
    results = agent.run()


