"""
# Author = ruben
# Date: 2/10/24
# Project: ml-vit-events
# File: main.py

Description: Entry point for ml-vit-events
"""
from classification.predict_agent import PredictAgentFactory

if __name__ == '__main__':
    agent = PredictAgentFactory.get_agent(mode='clinical',
                                          config='config/config.json')
    agent.run()
