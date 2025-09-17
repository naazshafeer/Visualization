# ITP 116, Spring 2025
# Final Project
# Name: Naazneen Shafeer Vemmerath Kulangara
# Email: vemmerat@usc.edu
# Description: This python file is a helper file

import pandas as pd


# def read_player_data(csv_file = 'players.csv'):
#     '''Parameter : csv file
#     Returns: reads the file
#     '''
#     # player_info = pd.read_csv(csv_file)
#     # return player_info
#     return pd.read_csv(csv_file) #made this change after rereading the directions

def create_options_dict(textFileStr = "menu_options.txt"):
    '''Parameter : dictionary
    Returns: creates dictionary for options
    '''
    option = {}
    file = open(textFileStr, 'r')
    for line in file:
        compoenents = line.strip().split(':')
        if len(compoenents) >= 2: #reading the components and then
            key = compoenents[0].strip() #making sure that the first one is the keys in the dict
            value = compoenents[1].strip() #then making sure that the second part is the value 
            option[key] = value #attaching the fact that the key are the uppercase letters and the values are teh descriptions of the options
    return option




def get_user_option(optionsDict):
    '''Parameter : dictionary
    Returns: Asks user for options and then returns the user input
    '''
    user_input = "" #to get the menu options from the dictionary
    user_input = input("Option: ").strip().upper() #loop until it is one the keys
    while user_input not in optionsDict:
        user_input = input("Option: ").strip().upper()
    return user_input

def create_options_dictforC(textFileStr = "plot_options.txt"):
    '''Parameter : dictionary
    Returns: creates dictionary for options
    '''
    option = {}
    file = open(textFileStr, 'r')
    for line in file:
        compoenents = line.strip().split(':')
        if len(compoenents) >= 2: #reading the components and then
            key = compoenents[0].strip() #making sure that the first one is the keys in the dict
            value = compoenents[1].strip() #then making sure that the second part is the value 
            option[key] = value #attaching the fact that the key are the uppercase letters and the values are teh descriptions of the options
    return option




def get_user_optionforC(optionsDict):
    '''Parameter : dictionary
    Returns: Asks user for options and then returns the user input
    '''
    user_input = "" #to get the menu options from the dictionary
    user_input = input("Option: ").strip() #loop until it is one the keys
    while user_input not in optionsDict:
        user_input = input("Option: ").strip()
    return user_input




def play_game(data):
    '''Parameter : data
    Returns: This runs the player data and allows you to play the game
    '''
    valid_id = list(data['id'].values)
    player_1 = 0

    while player_1 < 1 or player_1 > 120:
        player_1_input = input("Enter Player ID:").strip()

        if player_1_input.isdigit():
            player_1 = int(player_1_input)
            if player_1 not in valid_id:
                player_1 = 0
    
    player_2 = 0

    while player_2 < 1 or player_2 > 120:
        player_2_input = input("Enter Player ID:").strip()

        if player_2_input.isdigit():
            player_2 = int(player_2_input)
            if player_2 not in valid_id:
                player_2 = 0
    # i was a little stuck on how to call the gameplay.py functions into here

    from gameplay import play_game
    result = play_game(data, player_1, player_2)

    if result == "win":
        data.loc[data['id'] == player_1, 'game1_score'] += 10

        winner_name = data.loc[data['id'] == player_1, 'name'].values[0]
        print(winner_name + "wins 10 points!")
    else:
        data.loc[data['id'] == player_2, 'game1_score'] += 10

        winner_name = data.loc[data['id'] == player_2, 'name'].values[0]
        print(winner_name + "wins 10 points!")





