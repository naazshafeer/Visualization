# ITP 116, Spring 2025
# Final Project
# Name: Naazneen Shafeer Vemmerath Kulangara
# Email: vemmerat@usc.edu
# Description: This python file is used to display the functions ofr the player information

import pandas as pd

def display_user_menu(optionsDict):
    # print("Welcome to the Gaming Hub!")
    '''Parameter : optionsDict
    Returns: a menu option that has for example A -> Display Player Information by ID
    '''
    for key in optionsDict:
        print(key + "->" + optionsDict[key]) #optionsDict[key] should be the values (which is the short descritption)

def display_player(player):
    '''Parameter : player
    Returns: a player info that shows the name, id, and the total game scores
    '''
    print(player['name'] + "[#" + str(player['id']) + "]")
    print("   The game 1 score is " + str(player['game1_score']))
    print("   The game 2 score is " + str(player['game2_score']))
    print("   The game 3 score is " + str(player['game3_score']))
    totalscore = player['game1_score'] + player['game2_score'] + player['game3_score']
    print("   Total score is " + str(totalscore))

def display_smallest_values(data):
    '''Parameter : data
    Returns: This makes the game scores into a list, the user then chooses a key to which they will be able to see the smallest score
    they have gotten
    '''
    list = ['game1_score', 'game2_score', 'game3_score']
    print("Select from this list: " + str(list))
    key = ""
    user_choiceofkey = input("Enter a key: ").strip().lower()

    while user_choiceofkey not in list:
        user_choiceofkey = input("Enter a key: ").strip().lower()
    if user_choiceofkey == 'game1_score':
        key = 'game1_score'
    elif user_choiceofkey == 'game2_score':
        key = 'game2_score'
    elif user_choiceofkey == 'game3_score':
        key = 'game3_score'
    # print(data[data[key] == data[key].min()])
    min_row = data[data[key] == data[key].min()].iloc[0]
    # print(data[key].min())
    # min_row = data[data[key]] == data[key].min().iloc[0]
    

    display_player(min_row)

def display_largest_value(data):
    '''Parameter : data
    Returns: This makes the game scores into a list, the user then chooses a key to which they will be able to see the largest
    value row
    '''
    list = ['game1_score', 'game2_score', 'game3_score']
    print("Select from this list: " + str(list))
    key = ""
    user_choiceofkey = input("Enter a key: ").strip().lower()

    while user_choiceofkey not in list:
        user_choiceofkey = input("Enter a key: ").strip().lower()
    if user_choiceofkey == 'game1_score':
        key = 'game1_score'
    elif user_choiceofkey == 'game2_score':
        key = 'game2_score'
    elif user_choiceofkey == 'game3_score':
        key = 'game3_score'
    # print(data[data[key] == data[key].min()])
    max_row = data[data[key] == data[key].max()].iloc[0]
    # print(data[key].min())
    # min_row = data[data[key]] == data[key].min().iloc[0]
    display_player(max_row)


def display_player_by_ID(data):
    '''Parameter : data
    Returns: This allows the user to check the player id data of any id that they put from 1- 120 inclusive
    '''
    playerID_input = input("Enter Player ID: ").strip()
    if playerID_input.isdigit():
        playerID = int(playerID_input)
        if playerID >= 1 and playerID <= 120:
            player = data[data['id'] == playerID].iloc[0]
            display_player(player)
        else:
            print("Not Accepted.")

def display_top_scores(data):
    '''Parameter : data
    Returns: This shows the top scores and sorts it so it is kind of like a leaderboard
    '''
    print("How many top scores (max is 100) do you want to display?")
    num_scores = 0


    while num_scores < 1 or num_scores > 100:
        user_input = input("Enter a number ").strip()
        if user_input.isdigit():
            num_scores = int(user_input)

    total_scores = (data['game1_score'] + data['game2_score'] + data['game3_score']).values

    sorted_scores = sorted(list(total_scores), reverse=True)

    for i in range(min(num_scores, len(sorted_scores))):
        print(str(i+1) + ". " + str(sorted_scores[i]))

def find_players(data):
    '''Parameter : data
    Returns: This is to find the players based on the attributes and if the attributes don't match then 
    it ends
    '''
    correct_keys = ['name', 'hobby']
    print("Find players based on the following attributes: " + str(correct_keys))
    
    user_inputforkey = input("Enter a key: ").strip().lower()
    key = ""
    while user_inputforkey not in correct_keys: #I don't know if this is list comprehension but it seems like doing it the other way is causing too mnay debugging and the program is not running
        user_inputforkey = input("Enter a key: ").strip().lower()
    if user_inputforkey == 'name':
        key = 'name'
    elif user_inputforkey == 'hobby':
        key = 'hobby'
    
    search_phrase = input("Enter a search phrase: ").strip().lower()

    filtered_data = data[data[key].str.lower().str.contains(search_phrase.lower())]
    
    if len(filtered_data) == 0:
        print("No player contains '" + search_phrase + "' in key " + key)
    else:
        print("Found " + str(len(filtered_data)) + " player(s) that contain(s)" + search_phrase + " in key " + key)
        for i in range(len(filtered_data)):
            player = filtered_data.iloc[i]

            display_player(player)
            
                         



    










