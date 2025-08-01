# ITP 116, Spring 2025
# Final Project
# Name: Naazneen Shafeer Vemmerath Kulangara
# Email: vemmerat@usc.edu
# Description: This python file is a gameplay file taken from lab 7

#original code taken from lab 7 and then modifying with three new functions

import random as rand
import helper


def read_data(filename = "pokemons.csv"): #first function as defined in class
    '''Parameter : filenmae
    Returns: this reutrns the pokemon list after it has been read
    '''
    pokemon_list = []
    fileIn = open(filename, "r")
    header = fileIn.readline()
    for line in fileIn:
        line = line.strip() #remove the whitespace
        line_list = line.split(",") #remove the commas
        pokemon_list.append(line_list)
    fileIn.close()
    return pokemon_list
    #return pokemon_list is a list of lists. (from class)
def pokemon_location(pokemon_data):
    '''Parameter : pokemon_data
    Returns: This asks you what location and then sorts the list
    '''
    empty_list = []
    locations = ["cave", "water", "forest"]
    prompt = "Please choose a location: " + ", or ".join(locations) + ". " #the join function
    user_input = input(prompt).strip().lower()
    while user_input not in locations:
        user_input = input(prompt).strip().lower()
    for pokemon in pokemon_data:
        if pokemon[3] == user_input: #if pokemon locations matches the user_input locations
            empty_list.append(pokemon)
    return empty_list, user_input #returns the empty_list and user_input
    #returns two things
def write_data(collected, location):
    '''Parameter : collected, location
    Returns: This outputs the collected pokemon into one file
    '''
    filename = location + ".txt"
    fileOut = open(filename, "w")
    for pokemon in collected:
        prompt = "Collected pokemon "+ pokemon[1] + " with type " + pokemon[2] + "." #changed from commas to plus and then added a period at the end to complate the sentences
        print(prompt, file = fileOut)
    fileOut.close()
    #no return 
def assign_pokemon_val(pokemon_list):
    '''Parameter : list
    Returns: This assigns a random value to the pokemon list
    '''
    for pokemon in pokemon_list:
        pokemon.append(rand.randint(10,100))
    return pokemon_list

last_pokemon = ""
last_type = ""

def catch_list_create(pokemon_count):
    '''Parameter : pokemon_count
    Returns: This makes a catch_list and then assigns a 50% chance of catching the pokemon
    '''
    index_all = list(range(pokemon_count))
    catch_count = pokemon_count // 2 #making a 50% chance that someone will catch the pokemon
    return rand.sample(index_all, catch_count)

def player_catch(pokemon_list, catch_list, player_id, player_data, last_pokemon="", last_type=""):
    '''Parameter : collected all the lists to play the catch and to see if the player catches it
    Returns: this assings a list of points to the player and then outputs that, it also stores the last pokemon so that points can be deducted or increased
    '''
    rounds = 0 #had some problem here I checked to see if it could loop
    while rounds < 4:
        print(str(player_data['name'].iloc[player_id]) + "'s turn: ")

        points = 0 #starting from nothing
        choice = -1 #initalizing variable
        # user_choice = input("Choose a number (0-" + str(len(pokemon_list)-1) + "): ").strip()
        while choice < 0 or choice >= len(pokemon_list):
            user_choice = input("Choose a number (0-" + str(len(pokemon_list)-1) + "): ").strip()
            if user_choice.isdigit():
                choice = int(user_choice)
        
        if choice in catch_list:
            caught_pokemon = pokemon_list[choice]
            points = caught_pokemon[4]
            if last_pokemon == caught_pokemon[1]:
                points = points // 2
                print("You have caught the same pokemon... so your points are halved!")
            elif last_type == caught_pokemon[2]:
                points *= 2
                print("Yo! You caught the same type, so points are doubled!")
            print("You caught " + caught_pokemon[1] + "! And you got " + str(caught_pokemon[4]) + " points." )
            rounds += 1

            return points, caught_pokemon[1], caught_pokemon[2]
        else:
            print("You have caught no pokemon in this turn... try harder")
            rounds += 1
            return 0, "", ""  

def play_game(player_data, player1_id, player2_id): #had to modify this for main_nsvk.py had to reorderplayer_id and playerdata to make sure that the name was outputted for the id chosen from the csv
    '''Parameter : data, id
    Returns: This runs everything and allows you to play the game
    '''
    player1 = player_data[player_data['id'] == player1_id].iloc[0]
    player2 = player_data[player_data['id'] == player2_id].iloc[0]

    pokemon_data = read_data()
    pokemon_data = assign_pokemon_val(pokemon_data)

    location_pokemon, location = pokemon_location(pokemon_data)
    catch_list = catch_list_create(len(location_pokemon))

    last_p1_pokemon, last_p1_type = "","" #track and then remove so that we can append the points 
    last_p2_pokemon, last_p2_type = "","" 

    player1_points,last_p1_pokemon, last_p1_type = player_catch(location_pokemon, catch_list, player1_id, player_data, last_p1_pokemon, last_p1_type )
    player2_points, last_p2_pokemon, last_p2_type = player_catch(location_pokemon, catch_list, player2_id, player_data, last_p2_pokemon, last_p2_type)

    if player1_points > player2_points:
        player_data.loc[player_data['id'] == player1_id, 'game1_score'] += 10
        print(player1['name'] + " wins 10 points!")
        return "win"
    else:
        player_data.loc[player_data['id'] == player2_id, 'game1_score'] += 10
        print(player2['name'] + " wins 10 points!")
        return "lose" 