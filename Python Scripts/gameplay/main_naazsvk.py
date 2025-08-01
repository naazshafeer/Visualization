#With inspiration from ITP 116 final project
# Description: This python file is the main python file that compiles everything


import gameplay
import helper
import display

def main():
    ''' Parameter : none
    Returns : the entire game!'''

    print("Welcome to the Gaming Hub!")

    player_data = helper.read_player_data()
    options_dict = helper.create_options_dict()

    quit_program = False
    while not quit_program:
        display.display_user_menu(options_dict)
        choice = helper.get_user_option(options_dict)
        print(choice)

        if choice == 'A':
            display.display_player_by_ID(player_data)
        elif choice == 'B':
            display.display_smallest_values(player_data)
        elif choice == 'C':
            display.display_largest_value(player_data)
        elif choice == 'D':
            display.display_top_scores(player_data)
        elif choice == 'E':
            display.find_players(player_data)
        elif choice == 'P':
            user_input1 = int(input("Choose index of player 1: ").strip())
            while user_input1 not in player_data.index:
                user_input1 = int(input("Choose index of player 1: ").strip())
            user_input2 = int(input("Choose index of player 2: ").strip())
            while user_input2 not in player_data.index:
                user_input2 = int(input("Choose index of player 2: ").strip())

            gameplay.play_game(player_data, player1_id = user_input1, player2_id = user_input2)
        elif choice == 'Q':
            quit_program = True
            print("Goodbye!")

if __name__ == "__main__":
    main()





    


