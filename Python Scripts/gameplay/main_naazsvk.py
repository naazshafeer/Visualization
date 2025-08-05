#With inspiration from ITP 116 final project
# Description: This python file is the main python file that compiles everything


import gameplay
import helper
import functions
import os

def main():
    ''' Parameter : none
    Returns : the entire game!'''
    print("\n")
    print("\n")
    print("         Welcome to the Dynamical Friction Simulator         ")
    print(" -> This simulator aims to visualize a supermassive black hole binary system and how it comes to merge.")
    print(" -> When two galaxies, that contain supermassive black holes (SMBH) within them, merge so do the SMBHs.")
    print(" -> The dominant component that brings them together is called dynamical friction!")
    print(" -> This is a physical phenomena that occurs to a massive object (such as the SMBH) as it travels through space, it incrementally slows down due to the gravitational interactions from nearby stars.")
    print(" -> This simulation will be able to visualize your choice of SMBH pairs and how it comes closer together. You are able to request various different types of plots as well, a few examples are:")
    print("  ~ Orbital Path over Time ~")
    print("  ~ Velocity over Time, with DF and no DF ~")
    print("  ~ Dynamical Friction Acceleration over Time ~")
    print(" -> and many more!")



    options_dict = helper.create_options_dict()

    quit_program = False


    print("\n")
    print("\n")
    print("Intructions to use the simulator:")
    print("\n")
    print("• Remember to ALWAYS start with 'A'. This allows for the system to recognize the initial conditions, without this the program will crash.")
    print("• And ALWAYS finish with 'Q' as it will close the program, allowing for you to test out another initial condition. I am still working on how to ensure a clean methodology to try many different IC's in one run, but this makes it easier on your end to not confuse ICs")
    print("\n")
    print("\n")
    
    while not quit_program:
        functions.display_user_menu(options_dict)
        choice = helper.get_user_option(options_dict)
        print(choice)

        if choice == 'A':
            functions.choosing_IC()
        elif choice == 'B':
            if None in functions.initial_conditions.values():
                print("Please set initial conditions first (Option A)")
                continue
                
            print(f"\nRunning simulation with:")
            print(f"Mass: {functions.initial_conditions['mass']} * const.M_sun")
            print(f"Separation: {functions.initial_conditions['sep']} pc")
            print(f"Angle: {functions.initial_conditions['angle']} degrees")
            print(f"Velocity: {functions.initial_conditions['velocity']:.2f}")
            
            try:
                data_nodf, data_df = functions.simulation(
                    mass_=functions.initial_conditions['mass'],
                    sep_=functions.initial_conditions['sep'],
                    velocity_=functions.initial_conditions['velocity'],
                    angle_=functions.initial_conditions['angle']
                )
                
                gameplay.plot_orbits(data_nodf, data_df)
                
            except Exception as e:
                print(f"\nSimulation error: {str(e)}")
        elif choice == 'C':
            options_dictC = helper.create_options_dictforC()
            back_to_main_menu = False
            while not back_to_main_menu:
                functions.display_user_menuforoptionC(options_dictC)
                choiceforC = helper.get_user_optionforC(options_dictC)
                print(choiceforC)

                if choiceforC == '1':
                    gameplay.plot_velocity(data_df, data_df)
                elif choiceforC == '2':
                    gameplay.plot_gravitational_acceleration_vs_time(data_df)
                elif choiceforC == '3':
                    gameplay.plot_position_vs_time(data_df)
                elif choiceforC == '4':
                    gameplay.plot_df_acceleration_vs_time(data_df)
                elif choiceforC == '5':
                    gameplay.plot_separation_vs_time_bh1(data_df)
                elif choiceforC == '6':
                    gameplay.plot_separation_vs_time_bh2(data_df)
                elif choiceforC == '7':
                    back_to_main_menu = True
                    print("Going back to main menu")
        elif choice == 'D':
            functions.display_IC()
        elif choice == 'Q':
            quit_program = True
            print("Thank you for using this simulator. If you would like to try again, please rerun this python script")

if __name__ == "__main__":
    main()





    


