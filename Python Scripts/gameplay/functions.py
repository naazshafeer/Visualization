# ITP 116, Spring 2025
# Final Project
# Name: Naazneen Shafeer Vemmerath Kulangara
# Email: vemmerat@usc.edu
# Description: This python file is used to display the functions ofr the player information

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import scipy as sci
from scipy import integrate
from scipy.integrate import solve_ivp
from astropy import units as u
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import astropy.constants as const
import math
import pickle
import pandas as pd

import os
from datetime import datetime
import shutil
import sys

initial_conditions = {
    "mass": None,
    "sep": None,
    "angle": None,
    "velocity": None
}

def display_user_menu(optionsDict):
    # print("Welcome to the Gaming Hub!")
    '''Parameter : optionsDict
    Returns: a menu option that has for example A -> Display Player Information by ID
    '''
    for key in optionsDict:
        print(key + "-> " + optionsDict[key]) #optionsDict[key] should be the values (which is the short descritption)

def choosing_IC():
    global initial_conditions
    print("You have chosen to set initial conditions for your SMBH pair. In Cartesian coordinates. The SMBHs will be placed on the x-axis with an initial separation with the intial velocity being in the y-axis (tangential to the orbital path), unless you choose to determine an angle for the initial starting point.")

    #This is for the initial mass
    # print("\n")
    # print("You will now choose the mass of your SMBH. For now, both SMBH will have equal mass. A SMBH's mass is in ranges of 1e6-9e8 * const.M_sun")
    # user_inputmass = None

    # while user_inputmass is None:
    #     user_input = input("Choose the mass of each SMBH. Input a value in the range of 1e6 to 9e8: ").strip()
        
    #     if user_input.isdigit():
    #         value = int(user_input)
    #         if 1e6 <= value <= 9e8:
    #             user_inputmass = value
    #         else:
    #             print("That value is not within the range 1e6 to 9e8. Please try again.")
    #     else:
    #         print("That's not a valid integer. Please enter digits only.")
    print("\n")
    print("You will now choose the mass of your SMBH. For now, both SMBH will have equal mass. A SMBH's mass is in ranges of 1e6-9e8 * const.M_sun")
    user_inputmass = None

    while user_inputmass is None:
        user_input = input("Choose the mass of each SMBH. Input a value in the range of 1e6 to 9e8:\n ").strip()
        
        # Allow scientific notation and decimals
        if all(char in "0123456789.eE+-" for char in user_input):
            try_parse = False
            if "e" in user_input.lower() or "." in user_input:
                try:
                    value = float(user_input)
                    try_parse = True
                except:
                    pass
            else:
                try:
                    value = int(user_input)
                    try_parse = True
                except:
                    pass

            if try_parse:
                if 1e6 <= value <= 9e8:
                    user_inputmass = value
                else:
                    print("That value is not within the range 1e6 to 9e8. Please try again.")
            else:
                print("That is not a valid number. Please enter a number in standard or scientific notation.")
        else:
            print("That's not a valid number. Please use only digits or scientific notation (e.g., 4e8).")
    

    #This is for the initial separation
    print("\n")
    print("You will now choose the initial separation of your SMBH. A SMBH pair's separation is in within the range of (200 pc - 1000 pc)")
    user_inputsep = None

    while user_inputsep is None:
        user_input = input("Choose the separation of each SMBH. Input a value in the range of (200 - 1000):\n ").strip()
        
        if user_input.isdigit():
            value = int(user_input)
            if 200 <= value <= 1000:
                user_inputsep = value
            else:
                print("That value is not within the range (200 to 1000). Please try again.")
        else:
            print("That's not a valid integer. Please enter digits only.")

    #This is for the initial angle
    print("\n")
    print("You will now choose the initial angle that the SMBHs shoot out. Usually, the SMBHs move with an initial velcoity tangential to the orbital path (90 degrees, so only in the y-axis). A SMBH's initial angle can be chosen from this list (30, 45, 60, 90) degrees")
    user_inputangle = None

    while user_inputangle is None:
        user_input = input("Choose the initial angle of each SMBH. Choose a value from this list (30, 45, 60, 90):\n ").strip()
        
        if user_input.isdigit():
            value = int(user_input)
            if value == 30:
                user_inputangle = value
            elif value == 45:
                user_inputangle = value
            elif value == 60:
                user_inputangle = value
            elif value == 90:
                user_inputangle = value
            else:
                print("That value is not from the presented list. Please try again.")
        else:
            print("That's not a valid integer. Please enter digits only.")

    # This is for the initial velocity
    print("\n")
    ideal_velocity = ((np.sqrt(const.G * (2 * (user_inputmass * const.M_sun)) / ((user_inputsep/2) * u.pc))).to(u.km / u.s))
    print(f"As per the mass and separation that you have chosen, the ideal initial velocity for one of the SMBHs in the pair is:\n {ideal_velocity:.2f}.")
    print("Since this pair will have equal mass, they will also have equal and opposite velocities.\n")

    use_custom_velocity = input("Would you like to enter your own initial velocity instead? (yes/no):\n ").strip().lower()

    if use_custom_velocity == "yes":
        user_velocity = None
        while user_velocity is None:
            user_input = input("Enter your desired initial velocity in km/s from the range (40 - 300):\n ").strip()
            
            if user_input.replace(".", "", 1).isdigit():
                value = float(user_input)
                if 40 <= value <= 300:
                    user_velocity = value * u.km / u.s
                else:
                    print("That value is not within the valid range of 40–300 km/s. Please try again.")
            else:
                print("That's not a valid number. Please enter digits only.")
    else:
        user_velocity = ideal_velocity 

    #now that we have gotten the initial conditions, let's do the return statement

    initial_conditions["mass"] = user_inputmass
    initial_conditions["sep"] = user_inputsep
    initial_conditions["angle"] = user_inputangle
    initial_conditions["velocity"] = user_velocity

    return

def display_IC():
    
    if initial_conditions["mass"] is None:
        print("You must first set initial conditions using Option A. So try again!")
        return
    print("\n")
    print("\n")
    print(f"The mass of each of the SMBH is {initial_conditions['mass'] * const.M_sun:.2e} kg or {initial_conditions['mass']} * const.M_sun.")
    print(f"The initial separation of the SMBH is {initial_conditions['sep'] * u.pc}")
    print(f"The initial angle of each of the SMBH is {initial_conditions['angle']} degrees")
    print(f"The initial velocity of each of the SMBH is {initial_conditions['velocity']:.2f}")
    print("\n")
    print("\n")

def simulation(mass_, sep_, velocity_, angle_):

    G = const.G.to(u.pc**3/(u.kg*u.yr**2)) #change the innate time to years
    m_gal = 1e11 * const.M_sun
    #Masses
    m1= mass_ * const.M_sun #mass of black hole A
    m2= mass_ * const.M_sun #mass of black hole B


    sep = sep_ * u.pc
    r1_initial=[((sep/2.).value),0,0] * u.pc
    r2_initial=[-((sep/2.).value),0,0] * u.pc
    # #To arrays

    r1=np.array(r1_initial,dtype="float64") * u.pc
    r2=np.array(r2_initial,dtype="float64") * u.pc

    #Semi-major axis
    a = (0.5 * (sep))

    # #COM
    r_com=(m1*r1+m2*r2)/(m1+m2)

   
    v_circ = (np.sqrt((G*m1*(sep/2))/(sep)**2)).value


    v1_initial1=[0,v_circ, 0] *u.pc/u.yr
    v2_inital1=[0,-v_circ,0] *u.pc/u.yr #km/s
    # #change as needed to test
    v_df_fixed = (velocity_).to(u.pc/u.yr)

    angle_rad = np.radians(angle_)


    v1_df = np.array([-v_df_fixed.value * np.cos(angle_rad), v_df_fixed.value * np.sin(angle_rad), 0])
    v2_df = np.array([v_df_fixed.value * np.cos(angle_rad), -v_df_fixed.value * np.sin(angle_rad), 0])

    # #To arrays
    v1=np.array(v1_initial1,dtype="float64") *u.pc/u.yr
    v2=np.array(v2_inital1,dtype="float64") *u.pc/u.yr
    #Find velocity of COM
    v_com=(m1*v1+m2*v2)/(m1+m2)



    def velocity_dispersion(m_gal_):
        '''This is the velocity dispersion of the galaxy.
         Calculated at km/s and then converted to pc/yr for solver
          m_gal_ = 1e11 * M_sun - in kg'''
        
        o = ((10**(2.2969)*((m_gal_ * u.kg)/ (10**(11) * const.M_sun))**(0.299)) * (u.km/u.s)).to(u.pc / u.yr)
        return o.value #as float (in pc/yr) for faster solving

    def coulomb_logarithm(r_com_, o_, G_val, m_val):
        """Calculate Coulomb logarithm, no units in this either
        r_com = distance of each body to the com (which is at [0,0,0] hence unchanging
        o_ = vel dispersion
        G_val = taking val of G to be unitless (remember that we have converted the constant to respective pc/yr requirement
        m_val = in kg ))"""

        x = (r_com_ * (o_**2)) / (G_val * m_val)
        return math.log10(x)


    def dynamical_friction_a(r_com_, v_rel_, ln_A_, G_val, m_val):
        """Return dynamical friction force, kg * pc / yr^2
        Edits : made some changes such as directly adding in the vectors so that there is no confusion that it is not opposite to the velcoity
        v_rel = should be v_1 - v_com
        v_mag = magnitude of that (to get direction in vectorized form)
        F_D_force = simplified chandrasekhar formula
        then we add vector directions and make it opposite
        and hten make it an arr so that DIRECTION is emphasized lol - add units so that conversion is neat
        and then a_df_mag (just so other units dont get confused when called again in the last function)
        """
        v_mag = np.linalg.norm(v_rel_)
        unit_v = v_rel_ / v_mag
        F_D_force = 0.428 * ln_A_ * ((G_val * m_val**(2))/r_com_**(2)) #it should be confirmed here that though it is unitless the numbers are correct

        F_D_i = -1 * (F_D_force * unit_v[0]) #ensure opposite direction
        F_D_j = -1 * (F_D_force * unit_v[1])
        F_D_k = -1 * (F_D_force * unit_v[2])

        DF_forcearr = np.array([F_D_i, F_D_j, F_D_k]) *  ((u.kg * u.pc)/(u.yr**2))

        a_df_mag = (DF_forcearr / (m_val * u.kg)).to(u.pc / (u.yr**2)).value

        return a_df_mag


    def calculate_DF(G_val, m1_val, m2_val, m_gal_val, r1_pc, r2_pc, v1_pcyr, v2_pcyr): #added units to variables so confusion is less
        """Calculate dynamical friction for both bodies
        apply everything and just to make clear there are no units. Most have been .value so that the solver only clauclates floats - maybe that is main source of error
        but the other way takes too much time for more timesteps- ensured that units are correct as far as I know"""
        r_com = (m1_val*r1_pc + m2_val*r2_pc) / (m1_val + m2_val) #should be 0 and is 0 for first time step - will recheck this later 
        v_com = (m1_val*v1_pcyr + m2_val*v2_pcyr) / (m1_val + m2_val) # same here

        o_pcyr = velocity_dispersion(m_gal_val) #pc/yr

        r1_com_pc = np.linalg.norm(r1_pc - r_com) #pc
        r2_com_pc = np.linalg.norm(r2_pc - r_com) #pc
        v1_rel_pcyr = v1_pcyr - v_com #pc / yr
        v2_rel_pcyr = v2_pcyr - v_com #pc /yr

        ln_A1 = coulomb_logarithm(r1_com_pc, o_pcyr, G_val, m1_val) #no units
        ln_A2 = coulomb_logarithm(r2_com_pc, o_pcyr, G_val, m2_val) #no units

        a_df1 = dynamical_friction_a(r1_com_pc, v1_rel_pcyr, ln_A1, G_val, m1_val) #pc/yr**2
        a_df2 = dynamical_friction_a(r2_com_pc, v2_rel_pcyr, ln_A2, G_val, m2_val) #pc/yr^2

        return a_df1, a_df2
    
    def dydt(t, y, m1, m2, G, m_gal, df_on=False):
        '''
        This is the function for the solve_ivp - same structure like odeint but htere is freedom of choosing hte methods based on what you are solving
        AREAS OF DOUBT: (aka questions to ask when rechecking #3)
        - do i have to rename the r1,r2,v1,v2 since what if they are taken in as global variables and this means that the positon and velocity is not really changing (idk how to check this)
        - is the a1_grav that is being slowed down affect the vel and the position? as in:

        a1_total in index 0 then integrate to find position and vel for next time step? really need to think about the process in which the solver takes in teh values and calculates the next timestep
        '''

        r1 = y[0:3]
        r2 = y[3:6]
        v1 = y[6:9]
        v2 = y[9:12]

        r12 = r2 - r1
        r12_mag = np.linalg.norm(r12)
        a1_grav = (G * m2 / r12_mag**3) * r12
        a2_grav = (G * m1 / r12_mag**3) * (-r12)

        a1_df = np.zeros(3)
        a2_df = np.zeros(3)
        if df_on:
            a1_df, a2_df = calculate_DF(G, m1, m2, m_gal, r1, r2, v1, v2)

        a1_total = a1_grav + (1 * a1_df)
        a2_total = a2_grav + (1 * a2_df)

        return np.concatenate([v1, v2, a1_total, a2_total])
    
    t_i = 0.0  # yr
    t_f = (20 * u.Myr).to(u.yr).value  # in yr
    t_eval = np.linspace(t_i, t_f, 400) #linspace so 400 timesteps only

    G_val = G.value  # in pc^3 / (kg * yr^2)
    m1_val = m1.to(u.kg).value
    m2_val = m2.to(u.kg).value
    m_gal_val = m_gal.value

    y01 = np.concatenate([
        r1.to_value(u.pc),
        r2.to_value(u.pc),
        v1.to_value(u.pc/u.yr),
        v2.to_value(u.pc/u.yr) #.value intake
        ])

    y0_df = np.concatenate([
        r1.to_value(u.pc),
        r2.to_value(u.pc), #.value intake
        v1_df,
        v2_df
    ])

    print("Solving without DF...")
    sol_no_df = solve_ivp(
        fun=lambda t, y: dydt(t, y, m1_val, m2_val, G_val, m_gal_val, df_on=False),
        t_span=(t_i, t_f),
        y0=y01,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-8
    )

    print("Solving with DF...")
    sol_df = solve_ivp(
        fun=lambda t, y: dydt(t, y, m1_val, m2_val, G_val, m_gal_val, df_on=True),
        t_span=(t_i, t_f),
        y0=y0_df,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-8
    )

    def unpack_solution(sol):
        r1_ = sol.y[0:3].T  # shape (N, 3)
        r2_ = sol.y[3:6].T
        v1_ = sol.y[6:9].T
        v2_ = sol.y[9:12].T
        sep = np.linalg.norm(r1_ - r2_, axis=1)
        return r1_, r2_, v1_, v2_, sep

    r1_nodf_arr, r2_nodf_arr, v1_nodf_arr, v2_nodf_arr, sep_nodf_arr = unpack_solution(sol_no_df)
    r1_df_arr, r2_df_arr, v1_df_arr, v2_df_arr, sep_df_arr = unpack_solution(sol_df)

    def compute_forces(G_, m1, m2, m_gal_, r1_arr, r2_arr, v1_arr, v2_arr, df_on=False):
        N = r1_arr.shape[0]
        a_grav1 = np.zeros((N, 3))
        a_grav2 = np.zeros((N, 3))
        a_df1 = np.zeros((N, 3))
        a_df2 = np.zeros((N, 3))
        a_total1 = np.zeros((N, 3))
        a_total2 = np.zeros((N, 3))

        for i in range(N):
            r1_ = r1_arr[i]
            r2_ = r2_arr[i]
            v1_ = v1_arr[i]
            v2_ = v2_arr[i]

            r12 = r2_ - r1_
            r12_mag = np.linalg.norm(r12)


            a1_grav = (G_ * m2 / r12_mag**3) * r12
            a2_grav = (G_ * m1 / r12_mag**3) * (-r12)


            a1_df = np.zeros(3)
            a2_df = np.zeros(3)
            if df_on:
                a1_df, a2_df = calculate_DF(G_, m1, m2, m_gal_, r1_, r2_, v1_, v2_)

            a_grav1[i] = a1_grav
            a_grav2[i] = a2_grav
            a_df1[i] = a1_df
            a_df2[i] = a2_df


            a_total1[i] = a1_grav + a1_df
            a_total2[i] = a2_grav + a2_df

        return a_grav1, a_grav2, a_df1, a_df2, a_total1, a_total2



    a_grav1_nodf, a_grav2_nodf, a_df1_nodf, a_df2_nodf, a_tot1_nodf, a_tot2_nodf = compute_forces(
        G_val, m1_val, m2_val, m_gal_val,
        r1_nodf_arr, r2_nodf_arr, v1_nodf_arr, v2_nodf_arr, df_on=False
    )


    a_grav1_df, a_grav2_df, a_df1_df, a_df2_df, a_tot1_df, a_tot2_df = compute_forces(
        G_val, m1_val, m2_val, m_gal_val,
        r1_df_arr, r2_df_arr, v1_df_arr, v2_df_arr, df_on=True
    )

    v1_nodf = ((v1_nodf_arr * u.pc / u.yr).to(u.km / u.s)).value
    v2_nodf = ((v2_nodf_arr * u.pc / u.yr).to(u.km / u.s)).value
    a_grav1_nodf_arr = ((a_grav1_nodf * u.pc / u.yr**2).to(u.km / u.s**2)).value
    a_grav2_nodf_arr = ((a_grav2_nodf * u.pc / u.yr**2).to(u.km / u.s**2)).value
    a_df1_nodf_arr = ((a_df1_nodf * u.pc / u.yr**2).to(u.km / u.s**2)).value
    a_df2_nodf_arr = ((a_df2_nodf * u.pc / u.yr**2).to(u.km / u.s**2)).value
    a_tot1_nodf_arr = ((a_tot1_nodf * u.pc / u.yr**2).to(u.km / u.s**2)).value
    a_tot2_nodf_arr = ((a_tot2_nodf * u.pc / u.yr**2).to(u.km / u.s**2)).value

    v1_df = ((v1_df_arr * u.pc / u.yr).to(u.km / u.s)).value
    v2_df = ((v2_df_arr * u.pc / u.yr).to(u.km / u.s)).value
    a_grav1_df_arr = ((a_grav1_df * u.pc / u.yr**2).to(u.km / u.s**2)).value
    a_grav2_df_arr = ((a_grav2_df * u.pc / u.yr**2).to(u.km / u.s**2)).value
    a_df1_df_arr = ((a_df1_df * u.pc / u.yr**2).to(u.km / u.s**2)).value
    a_df2_df_arr = ((a_df2_df * u.pc / u.yr**2).to(u.km / u.s**2)).value
    a_tot1_df_arr = ((a_tot1_df * u.pc / u.yr**2).to(u.km / u.s**2)).value
    a_tot2_df_arr = ((a_tot2_df * u.pc / u.yr**2).to(u.km / u.s**2)).value

    data_dict_nodf = {
    "r1": r1_nodf_arr,          
    "r2": r2_nodf_arr,          
    "v1": v1_nodf,          
    "v2": v2_nodf,          
    "sep": sep_nodf_arr,        
    "a_grav1": a_grav1_nodf_arr,    
    "a_grav2": a_grav2_nodf_arr,    
    "a_df1": a_df1_nodf_arr,        
    "a_df2": a_df2_nodf_arr,        
    "a_total1": a_tot1_nodf_arr,    
    "a_total2": a_tot2_nodf_arr,    
    "t": t_eval,
    "initial_conditions": {  # Create new dict from parameters
            "mass": mass_,
            "sep": sep_,
            "angle": angle_,
            "velocity": velocity_
        }               
    }


    data_dict_df = {
        "r1": r1_df_arr,
        "r2": r2_df_arr,
        "v1": v1_df,
        "v2": v2_df,
        "sep": sep_df_arr,
        "a_grav1": a_grav1_df_arr,
        "a_grav2": a_grav2_df_arr,
        "a_df1": a_df1_df_arr,
        "a_df2": a_df2_df_arr,
        "a_total1": a_tot1_df_arr,
        "a_total2": a_tot2_df_arr,
        "t": t_eval,
        "initial_conditions": {  # Create new dict from parameters
            "mass": mass_,
            "sep": sep_,
            "angle": angle_,
            "velocity": velocity_
        }
    }

    with open("functionsspeed6.pkl", "wb") as f:
        pickle.dump(data_dict_nodf, f)

    with open("functionsspeed6_df.pkl", "wb") as f:
        pickle.dump(data_dict_df, f)

    return data_dict_nodf, data_dict_df

def safe_save_simulation(data_dict_nodf, data_dict_df, 
                         base_folder="Python Scripts/gameplay/simulation_data",
                         backup_folder="Python Scripts/gameplay/backups"):
    '''I want to make sure that pkl files are saved regardless until i have no more use and i can reqrite over pkl files to make immediate changes
    Credit : versioning of data is not something that I have done but I have some prior experience in ITP 116 to append to files/making changes and then I ask copilot on how i can make this even better and they suggested
    some changes to the lines of code and cleaned up some portions'''
    try:

        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_full = os.path.join(script_dir, base_folder)
        backup_full = os.path.join(script_dir, backup_folder)
        os.makedirs(base_full, exist_ok=True)
        os.makedirs(backup_full, exist_ok=True)


        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nodf_name = f"sim_data_no_df_{timestamp}.pkl"
        df_name = f"sim_data_with_df_{timestamp}.pkl"
        nodf_path = os.path.join(base_full, nodf_name)
        df_path = os.path.join(base_full, df_name)


        with open(nodf_path, "wb") as f:
            pickle.dump(data_dict_nodf, f)
        with open(df_path, "wb") as f:
            pickle.dump(data_dict_df, f)


        shutil.copy2(nodf_path, os.path.join(backup_full, nodf_name))
        shutil.copy2(df_path, os.path.join(backup_full, df_name))

        print("Simulation data saved successfully.")
        print(f"Main files: {nodf_path}, {df_path}")
        print(f"Backups: {os.path.join(backup_full, nodf_name)}, {os.path.join(backup_full, df_name)}")

        return {
            'main': [nodf_path, df_path],
            'backups': [os.path.join(backup_full, nodf_name), os.path.join(backup_full, df_name)]
        }
    except Exception as e:
        print(f"Error saving simulation data: {str(e)}", file=sys.stderr)
        raise


