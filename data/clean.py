#This is where you will be inputting data to parse it so that it can run through conversions and be converted into a ASCii file


# ---------------------- Importing ---------------------- #
# import functools will need it for comparison but not this
import numpy
import pandas as pd
import os




# ---------------------- Functions ---------------------- #

# def parse_data_genfunc(f_in):
#     for line in f_in:
#         cols = line.strip().split()
#         yield cols


def parse_data(f_in):
    return pd.read_csv(f_in, delim_whitespace=True, header=None)

def label_cols(dataframe):
    designations = []
    for i in range(dataframe.shape[1]):
        user_designation = input("You are at column {}, what would you like to name it: ".format(i)).strip() #updating the function to show how they are going through the columns
        designations.append(user_designation)
    dataframe.columns = designations
    print("We have now updated the headers, here they are: ")

    print(dataframe.head())
    return dataframe

def choice_of_columns(dataframe, file_path):
    print("Now that you know which columns are labelled - choose the columns that you will need. The data frame will update accordingly (will make a copy).")

    user_choice = input("Would you like to minimize the amount of columns in the final product or would you like to keep all the columns? \nReply with Y if you would like all columns to be kept. \nReply with N if you would like to choose specific columns.\n> ").strip()

    if user_choice.upper() == "Y":
        print("You chose to keep all columns.")
        dataframe_copy = dataframe.copy()

    elif user_choice.upper() == "N":
        print("Here are the current headers with their indices:")
        for i, col in enumerate(dataframe.columns): #past code didnt work so i had to redo it with the help of prof gpt and prof google
            print(i, ":", col)

        user_cols = input("Select the columns you would like (numbers separated by commas): ").strip()
        selected_indices = [int(x.strip()) for x in user_cols.split(",")]
        dataframe_copy = dataframe.iloc[:, selected_indices].copy() #pandas locates which columns and makes a copy
        print("Dataframe has been updated with the selected columns.")
        # elif user_choice.upper() == "N":
        # print("Here are the headers {}, choose which you would like to keep (by column index so first column would be 0): ".format(dataframe.head()))
        # user_cols = input("Select the columns you would like (numbers separated by commas): ").strip()
        # for i in list(user_cols):
        #     final_list = list(i.strip(","))
        #     print(final_list)

    else:
        print("Invalid input. Returning original dataframe.")
        dataframe_copy = dataframe.copy()

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_folder = os.path.join("data", "Dataframe_csv") #needed help to learn how to code this - made this and it looks good
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{base_name}_minimized.csv")

    dataframe_copy.to_csv(output_path, index=False)
    print(f"Final copy of the dataframe has been saved to: {output_path}")
    return dataframe_copy

            


def main():
    file_name = input("Enter the name of the data file (in the 'data' folder): ").strip()
    file_path = os.path.join("RAWdata_txt", file_name)
    if not os.path.exists(file_path):
        print("Add your {} in /data/ folder".format(file_path))
        return
    
    df = parse_data(f_in=file_path)
    labeled_df = label_cols(df)
    final_dataframe = choice_of_columns(labeled_df, file_path)

    print(final_dataframe.head())




if __name__ == "__main__":
    main()





    
        

