from tqdm import tqdm
from datetime import date
import utils
import pandas as pd
import numpy as np

MEAN_ELO = 1500
ELO_WIDTH = 400
K_FACTOR = 64

@utils.timerun
def add_elo(concat_to):
    data = pd.read_csv('FullGamesFinal.csv')
    data['Date'] = pd.to_datetime(data['Date'])

    s = 0
    full_arrays = []    

    for season in tqdm(data['SeasonID'].unique()):
        mask = (data['SeasonID'] == season)
        filtered_data = data.loc[mask].reset_index(drop=True)
        filtered_data.drop(filtered_data.columns[[1, 6, 7, 8]], axis = 1, inplace = True)

        if s > 0:
            current_elos = update_new_season(current_elos)

        else:
            n_teams = len(filtered_data['Away'].unique())
            current_elos = np.ones(shape=(n_teams)) * MEAN_ELO

        map_df = filtered_data.drop(filtered_data.columns[[0, 2, 4]], axis = 1)
        teams = map_df.stack().unique()
        teams.sort()
        f = pd.factorize(teams)
        codes = pd.Series(f[0], f[1])
        map_teams = map_df.stack().map(codes).unstack()

        filtered_data['Outcome'] = np.where(filtered_data['H-Pts'] > filtered_data['A-Pts'], 1, 0)

        final = pd.concat([map_teams, 
                           filtered_data['Outcome']], 
                           axis=1, 
                           join='outer')

        s += 1

        for i in final.index:
            arr = []

            away_team = final.at[i, 'Away']
            home_team = final.at[i, 'Home']
            outcome = final.at[i, 'Outcome']

            a_start_elo = current_elos[away_team]
            h_start_elo = current_elos[home_team]

            a_end_elo, h_end_elo = adjust_elo(a_start_elo, h_start_elo, outcome)

            current_elos[away_team] = a_end_elo
            current_elos[home_team] = h_end_elo
                
            arr.extend([round(a_start_elo,2), round(h_start_elo,2)])
            full_arrays.append(arr)

    df = pd.DataFrame(full_arrays, columns=['A_ELO', 'H_ELO'])
    final = pd.concat([concat_to, df], axis=1, join='outer')    

    return final    

def adjust_elo(a_elo, h_elo, outcome):
    expected_win = expected(a_elo, h_elo, outcome)
    change_in_elo = K_FACTOR * (1-expected_win)

    if outcome == 1:
        h_elo += change_in_elo
        a_elo -= change_in_elo
    else:
        h_elo -= change_in_elo
        a_elo += change_in_elo

    return a_elo, h_elo

def expected(a_elo, h_elo, outcome):
    if outcome == 1:
        expected = 1.0/(1+10**((a_elo - h_elo)/ELO_WIDTH))
    else:
        expected = 1.0/(1+10**((h_elo - a_elo)/ELO_WIDTH))

    return expected

def update_new_season(elos):
    #  Instead of starting from scratch each season, this uses mean regression to shift teams towards base ELO

    diff_from_mean = elos - MEAN_ELO
    elos -= diff_from_mean/3

    return elos

def get_massey(data, season_id, game_date):
    #  Filters the data so that it only calculates rating based off season and date provided
    mask = (data['SeasonID'] == season_id) & (data['Date'] < game_date)
    filtered_data = data.loc[mask].reset_index(drop=True)
    filtered_data.drop(filtered_data.columns[[1, 3, 5, 6, 7]], axis=1, inplace=True)

    #  Converts team name to numeric codes for the algorithm
    map_df = filtered_data.drop(filtered_data.columns[[0,3]], axis = 1)
    teams = map_df.stack().unique()
    teams.sort()
    f = pd.factorize(teams)
    codes = pd.Series(f[0], f[1])
    map_teams = map_df.stack().map(codes).unstack()

    #  Creates a new DataFrame with Team Names replaced with Team Codes
    #  Team and Game Counts will shape the equation matrixes
    final = pd.concat([map_teams, filtered_data['MOV']], axis=1, join='outer')
    team_count = len(teams)
    game_count = len(filtered_data['Home'])
    
    #  Setting up the right side matrix for the equation
    #  The +1 to team count is to account for adding Home Court Advantage
    R = np.zeros([game_count, team_count+1])

    for i, game in final.iterrows():

        #  Fills Matrix with numeric values representing which teams played in the game
        #  If the order seems awkward, it's because in my dataset the away team is listed before the home team
        R[i, int(game[1])] = 1
        R[i, int(game[0])] = -1

        #  Appends another "team" representing Home Court Advantage for calculating adjusted rating
        R[i, team_count] = 1      

    #  Setting up the left side matrix for the equation
    L = np.zeros([game_count])

    for i, game in final.iterrows():
        #  Adds score margin for game in matrix
        L[i] = game[2]

    #  In order to avoid an unsolvable equation, connectivity is added
    #  In this case, I'm using a variant of connectivity by adding a fake "game" so actual ratings aren't adjusted
    #  The appended 0 is to keep this fake "game" factored out of Home Court Advantage as well
    connectivity = np.ones(team_count)
    connectivity = np.append(connectivity, 0)
    R = np.vstack((R, connectivity))
    L = np.append(L, 0)

    #  Solves equations using a least squared algorithm
    solutions = np.linalg.lstsq(R, L, rcond=None)[0]
    ratings = list(zip(teams, solutions))

    massey = pd.DataFrame(ratings, columns=['Name', 'Rating']).groupby('Name')

    return massey

@utils.timerun
def add_massey(concat_to):
    '''
    Calculates Massey Ratings for all seasons in Training File
    Concats Massey Ratings to provided file (usually schedule or raw stats)
    '''

    data = pd.read_csv('FullGamesFinal.csv')
    data['Date'] = pd.to_datetime(data['Date'])

    check_date = date.today()
    full_arrays = []

    for season in tqdm(data['SeasonID'].unique()):
        j = 0
        mask = (data['SeasonID'] == season)
        filtered_data = data.loc[mask].reset_index(drop=True)

        for i in filtered_data.index:
            arr = []

            if j < 20:
                arr.extend([0, 0])
                full_arrays.append(arr)
                j += 1
                continue

            else:
                game_date = filtered_data.at[i, 'Date']

                if check_date != game_date:
                    check_date = game_date
                    date_mask = (filtered_data['Date'] < game_date)
                    dated_data = filtered_data.loc[date_mask].reset_index(drop=True)
                    massey = get_massey(dated_data, season, game_date)

                    away_rating = massey.get_group(filtered_data.at[i, 'Away'])['Rating']
                    home_rating = massey.get_group(filtered_data.at[i, 'Home'])['Rating']

                    arr.extend([float(away_rating), float(home_rating)])
                    full_arrays.append(arr)

                else:
                    away_rating = massey.get_group(filtered_data.at[i, 'Away'])['Rating']
                    home_rating = massey.get_group(filtered_data.at[i, 'Home'])['Rating']

                    arr.extend([float(away_rating), float(home_rating)])
                    full_arrays.append(arr)

    df = pd.DataFrame(full_arrays, columns=['A_Massey', 'H_Massey'])
    final = pd.concat([concat_to, df], axis=1, join='outer')

    return final

def current_massey(data, season_code):
    '''
    Gets Massey Ratings for the current season up to the most recent played game
    '''
    check_date = date.today()
    full_arrays = []
    j = 0

    for i in data.index:
        arr = []

        if j < 20:
            arr.extend([0, 0])
            full_arrays.append(arr)
            j += 1
            continue

        else:
            game_date = data.at[i, 'Date']

            if check_date != game_date:
                check_date = game_date
                date_mask = (data['Date'] < game_date)
                dated_data = data.loc[date_mask].reset_index(drop=True)
                massey = get_massey(dated_data, season_code, game_date)

                away_rating = massey.get_group(data.at[i, 'Away'])['Rating']
                home_rating = massey.get_group(data.at[i, 'Home'])['Rating']

                arr.extend([data.at[i, 'Away'], float(away_rating)])
                full_arrays.append(arr)

                arr = []

                arr.extend([data.at[i, 'Home'], float(home_rating)])
                full_arrays.append(arr)

            else:
                away_rating = massey.get_group(data.at[i, 'Away'])['Rating']
                home_rating = massey.get_group(data.at[i, 'Home'])['Rating']

                arr.extend([data.at[i, 'Away'], float(away_rating)])
                full_arrays.append(arr)

                arr = []

                arr.extend([data.at[i, 'Home'], float(home_rating)])
                full_arrays.append(arr)

    df = pd.DataFrame(full_arrays, columns=['Name', 'Massey'])
    df['Name'] = np.where(df['Name'] == 'Los Angeles Clippers', 'LA Clippers', df['Name'])

    cur_massey = df.sort_index(axis=0, ascending=False)
    cur_massey = cur_massey.groupby('Name').head(1).reset_index(drop=True)
    cur_massey.drop(cur_massey.tail(1).index, inplace=True)
    cur_massey['Conf'] = cur_massey['Name'].map(utils.conf_dict)
    cur_massey = cur_massey.sort_values('Massey', ascending=False).reset_index(drop=True)    
    cur_massey.to_sql('Current_Massey', utils.engine, if_exists='replace', index=False)

    massey_ratings = df.groupby('Name')

    return massey_ratings

if __name__ == '__main__':
    # data = pd.read_csv('FullGamesFinal.csv')
    # season_id = '2021-22'
    # game_date = '2022-05-01'

    # massey = get_massey(data, season_id, game_date)
    #add_massey()
    add_elo()
