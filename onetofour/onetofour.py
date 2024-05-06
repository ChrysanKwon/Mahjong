import pandas as pd
import numpy as np

# 讀excel資料，用iloc讀取特定位置
df = pd.read_excel("ma.xlsx")
players=pd.Series([])
labels =pd.Series([])
def strtolist(str):
    str=str.replace("[","")
    str=str.replace("]","")
    list=str.split(",")
    return list


for row in range(df.shape[0]):

    #fourma是4人手牌
    fourma = strtolist(df.iloc[row,0])
    #fourma = np.array(df.iloc[0,0])

    player0 = []
    player1 = []
    player2 = []
    player3 = []

    #56張手牌分給4個玩家
    for tile in range(56):
        if tile % 4 == 0 :
            player0.append(fourma[tile])
        elif tile % 4 == 1 :
            player1.append(fourma[tile])
        elif tile % 4 == 2 :
            player2.append(fourma[tile])
        else :
            player3.append(fourma[tile])
    fourplayer = pd.Series([player0,player1,player2,player3])
    players = pd.concat([players,fourplayer])

    #label to list
    fourla = strtolist(df.iloc[row,1])
    fourlabel = pd.Series([fourla[0],fourla[1],fourla[2],fourla[3]])
    labels = pd.concat([labels,fourlabel])

New_ma=pd.concat([players,labels],axis=1)
New_ma.rename(columns = {0:'players', 1:'labels'}, inplace = True)
New_ma = New_ma.reset_index(drop=True)
New_ma.to_excel("Data.xlsx")