from copy import copy
import numpy as np
import math
hand=[0,0,0,0,1,1,1,2,2,3,4,5,6,8]
epochs=13

#牌代號
tiles={
    0:'一筒',
    1:'二筒',
    2:'三筒',
    3:'四筒',
    4:'五筒',
    5:'六筒',
    6:'七筒',
    7:'八筒',
    8:'九筒',
    9:'一條',
    10:'二條',
    11:'三條',
    12:'四條',
    13:'五條',
    14:'六條',
    15:'七條',
    16:'八條',
    17:'九條',
    18:'一萬',
    19:'二萬',
    20:'三萬',
    21:'四萬',
    22:'五萬',
    23:'六萬',
    24:'七萬',
    25:'八萬',
    26:'九萬',
    27:'中',
    28:'發',
    29:'白',
    30:'東',
    31:'西',
    32:'南',
    33:'北',
}

#list變成one hot格式
def one_hot(hand):
    one_hot_hand=np.zeros(34*4).reshape((4,34))
    for tile in hand:
        for count in range(hand.count(tile)):
            one_hot_hand[count][tile]=1
    return one_hot_hand
#onehot變成list
def off_one_hot(one_hot_hand):
    hand=[]
    for i in range(4):
        for j in range(34):
            if one_hot_hand[i][j]==1:
                hand.append(j)
    hand=sorted(hand)
    return hand
#國士無雙有效牌
def Thirteen_orphans(hand):
    #thirteen是聽13張的情況
    thirteen = [0,8,9,17,18,26,27,28,29,30,31,32,33]
    #複製手牌來計算出牌，count是有效牌數量
    discard=copy(hand) 
    count=0
    for tile in thirteen:
        if tile in discard:
            discard.remove(tile)
            count+=1
        else:
            pass
    for tile in thirteen:
        if tile in discard:
            discard.remove(tile)
            count+=1
            break
        else:
            pass        
    return count,discard

def actor():
    actor
    for i in range(4):
        for j in range(34):
                hand.append(j)
def discard(one_hot_hand,action):
    #動作集
    all_action=np.zeros(34*4*4*34).reshape((4*34,4,34))
    one_hot_new_hand=one_hot_hand
    for i in range(4*34):
        j=math.floor(i/34)
        z=i%34
        all_action[i][j][z]=1
    discard=all_action[action,:,:]
    #移去一張
    for i in range(4):
        for j in range(34):
            if discard[i][j]==1:
                one_hot_new_hand[i][j]=0

    return one_hot_new_hand


discard(3) 
"""gamma = 0.9
for i in epochs:
    #state = environment.get_state()
    # Critic
    #state_value = crtic(state)
    state_value = Thirteen_orphans(hand)
    one_hot_hand=one_hot(hand)
    # actor
    #policy = actor(state)
    #action = policy.sample()    # 選擇動作
    #next_state, reward = environment.take_action(action)
    policy = actor(one_hot_hand)
    action = policy.sample()
    one_hot_new_hand ,reward = discard()
    new_hand = off_one_hot(one_hot_new_hand)
    # 計算 advantage
    #value_next = crtic(new_state)
    #advantage = (reward + gamma*value_next) - state_value
    value_next = Thirteen_orphans(new_hand)
    advantage = (reward + gamma*value_next) - state_value

    # 計算並最小化 loss
    loss = -1 * policy.logprob(action) * advantage
    minimize(loss)


"""