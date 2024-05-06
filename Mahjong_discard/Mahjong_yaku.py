from copy import copy
from ultralytics import YOLO
from random import sample
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

#從yolo辨識的圖片讀取手牌
model = YOLO('/practice/runs/detect/train2/weights/best.pt')
results = model.predict(source='/practice/test.png', conf=0.25)
hand=results[0].boxes.cls.detach().cpu().numpy()

#for i in range(200):
#hand 手牌
#tile_list=[]
#for i in range(34):
#    for j in range(4):
#        tile_list.append(i)
#hand=sample(tile_list,14)


#牌理解析-刻（三張一樣），輸出幾個刻子以及是哪些牌，不直接刪除輸入手牌
def Triplet(hand):
    triplet_tiles=[]
    triplet_num = 0
    hand_for_Triplet=copy(hand)
    for i in range(34):
        if hand_for_Triplet.count(i)>=3:
            triplet_tiles.append(i)            
            hand_for_Triplet.remove(i)
            hand_for_Triplet.remove(i)
            hand_for_Triplet.remove(i)
            triplet_num +=1
    return triplet_num,triplet_tiles
#牌理解析-對（兩張一樣），輸出幾個對子以及是哪些牌，不直接刪除輸入手牌
def Pair(hand):
    pair_tiles=[]
    pair_num = 0
    hand_for_pair = copy(hand)
    for i in range(34):
        if hand_for_pair.count(i)==2:
            pair_tiles.append(i)
            hand_for_pair.remove(i)
            hand_for_pair.remove(i)        
            pair_num +=1        
    return pair_num, pair_tiles
#牌理解析-順（三張連號），輸出幾個順子以及是哪些牌，不直接刪除輸入手牌
def Sequence(hand):
    sequence_tiles=[]
    sequence_num = 0
    hand_for_sequence = copy(hand)
    #筒子的順子
    for i in range(9):
        if i in hand_for_sequence:
            if i+1 in hand_for_sequence and i+2 in hand_for_sequence and i+2 <9:
                sequence_tiles.append(i)
                hand_for_sequence.remove(i)
                hand_for_sequence.remove(i+1)
                hand_for_sequence.remove(i+2)
                sequence_num+=1
    #條子的順子
    for i in range(9,18):
        if i in hand_for_sequence:
            if i+1 in hand_for_sequence and i+2 in hand_for_sequence and i+2 <18:
                sequence_tiles.append(i)
                hand_for_sequence.remove(i)
                hand_for_sequence.remove(i+1)
                hand_for_sequence.remove(i+2)
                sequence_num+=1
    #萬子的順子
    for i in range(18,27):
        if i in hand_for_sequence:
            if i+1 in hand_for_sequence and i+2 in hand_for_sequence and i+2 <27:
                sequence_tiles.append(i)
                hand_for_sequence.remove(i)
                hand_for_sequence.remove(i+1)
                hand_for_sequence.remove(i+2)
                sequence_num+=1
    return sequence_num, sequence_tiles
#牌理解析-搭(差一張成順)，輸出幾個搭子以及是哪些牌，不直接刪除輸入手牌
def tatsu(hand):
    tatsu_tiles=[]
    tatsu_num = 0
    hand_for_tatsu = copy(hand) 
    for i in range(9):
        if i in hand_for_tatsu:
            if i+1 in hand_for_tatsu and i+1 <9:
                tatsu_tiles.append([i,i+1])
                hand_for_tatsu.remove(i)
                hand_for_tatsu.remove(i+1)
                tatsu_num+=1
            elif i+2 in hand_for_tatsu and i+2 <9:
                tatsu_tiles.append([i,i+2])
                hand_for_tatsu.remove(i)
                hand_for_tatsu.remove(i+2)
                tatsu_num+=1
    #條子的搭子
    for i in range(9,18):
        if i in hand_for_tatsu:
            if i+1 in hand_for_tatsu and i+1 <18:
                tatsu_tiles.append([i,i+1])
                hand_for_tatsu.remove(i)
                hand_for_tatsu.remove(i+1)
                tatsu_num+=1
            elif i+2 in hand_for_tatsu and i+2 <18:
                tatsu_tiles.append([i,i+2])
                hand_for_tatsu.remove(i)
                hand_for_tatsu.remove(i+2)
                tatsu_num+=1
    #萬子的搭子
    for i in range(18,27):
        if i in hand_for_tatsu:
            if i+1 in hand_for_tatsu and i+1 <27:
                tatsu_tiles.append([i,i+1])
                hand_for_tatsu.remove(i)
                hand_for_tatsu.remove(i+1)
                tatsu_num+=1
            elif i+2 in hand_for_tatsu and i+2 <27:
                tatsu_tiles.append([i,i+2])
                hand_for_tatsu.remove(i)
                hand_for_tatsu.remove(i+2)
                tatsu_num+=1
    return  tatsu_num, tatsu_tiles       
                

#Thirteen_orphans國士無雙，輸出有效牌張數以及出牌
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
#Nine_gates九蓮寶燈，輸出有效牌張數以及出牌（筒條萬分開）
def Nine_gates(hand):
    #萬筒條聽9張的情況
    nine={'ninem':[18,18,18,19,20,21,22,23,24,25,26,26,26],
    'nines':[9,9,9,10,11,12,13,14,15,16,17,17,17],
    'ninep':[0,0,0,1,2,3,4,5,6,7,8,8,8]}
    countarray=[]
    discardarray=[]
    for yaku in nine:
        discard=copy(hand)
        count=0
        for tile in nine[yaku]:
            if tile in discard:
                discard.remove(tile)
                count+=1
            else:
                pass
        for tile in nine[yaku]:
            if tile in discard:
                discard.remove(tile)
                count+=1
                break
            else:
                pass
        countarray.append(count)
        discardarray.append(discard)
    return max(countarray),discardarray[countarray.index(max(countarray))]
#Four_concealed_triplets四暗刻，輸出有效牌張數以及出牌
def Four_concealed_triplets(hand):
    discard = copy(hand)
    count=0
    #將能當有效牌的刻子刪除 
    triplet_num,triplet_tiles = Triplet(discard)
    for triplet_tile in triplet_tiles:
        for i in range(3):
            discard.remove(triplet_tile)
        count += 3
    #將能當有效牌的對子刪除
    pair_num,pair_tiles= Pair(discard)
    if triplet_num + pair_num <=5:    #刻對數量小於五，直接刪
        for pair_tile in pair_tiles:                           
            discard.remove(pair_tile)
            discard.remove(pair_tile)
            count += 2
        for i in range(5-(triplet_num + pair_num)):
            discard.remove(discard[0])
        count += 5-(triplet_num + pair_num)
    else: #刻對數量大於五，合計最多刪五個
        for pair_tile in pair_tiles:
            if pair_tiles.index(pair_tile) < (5-triplet_num):
                discard.remove(pair_tile)
                discard.remove(pair_tile)                
                count += 2  
    return count, discard
#Big_three_dragons大三元，輸出有效牌張數以及出牌
def Big_three_dragons(hand):
    discard = copy(hand)
    count=0
    #先將三元牌剃除，多於三張不計
    for dragon in range(27,30):
        for i in range(discard.count(dragon)):
            if i == 3:
                break
            count += 1
            discard.remove(dragon)
    #牌理分析
    sequence_num, sequence_tiles = Sequence(discard)
    triplet_num,triplet_tiles = Triplet(discard) 
    pair_num,pair_tiles= Pair(discard)
    #考慮面子的狀況
    if sequence_num + triplet_num > 1: #刻+順大於1
        if triplet_num>=2 or (triplet_num == 1 and pair_num>=1):#2個刻子，直接取刻，沒對子的情況從刻子取對
            for i in range(3):
                discard.remove(triplet_tiles[0])
            count += 3
        elif triplet_num == 1:#1個刻子，表示順子大於1，e.g.33345，直接取順
            for sequence_tile in sequence_tiles:
                discard.remove(sequence_tile)
                discard.remove(sequence_tile+1)
                discard.remove(sequence_tile+2)   
                count += 3
                break
        elif triplet_num == 0 :#沒刻子，表示順子大於2，盡量不取含對子的順，無法則直接取1
            for sequence_tile in sequence_tiles:
                if (sequence_tile not in pair_tiles) and (sequence_tile+1 not in pair_tiles) and (sequence_tile+2 not in pair_tiles):
                    discard.remove(sequence_tile)
                    discard.remove(sequence_tile+1)
                    discard.remove(sequence_tile+2)   
                    count += 3
                    break
                else:
                    discard.remove(sequence_tiles[0])
                    discard.remove(sequence_tiles[0]+1)
                    discard.remove(sequence_tiles[0]+2)   
                    count += 3
                    break
    elif sequence_num == 1: #刻+順<=1的情況
        for sequence_tile in sequence_tiles:
            discard.remove(sequence_tile)
            discard.remove(sequence_tile+1)
            discard.remove(sequence_tile+2)   
        count += 3
    elif triplet_num == 1: #刻+順<=1的情況
        for i in range(3):
            discard.remove(triplet_tiles[0])
        count += 3
    #面子移除後，檢測搭子
    tatsu_num, tatsu_tiles = tatsu(hand)
    #刪除搭子
    if sequence_num+triplet_num==0 and tatsu_num>0:
        for tatsu_tile in tatsu_tiles:
            if (tatsu_tile[1]-tatsu_tile[0])==1:
                discard.remove(tatsu_tile[0])
                discard.remove(tatsu_tile[1])
                count += 2
                break
            else:
                discard.remove(tatsu_tile[0])
                discard.remove(tatsu_tile[1])
                count += 2
                break
    #搭子移除後，重新檢測對子     
    pair_num,pair_tiles= Pair(discard)
    #刪除對子   
    if triplet_num>=2 and pair_num ==0: #沒對子的話從刻子拿1組
        for i in range(2):
            discard.remove(triplet_tiles[1])
        count += 2
    elif sequence_num+triplet_num+tatsu_num>0 and pair_num != 0: #有面子的話取1組對
        discard.remove(pair_tiles[0])
        discard.remove(pair_tiles[0])
        count += 2
    elif sequence_num+triplet_num+tatsu_num==0: #沒面子的話，看對子數量取
        if pair_num ==0:
            discard.remove(discard[0])
            discard.remove(discard[0])
            count += 2
        elif pair_num ==1:
            discard.remove(pair_tiles[0])
            discard.remove(pair_tiles[0])
            discard.remove(discard[0])
            count += 3
        elif pair_num >1:
            discard.remove(pair_tiles[0])
            discard.remove(pair_tiles[0])
            discard.remove(pair_tiles[1])
            discard.remove(pair_tiles[1])
            count += 4
    return count, discard
#Four_winds 四喜，輸出有效牌張數以及出牌
def Four_winds(hand):
    discard = copy(hand)
    count=0
    #先將風牌剃除，多於三張不計
    for wind in range(30,34):
        for i in range(discard.count(wind)):
            if i == 3:
                break
            count += 1
            discard.remove(wind)
    #牌理分析        
    sequence_num, sequence_tiles = Sequence(discard)
    triplet_num,triplet_tiles = Triplet(discard)
    
    if count<12: #小四喜
        if sequence_num > 0: #順的情況
            for sequence_tile in sequence_tiles:
                discard.remove(sequence_tile)
                discard.remove(sequence_tile+1)
                discard.remove(sequence_tile+2)   
                count += 3
                break
        elif triplet_num > 0: #刻的情況
            for i in range(3):
                discard.remove(triplet_tiles[0])
                count += 3
                break
        if sequence_num+triplet_num==0:
            tatsu_num, tatsu_tiles = tatsu(hand)
            pair_num,pair_tiles= Pair(discard)
            if tatsu_num>0:
                for tatsu_tile in tatsu_tiles:
                    if (tatsu_tile[1]-tatsu_tile[0])==1:
                        discard.remove(tatsu_tile[0])
                        discard.remove(tatsu_tile[1])
                        count += 2
                        break
                    else:
                        discard.remove(tatsu_tile[0])
                        discard.remove(tatsu_tile[1])
                        count += 2
                        break
            elif pair_num>0:
                discard.remove(pair_tiles[0])
                discard.remove(pair_tiles[0])
                count += 2
            else:    
                discard.remove(discard[0])
                count+=1
        
    return count, discard
#All_honors 字一色，輸出有效牌張數以及出牌
def All_honors(hand):
    discard = copy(hand)
    count=0
    #先將字牌拉出
    honors=[]
    for honor in range(27,34):
        for i in range(discard.count(honor)):
            honors.append(honor)
        for i in range(honors.count(honor)):
            discard.remove(honor)

    #從字牌找刻子
    triplet_num,triplet_tiles = Triplet(honors)
    if triplet_num>0:
        for triplet_tile in triplet_tiles:
            for i in range(3):
                honors.remove(triplet_tile)
            count += 3
    #將能當有效牌的對子刪除
    pair_num,pair_tiles= Pair(honors)
    if triplet_num+pair_num <=5 :#刻對數量小於五，直接刪
        if pair_num>0:
            for pair_tile in pair_tiles:                           
                honors.remove(pair_tile)
                honors.remove(pair_tile)
                count += 2
        for i in range(5-(triplet_num + pair_num)):
            if len(honors)>0:
                honors.remove(honors[0])
                count += 1
    elif  triplet_num + pair_num >5: #刻對數量大於五，合計最多刪五個
        for pair_tile in pair_tiles:
            if pair_tiles.index(pair_tile) < (5-triplet_num):
                honors.remove(pair_tile)
                honors.remove(pair_tile)                
                count += 2
    elif triplet_num==0 and pair_num>=5:
        for pair_tile in pair_tiles:
            honors.remove(pair_tile)
            honors.remove(pair_tile)                
            count += 2
        if len(honors)>0:
            honors.remove(honors[0])
            count += 1
    for i in range(len(honors)):
        discard.append(honors[i])
    return count, discard
#All_terminals 清老頭，輸出有效牌張數以及出牌
def All_terminals(hand):
    discard = copy(hand)
    count=0
    #先將老頭牌拉出
    terminals_list = [0,8,9,17,18,26]
    terminals=[]
    for terminal in terminals_list:
        for i in range(discard.count(terminal)):
            terminals.append(terminal)
        for i in range(terminals.count(terminal)):
            discard.remove(terminal)

    #從字牌找刻子
    triplet_num,triplet_tiles = Triplet(terminals)
    if triplet_num>0:
        for triplet_tile in triplet_tiles:
            for i in range(3):
                terminals.remove(triplet_tile)
            count += 3
    #將能當有效牌的對子刪除
    pair_num,pair_tiles= Pair(terminals)
    if  triplet_num+pair_num <=5:#刻對數量小於五，直接刪
        if pair_num>0:
            for pair_tile in pair_tiles:                           
                terminals.remove(pair_tile)
                terminals.remove(pair_tile)
                count += 2
        for i in range(5-(triplet_num + pair_num)):
            if len(terminals)>0:
                terminals.remove(terminals[0])
                count += 1
    elif triplet_num>0 and triplet_num + pair_num >5: #刻對數量大於五，合計最多刪五個
        for pair_tile in pair_tiles:
            if pair_tiles.index(pair_tile) < (5-triplet_num):
                terminals.remove(pair_tile)
                terminals.remove(pair_tile)                
                count += 2
    elif triplet_num==0 and pair_num>=5:
        for pair_tile in pair_tiles:
            terminals.remove(pair_tile)
            terminals.remove(pair_tile)                
            count += 2
        if len(terminals)>0:
            terminals.remove(terminals[0])
            count += 1
    for i in range(len(terminals)):
        discard.append(terminals[i])
    return count, discard
#All_green 綠一色，輸出有效牌張數以及出牌
def All_green(hand):
    discard = copy(hand)
    count=0
    #先將綠牌拉出
    green_list = [10,11,12,14,16,28]
    greens=[]
    for green in green_list:
        for i in range(discard.count(green)):
            greens.append(green)
        for i in range(greens.count(green)):
            discard.remove(green)

    #從字牌找刻子
    triplet_num,triplet_tiles = Triplet(greens)
    if triplet_num>0:
        for triplet_tile in triplet_tiles:
            for i in range(3):
                greens.remove(triplet_tile)
            count += 3
    #將能當有效牌的對子刪除
    pair_num,pair_tiles= Pair(greens)
    if  triplet_num+pair_num <=5:#刻對數量小於五，直接刪
        if pair_num>0:
            for pair_tile in pair_tiles:                           
                greens.remove(pair_tile)
                greens.remove(pair_tile)
                count += 2
        for i in range(5-(triplet_num + pair_num)):
            if len(greens)>0:
                greens.remove(greens[0])
                count += 1
    elif triplet_num>0 and triplet_num + pair_num >5: #刻對數量大於五，合計最多刪五個
        for pair_tile in pair_tiles:
            if pair_tiles.index(pair_tile) < (5-triplet_num):
                greens.remove(pair_tile)
                greens.remove(pair_tile)                
                count += 2
    elif triplet_num==0 and pair_num>=5:
        for pair_tile in pair_tiles:
            greens.remove(pair_tile)
            greens.remove(pair_tile)                
            count += 2
        if len(greens)>0:
            greens.remove(greens[0])
            count += 1
    for i in range(len(greens)):
        discard.append(greens[i])
    return count, discard

#役種表
yakus={
    '國士無雙':Thirteen_orphans,
    '九蓮寶燈':Nine_gates,
    '大小四喜':Four_winds,
    '大三元':Big_three_dragons,
    '字一色':All_honors,
    '清老頭':All_terminals,
    '綠一色':All_green,
    '四暗刻':Four_concealed_triplets
    }

#if len(hand) ==14:
#將全部役種的有效牌與出牌存進陣列中
countarray=[]
discardarray=[]
yakuarray=[]
for yakuname in yakus:
    yakuarray.append(yakuname)
    countarray.append(yakus[yakuname](sorted(hand))[0])
    discardarray.append(yakus[yakuname](sorted(hand))[1])

#將有效牌最多的存取出來
target = yakuarray[countarray.index(max(countarray))]
count = max(countarray)
discard_num = discardarray[countarray.index(max(countarray))]

#無效牌圖片合併
def discard_img(discard_num):
    discard_img=[]
    for tile in discard_num:
        img =cv2.imread("/practice/tile/"+str(int(tile))+".png")
        discard_img.append(img)
    discard_img =np.hstack(discard_img)     
    return discard_img

#將數字轉換成中文
hand_str=[]
discard_str=[]
for i in sorted(hand):
    hand_str.append(tiles[i])
for i in discard_num:
    discard_str.append(tiles[i])

if count<=13 and 0<=count:
    print('手牌是：',hand_str,'\n目標牌型是：',target,'\n已有',count,'張有效牌\n可以打',discard_str)
    img = cv2.cvtColor(discard_img(discard_num), cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

elif count==14:
    print('手牌是：',hand_str,'\n目標牌型是：',target,'\n已有',count,'張有效牌\n你已經胡了')
else:#test
    print('有效牌計算錯誤')
#else:
    #print("手牌數量錯誤，有",len(hand),"張")
    