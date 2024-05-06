# 麻將役滿牌推薦
會以最大有效牌判斷，距離何種役滿牌最近

## 資料集dataset
#### 牌資料集
https://github.com/Camerash/mahjong-dataset
#### 圖片資料生成
https://github.com/black-desk/mahjim?tab=readme-ov-file
#### 數據資料生成.
https://github.com/limichange/majiang-pai-data-tool

## 模型訓練training
本地使用yolotrain.py，內有data_augment參數
也可直接使用mahjong.ipynb

## 具體使用using
使用Mahjong_discard\Mahjong_yaku.py
目前只可輸入圖片，change#48 png

## onetofour.py（後期專案沒有使用）

將 https://github.com/limichange/majiang-pai-data-tool 生成的數據做整理
將4副牌(ma.xlsx)整理成4個1副牌(data.xlsx)

ma的格式
|theme|label|
|--|--|
|[0,0,2,30,0,8,2,30,0,9,2,30,8,17,12,32,8,18,12,32,8,26,12,32,9,30,25,33,9,31,25,33,9,32,25,33,17,33,15,27,17,27,15,27,17,28,15,27,18,29,31,28,18,29,31,28]|[0,1,2,3]|

