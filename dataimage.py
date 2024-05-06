from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import os
import shutil

# 輸入資料夾路徑和輸出資料夾路徑
input_folder = "C:\ma"
output_folder = "C:\maoutput"

# 建立輸出資料夾（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 定義圖像增強器
datagen = ImageDataGenerator(
    zoom_range=[0.8, 1.1],
    rotation_range=15,
    brightness_range=[0.3, 1.5]
)

# 遍歷輸入資料夾中的圖像檔案
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # 載入圖像
        img = load_img(os.path.join(input_folder, filename))
        # 將圖像轉換為數組
        img_array = img_to_array(img)

        # 將圖像數組轉換為形狀為 (1, height, width, channels) 的批次
        img_array = img_array.reshape((1,) + img_array.shape)

        # 進行圖像增強
        augmented_images = datagen.flow(img_array, batch_size=1)
        for i, augmented_image in enumerate(augmented_images):
            augmented_image = augmented_image.reshape(img_array.shape[1:])
            # 儲存增強後的圖像到輸出資料夾
            save_path = os.path.join(output_folder, f"{i+1}_{filename}")
            augmented_img = array_to_img(augmented_image)
            augmented_img.save(save_path)
            if i == 4:  # 僅儲存五個增強後的圖像
                break
# 新增txt檔案
# 輸入資料夾路徑和輸出資料夾路徑
txt_input_folder = r"C:\txtinput"
txt_output_folder = r"C:\txtoutput"

# 重複執行五次
for i in range(5):
    # 遍歷讀取資料夾中的檔案
    for filename in os.listdir(txt_input_folder):
        if filename.endswith(".txt"):
            # 原始檔案路徑
            original_file_path = os.path.join(txt_input_folder, filename)
            
            # 修改後的檔案名稱
            new_filename = f"{i+1}_{filename}"
            
            # 修改後的檔案路徑
            new_file_path = os.path.join(txt_output_folder, new_filename)
            
            # 複製原始檔案到新檔案路徑
            shutil.copy(original_file_path, new_file_path)
            
            # 建立新檔案
            new_file = open(new_file_path, "a")  # 開啟檔案以供附加寫入
            new_file.close()
            
            
            print(f"檔案 {filename} 已修改為 {new_filename}，並新增了新檔案")