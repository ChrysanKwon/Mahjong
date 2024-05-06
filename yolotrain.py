from ultralytics import YOLO
from sklearn.model_selection import ParameterGrid

space={"hsv_h": [0.015,0.023],
        "hsv_s": [0.7],
        "hsv_v": [0.4],
        "mosaic": [1],
        "degrees": [5],
        "translate": [0.1],
        "scale": [0.5],
        "shear": [0.1,0.15],
        "perspective": [0.0001,0.0002]
        }
parameter_grid = ParameterGrid(space)

if __name__ == '__main__':
    #train
    '''
    for parameters in parameter_grid:
        model = YOLO('yolov8s.pt')
        model.train(data='mahjong.yaml',epochs=500,imgsz=640,plots=True,batch=8,
                    fliplr=0,
                    hsv_h=parameters['hsv_h'],
                    hsv_s=parameters['hsv_s'],
                    hsv_v=parameters['hsv_v'],
                    mosaic=parameters['mosaic'],
                    degrees=parameters['degrees'],
                    translate=parameters['translate'],
                    scale=parameters['scale'],
                    shear=parameters['shear'],
                    perspective=parameters['perspective']
                    )

    '''
    
    #test
    for i in range(2,10):
        i
        model = YOLO('/practice/runs/detect/train'+str(i)+'/weights/best.pt')
        source=['/practice/test.png','/practice/test2.png',
                '/practice/test3.png','/practice/test4.png',
                '/practice/test5.png','/practice/test6.png',
                '/practice/test7.png','/practice/test8.png',
                '/practice/test9.png','/practice/test10.png']
        results=model.predict(source=source, conf=0.25,save=True)
    # classes
    output=results[0].boxes.cls.detach().cpu().numpy()
    print(output)
    #'''

    #train38 pre26 刪資料 degree=4 0.1 0.5 imgsz 640 mo開 perspective= 0.0001 最優
    #train39 pre27 刪資料 沒用degree 0.1 0.5 imgsz 640 mo開 perspective= 0.0001 差不多
    #train42 pre29 刪資料 沒用degree 0.1 0.5 imgsz 640 mo開 perspective= 0.0001 差不多

    #train49-227 超參數
    #pre30-208
    