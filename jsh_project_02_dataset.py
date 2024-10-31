import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 랜덤시드 고정
#np.random.seed(8)

# 데이터셋 준비
train_datagen = ImageDataGenerator(
                rescale=1/255, rotation_range=20,
                width_shift_range=0.2, height_shift_range=0.2,
                shear_range=0.2, zoom_range=0.2,
                horizontal_flip=True, fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
                  'train_edit',
                  target_size=(200, 130),
                  batch_size=1,
                  class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
                 'test_edit',
                 target_size=(200, 130),
                 batch_size=1,
                 class_mode='categorical')

# 1개씩 읽어와서 추가할 리스트
x_train_list = []
y_train_list = []
x_test_list = []
y_test_list = []

# train
for i in range(13000) :
    img, label = next(train_generator)
    x_train_list.extend(img)
    y_train_list.extend(label)

# test
for i in range(130) :
    img, label = next(test_generator)
    x_test_list.extend(img)
    y_test_list.extend(label)

# numpy 배열로 변경
x_train = np.array(x_train_list)
y_train = np.array(y_train_list)
x_test = np.array(x_test_list)
y_test = np.array(y_test_list)
    
# 검증데이터 분리 
x_val = x_train[11500:]
y_val = y_train[11500:]
x_train = x_train[:11500]
y_train = y_train[:11500]

# 데이터 형변경(메모리 부족)
x_train = x_train.astype('float16')
x_test = x_test.astype('float16')
x_val = x_val.astype('float16')
'''
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_val.shape)
print(y_val.shape)
print('-' * 20)
'''
