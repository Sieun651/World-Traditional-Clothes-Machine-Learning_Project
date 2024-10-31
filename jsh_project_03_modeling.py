#import jsh_project_02_dataset
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Flatten, Dropout, BatchNormalization
from skimage import io, transform
from tensorflow.keras.callbacks import EarlyStopping
'''
x_train = jsh_project_02_dataset.x_train
y_train = jsh_project_02_dataset.y_train
x_test = jsh_project_02_dataset.x_test
y_test = jsh_project_02_dataset.y_test
x_val = jsh_project_02_dataset.x_val
y_val = jsh_project_02_dataset.y_val  

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_val.shape)
print(y_val.shape) 
print('-' * 20)

# 2. 모델 구성하기
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(200, 130, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(13, activation='softmax'))



model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', 
                 input_shape=(200, 130, 3)))
#model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
#model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(13, activation='softmax'))


# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

# 4. 모델 학습하기
# 모델 조기종료 객체 만들기
early_stopping = EarlyStopping(monitor='val_accuracy', patience=12)

hist = model.fit(x_train, y_train, epochs=100, 
                 validation_data=(x_val, y_val),
                 callbacks=[early_stopping])

# 5. 모델 학습과정 살펴보기
plt.rcParams['figure.figsize'] = (10, 6)

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='loss')
loss_ax.plot(hist.history['val_loss'], 'g', label='val_loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='accuracy')
acc_ax.plot(hist.history['val_accuracy'], 'r', label='val_accuracy')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()

# 6. 모델 평가하기
scores = model.evaluate(x_test, y_test)
print('loss :', scores[0])
print('acc : %.2f%%' %(scores[1]*100))
print('-' * 20)

# 모델 저장
model.save('project.h5')
print('성공')


# 모델 로드
model = load_model('project.h5')

# 7-1. 모델 사용하기
x_test = x_test.astype('float32')

yhat = model.predict(x_test)

# 8) 정답데이터와 예측값을 일부 시각화
plt.rcParams['figure.figsize'] = (12, 12)
fig, ax_arr = plt.subplots(5, 5)

for i in range(25) :
    sub_plt = ax_arr[i//5, i%5] # 보조창 1개 선택
    sub_plt.axis('off')
    sub_plt.imshow(x_test[i])
    title = 'R: %s, P: %s' %(np.argmax(y_test[i]),
                             np.argmax(yhat[i]))
    sub_plt.set_title(title)

plt.show()
'''

# 모델 로드
model = load_model('project.h5')

# 7-2. (시연)모델 사용하기
s_datagen = ImageDataGenerator(rescale=1/255)

ss_generator = s_datagen.flow_from_directory(
    '시연자료',
    target_size=(200, 130), 
    batch_size=1,
    class_mode='categorical'
)

# 이미지 및 레이블 리스트 초기화
p_list = []
pl_list = []
for i in range(13):
    img, label = next(ss_generator)
    p_list.append(img[0])
    pl_list.append(label[0])

photo = np.array(p_list)
label = np.array(pl_list)


# 예측 수행
yhat = model.predict(photo)

# 시각화
plt.rcParams['figure.figsize'] = (14, 14)
fig, ax_arr = plt.subplots(3, 5)

for i in range(13):
    sub_plt = ax_arr[i//5, i%5]  
    sub_plt.axis('off')
    sub_plt.imshow(photo[i])
    title = 'R: %s, P: %s' % (np.argmax(label[i]), np.argmax(yhat[i]))
    sub_plt.set_title(title)

plt.show()


