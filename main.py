from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    data_train_gen = ImageDataGenerator();
    data_val_gen = ImageDataGenerator();

    train_gen = data_train_gen.flow_from_directory('train',
                                                   target_size=(48,48),
                                                   batch_size=64,
                                                   color_mode="grayscale",
                                                   class_mode='categorical')
    val_gen = data_val_gen.flow_from_directory('test',
                                               target_size=(48, 48),
                                               batch_size=64,
                                               color_mode="grayscale",
                                               class_mode='categorical')

    # Creating Sequential model
    emotion_model = Sequential()

    #Convolutional layers
    emotion_model.add(Conv2D(128, kernel_size=4, activation='relu', input_shape=(48, 48, 1)))
    emotion_model.add(Conv2D(64, kernel_size=4, activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=2))
    emotion_model.add(Dropout(0.2))

    # More convolutional layers
    emotion_model.add(Conv2D(32, kernel_size=4, activation='relu'))
    emotion_model.add(Conv2D(32, kernel_size=4, activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=2))
    emotion_model.add(Dropout(0.2))

    # Dense layer
    emotion_model.add(Flatten())
    emotion_model.add(Dense(1024, activation='relu'))
    emotion_model.add(Dropout(0.2))
    emotion_model.add(Dense(7, activation='softmax'))

    emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    # Training
    emotion_model_info = emotion_model.fit(
        train_gen,
        steps_per_epoch=28709 // 64,
        epochs=64,
        validation_data=val_gen,
        validation_steps=3589 // 64)

    # Model saving
    model_json = emotion_model.to_json()
    with open("trained_model.json", "w") as json_file:
        json_file.write(model_json)

    # Weight saving
    emotion_model.save_weights('trained_model.h5')