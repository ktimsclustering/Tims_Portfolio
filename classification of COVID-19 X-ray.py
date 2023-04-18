import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load dataset using ImageDataGenerator with data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('covid_xray/train', target_size=(64, 64), batch_size=32, class_mode='categorical')
test_set = test_datagen.flow_from_directory('covid_xray/test', target_size=(64, 64), batch_size=32, class_mode='categorical')


# Initialize CNN model
model = Sequential()

# Add convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation function
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))

# Add max pooling layer with 2x2 pool size
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add additional convolutional and max pooling layers
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output of the convolutional layers
model.add(Flatten())

# Add fully connected layer with 128 units and ReLU activation function
model.add(Dense(units=128, activation='relu'))

# Add output layer with softmax activation function for multiclass classification
model.add(Dense(units=3, activation='softmax'))

# Compile the model with categorical cross-entropy loss function and Adam optimizer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model on training data
model.fit(training_set, epochs=10, validation_data=test_set)

# Evaluate model performance using test set
test_loss, test_accuracy = model.evaluate(test_set)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
