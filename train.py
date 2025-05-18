import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import (
    Input,
    AveragePooling2D,#AveragePooling2D is when you have a fixed dimension, GlobalAveragePooling2D is for variable dimensions
    Dropout,
    Flatten,
    Dense
)
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.optimizers import SGD
from keras.regularizers import l2
import keras.backend as K

from utils import *
    
def schedule(epoch):
    if epoch < 5:
        return 0.001
    elif epoch < 10:
        return .0002
    elif epoch < 15:
        return 0.00002
    else:
        return .0000005

NUM_EPOCHS = 10 #set num epochs

shape = (224, 224) #changed
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

X_train = train_datagen.flow_from_directory(
    'food-101/train',
    target_size=shape,
    batch_size=batch_size,
    class_mode='categorical'
)
# — save class_indices for inference later —
with open('class_indices.json', 'w') as f:
    json.dump(X_train.class_indices, f, indent=2)

X_test = test_datagen.flow_from_directory(
    'food-101/test',
    target_size=shape,
    batch_size=batch_size,
    class_mode='categorical'
)

base_model = InceptionV3( #changed 
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

x = base_model.output
x = AveragePooling2D(pool_size=(5, 5))(x) #changed

x = Dropout(.5)(x)
x = Flatten()(x)
predictions = Dense(
    X_train.num_classes,
    kernel_initializer='glorot_uniform',
    kernel_regularizer=l2(0.0005),
    activation='softmax'
)(x) #changed
model = Model(inputs=base_model.input, outputs=predictions) #changed
opt = SGD(learning_rate=0.1, momentum=0.9) #changed
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

checkpointer = ModelCheckpoint(
    filepath='model.{epoch:02d}-{val_loss:.2f}.keras',
    verbose=1,
    save_best_only=True
) #changed
csv_logger = CSVLogger('model.log')


lr_scheduler = LearningRateScheduler(schedule)
model.summary()

model.fit(
    X_train,
    validation_data=X_test,
    epochs=2,
    callbacks=[checkpointer]
)