#%%
# imports
from keras import models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.applications import InceptionResNetV2

#%%
# preprocessing and datagen
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=40,
    horizontal_flip=True,
)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
train_generator = train_datagen.flow_from_directory(
    "../food-5k/training", target_size=(512, 512), batch_size=32, class_mode="binary"
)
test_generator = test_datagen.flow_from_directory(
    "../food-5k/evaluation",
    target_size=(512, 512),
    batch_size=32,
    class_mode="binary",
)

#%%
# build model
model = models.Sequential()
resnet = InceptionResNetV2(
    weights="imagenet", include_top=False, input_shape=(512, 512, 3)
)
resnet.trainable = False
model.add(resnet)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))
print(model.summary())
model.compile(
    loss="binary_crossentropy", optimizer=optimizers.RMSprop(lr=2e-5), metrics=["acc"]
)

#%%
# train model
history = model.fit_generator(train_generator, validation_data=test_generator, epochs=3)
# %%
# save model
model.save("model")
