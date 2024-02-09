import horovod.tensorflow.keras as hvd
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers.legacy import Adam
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import time
# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# Parameters
batch_size = 32 // hvd.size()
epochs = 35
latent_dim = 100

# Data loading and preprocessing
root_path = "/kaggle/input/animefacedataset"  # Update with your path
root_path = pathlib.Path(root_path)
data = tf.keras.utils.image_dataset_from_directory(
    directory=root_path,
    label_mode=None,
    batch_size=batch_size,
    image_size=(64, 64)
)
data = data.map(lambda d: (d - 127.5) / 127.5)

# Define the Discriminator model
def Discriminator():
    model = Sequential()
    model.add(layers.Conv2D(64, kernel_size=3, strides=(2, 2), padding="same", activation="LeakyReLU", input_shape=(64, 64, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, kernel_size=3, strides=(2, 2), padding="same", activation="LeakyReLU"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(256, kernel_size=3, strides=(2, 2), padding="same", activation="LeakyReLU"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model

# Define the Generator model
def Generator():
    model = Sequential()
    model.add(layers.Dense(units=4*4*256, input_shape=[latent_dim], use_bias=False))
    model.add(layers.Reshape((4, 4, 256)))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="ReLU"))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="ReLU"))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="ReLU"))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2DTranspose(3, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="tanh"))
    return model

# Initialize the models
D_model = Discriminator()
G_model = Generator()

class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        seed = tf.random.normal(shape=(batch_size, self.latent_dim))
        # Decode them to fake images
        generated_images = self.generator(seed)
        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)
        # Assemble labels discriminating real from fake images
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))
        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Sample random points in the latent space
        seed = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(seed))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}

# Horovod: adjust learning rate based on number of GPUs.
optimizer = Adam(1e-4 * hvd.size())

# Compile the GAN model
gan = GAN(discriminator=D_model, generator=G_model, latent_dim=latent_dim)
gan.compile(d_optimizer=optimizer, g_optimizer=optimizer, loss_fn=tf.keras.losses.BinaryCrossentropy())

# Horovod: Broadcast initial variable states from rank 0 to all other processes.
callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

def save_generated_images(epoch, generator, examples=16, dim=(4, 4), figsize=(10, 10)):
    noise = tf.random.normal([examples, latent_dim])
    generated_images = generator.predict(noise)
    generated_images = (generated_images * 127.5 + 127.5).astype(np.uint8)  # Rescale to 0-255

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i, :, :, :], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'gan_generated_image_epoch_{epoch}.png')
    plt.close()

start_time = time.time()
# Training loop
for epoch in range(epochs):
    for image_batch in data:
        gan.train_step(image_batch)

    # At the end of each epoch, generate and save images
    if hvd.rank() == 0:
        save_generated_images(epoch, G_model)

    # Optionally save model checkpoints
    if hvd.rank() == 0:
        G_model.save(f'generator_epoch_{epoch}.h5')
        D_model.save(f'discriminator_epoch_{epoch}.h5')
end_time = time.time()
total_time = end_time - start_time
# Save the final model, only on the first worker
if hvd.rank() == 0:
    G_model.save('final_generator.h5')
    D_model.save('final_discriminator.h5')
print(f"Total time taken for {epochs} epochs: {total_time} sec.")