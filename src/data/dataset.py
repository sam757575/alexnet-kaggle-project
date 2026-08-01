class Dataset:
    def __init__(self, data_dir='./data', img_size=(32, 32), num_classes=10):
        """Simple Dataset helper that downloads Fashion-MNIST and returns tf.data datasets.

        This is a pragmatic default for running/troubleshooting the training pipeline.
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.num_classes = num_classes

    def load_data(self, batch_size=32, shuffle=True):
        """Download Fashion-MNIST and return (train_ds, val_ds) as tf.data.Dataset objects.

        The function uses the test split as validation for simplicity.
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.datasets import fashion_mnist
        except Exception:
            raise ImportError("TensorFlow is required to download and prepare the Fashion-MNIST dataset.")

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        # Normalize images to [0, 1] and adapt grayscale images to AlexNet's 3-channel input.
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        x_train = tf.image.resize(x_train[..., tf.newaxis], self.img_size)
        x_test = tf.image.resize(x_test[..., tf.newaxis], self.img_size)
        x_train = tf.image.grayscale_to_rgb(x_train)
        x_test = tf.image.grayscale_to_rgb(x_test)

        # Convert labels to one-hot for categorical_crossentropy.
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=self.num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=self.num_classes)

        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        if shuffle:
            train_ds = train_ds.shuffle(buffer_size=10000)

        train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds

    def preprocess(self, data):
        """Placeholder preprocessing hook (kept for compatibility).

        If additional preprocessing is required elsewhere in the codebase, update this
        method accordingly.
        """
        return data
