import tensorflow as tf



def train_nn(X,y):
    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=[X.shape[1]]),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(6,activation='softmax'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"])
    model.fit(X,y,
                epochs=30,
                verbose=0)
    return model