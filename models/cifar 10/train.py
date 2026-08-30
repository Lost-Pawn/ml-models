import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from data import load_cifar10
from model import build_cnn, build_densenet121
from visualize import plot_history, evaluate_and_report

METRICS = ['accuracy', tf.keras.metrics.Precision(name='precision'),
           tf.keras.metrics.Recall(name='recall')]


def compile_model(model, metrics=METRICS):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)
    return model


def train_cnn(x_train, y_train_cat, x_test, y_test_cat, epochs=50, batch_size=32):
    model = compile_model(build_cnn())
    model.summary()

    # patience bumped to 4 - 2 was stopping runs before val_loss had a
    # chance to settle, especially with augmentation adding some noise
    early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

    history = model.fit(
        x_train, y_train_cat,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test_cat),
        callbacks=[early_stop],
    )
    return model, history


def train_densenet(x_train, y_train_cat, x_test, y_test_cat, epochs=100, batch_size=32):
    model = compile_model(build_densenet121(), metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

    history = model.fit(
        x_train, y_train_cat,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test_cat),
        callbacks=[early_stop],
    )
    return model, history


if __name__ == '__main__':
    (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat) = load_cifar10()

    cnn_model, cnn_history = train_cnn(x_train, y_train_cat, x_test, y_test_cat)
    plot_history(cnn_history)
    cnn_score = cnn_model.evaluate(x_test, y_test_cat)
    print(f'CNN test accuracy: {cnn_score[1] * 100:.2f}%')

    densenet_model, densenet_history = train_densenet(x_train, y_train_cat, x_test, y_test_cat)
    densenet_score = densenet_model.evaluate(x_test, y_test_cat)
    print(f'DenseNet121 test accuracy: {densenet_score[1] * 100:.2f}%')

    # pick a winner before saving/reporting, rather than always saving the CNN
    if cnn_score[1] >= densenet_score[1]:
        best_model, best_name = cnn_model, 'cnn'
    else:
        best_model, best_name = densenet_model, 'densenet121'

    print(f'Best model: {best_name}')
    evaluate_and_report(best_model, x_test, y_test)
    best_model.save('cnn_20_epochs.h5')
