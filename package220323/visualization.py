import matplotlib.pyplot as plt


def accuracy_graf(df):
    plt.figure(figsize=(9, 6))
    plt.plot(df['accuracy'], marker='D', color='b',
             label='Доля верных ответов на обучающем наборе')
    plt.plot(df['val_accuracy'], marker='o', color='g',
             label='Доля верных ответов на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Доля верных ответов')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

