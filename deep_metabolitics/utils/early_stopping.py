class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: Kaç epoch boyunca iyileşme gözlenmezse durdurma işlemi yapılır.
        :param min_delta: Kaydedilmesi gereken minimum iyileşme miktarı.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif ((self.best_loss - val_loss) / self.best_loss) > self.min_delta:
            # val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # İyileşme olduğunda sıfırlanır
        else:
            self.counter += 1
            print(f"EarlyStopping Counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
