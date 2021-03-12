from utility.loading_bar import LoadingBar
import time
import numpy as np

class Log:
    def __init__(self, log_each: int, initial_epoch=-1):
        self.loading_bar = LoadingBar(length=27)
        self.best_accuracy = 0.0
        self.log_each = log_each
        self.epoch = initial_epoch
        self.test_class_accuracies = np.zeros((10), dtype=float)
        self.average_class_accuracies = np.zeros((10), dtype=float)
        self.test_class_sizes = np.zeros((10), dtype=float)
        self.alpha = 0.1

    def train(self, len_dataset: int) -> None:
        self.epoch += 1
        if self.epoch == 0:
            self._print_header()
        else:
            self.flush()

        self.is_train = True
        self.last_steps_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0}
        test_class_accuracies = self.test_class_accuracies
        self._reset(len_dataset)

    def eval(self, len_dataset: int) -> None:
        self.flush()
        self.is_train = False
        self._reset(len_dataset)

    def __call__(self, model, loss, accuracy, targets, learning_rate: float = None) -> None:
        if self.is_train:
            self._train_step(model, loss, accuracy, learning_rate)
        else:
            self._eval_step(loss, accuracy, targets)

    def flush(self) -> None:
        if self.is_train:
            loss = self.epoch_state["loss"] / self.epoch_state["steps"]
            accuracy = self.epoch_state["accuracy"] / self.epoch_state["steps"]

            print(
                f"\r┃{self.epoch:12d}  ┃{loss:12.4f}  │{100*accuracy:10.2f} %  ┃{self.learning_rate:12.3e}  │{self._time():>12}  ┃",
                end="",
                flush=True,
            )

        else:
            loss = self.epoch_state["loss"] / self.epoch_state["steps"]
            accuracy = self.epoch_state["accuracy"] / self.epoch_state["steps"]
            class_accuracies = ["", "", "", "", "", "", "", "", "", ""]

            for i in range(10):
                self.test_class_accuracies[i] = self.test_class_accuracies[i] / self.test_class_sizes[i]
                self.average_class_accuracies[i] = self.alpha*self.test_class_accuracies[i] + (1.0-self.alpha)*self.average_class_accuracies[i]
                class_accuracies[i] = "{:.2f}".format(100*self.average_class_accuracies[i])

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy

            print(f"{loss:8.4f} │{100*accuracy:10.2f}% │{100*self.best_accuracy:10.2f}%  ┃", flush=True)
            print("┃   Test accuracies by class ", class_accuracies)

    def _train_step(self, model, loss, accuracy, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self.last_steps_state["loss"] += loss.sum().item()
        self.last_steps_state["accuracy"] += accuracy.sum().item()
        self.last_steps_state["steps"] += loss.size(0)
        self.epoch_state["loss"] += loss.sum().item()
        self.epoch_state["accuracy"] += accuracy.sum().item()
        self.epoch_state["steps"] += loss.size(0)
        self.step += 1

        if self.step % self.log_each == self.log_each - 1:
            loss = self.last_steps_state["loss"] / self.last_steps_state["steps"]
            accuracy = self.last_steps_state["accuracy"] / self.last_steps_state["steps"]

            self.last_steps_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0}
            progress = self.step / self.len_dataset

            print(
#                f"\r┃{self.epoch:12d}  ┃{loss:12.4f}  │{100*accuracy:10.2f} %  ┃{learning_rate:12.3e}  │{self._time():>12}  {self.loading_bar(progress)}",
                f"\r┃{self.epoch:12d}  ┃{loss:12.4f}  │{100*accuracy:10.2f} %  ┃{learning_rate:12.3e}  │{self._time():>12}",
                end="",
                flush=True,
            )

    def _eval_step(self, loss, accuracy, targets) -> None:
        self.epoch_state["loss"] += loss.sum().item()
        self.epoch_state["accuracy"] += accuracy.sum().item()
        self.epoch_state["steps"] += loss.size(0)
        for i in range(10):
            indicies = (targets == i)
            self.test_class_sizes[i] += sum(indicies.float())
            self.test_class_accuracies[i] += sum(accuracy[indicies].float())

    def _reset(self, len_dataset: int) -> None:
        self.start_time = time.time()
        self.step = 0
        self.len_dataset = len_dataset
        self.epoch_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0}
        self.test_class_accuracies = 0*self.test_class_accuracies
        self.test_class_sizes = 0*self.test_class_sizes 

    def _time(self) -> str:
        time_seconds = int(time.time() - self.start_time)
        return f"{time_seconds // 60:02d}:{time_seconds % 60:02d} min"

    def _print_header(self) -> None:
        print(f"┏━━━━━━━━━━━━━━┳━━━━━━━TRAIN━━━━━━━━━━━━━━━━━┳━━━━━━━STATS━━━━━━━━━━━━━━━━━┳━━━━━━━VALID━━━━━━━━━━━━━━━━━━━━━━━━┓")
        print(f"┃              ┃              ╷              ┃              ╷              ┃                                    ┃")
        print(f"┃       epoch  ┃        loss  │    accuracy  ┃        l.r.  │     elapsed  ┃   loss  │  accuracy  │    Best     ┃")
        print(f"┠──────────────╂──────────────┼──────────────╂──────────────┼──────────────╂─────────┼────────────┼─────────────┨")
