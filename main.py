from src.model import Interpreter
from data.loader import Loader

if __name__ == "__main__":
    model = Interpreter()
    train_data = Loader(mode="train", base_path="data")
    test_data = Loader(mode="test", base_path="data")
    model.fit(train_data, 5)
    model.test(test_data)