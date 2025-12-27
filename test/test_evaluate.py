from network.model import CIFARClassifier

def test() -> None:
    model = CIFARClassifier()

    model.load_state()
    print("Model is loaded.")

    print(f"Predicted value: {model.evaluate("img/1.jpg", show=True)}")

if __name__ == "__main__":
    test()