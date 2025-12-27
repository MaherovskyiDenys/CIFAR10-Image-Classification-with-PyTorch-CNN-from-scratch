from network.model import CIFARClassifier
import argparse


def test() -> None:
    model = CIFARClassifier()

    parser = argparse.ArgumentParser(
        prog="CIFAR-10 model",
        description='Pre-trained CIFAR-10 model'
    )
    parser.add_argument('path', help="Path of the JPEG, PNG or GIF image")
    parser.add_argument('--show', help="Displays your image before evaluation, resized", action="store_true")

    args = parser.parse_args()

    print("Model weights are " + "loaded correctly." if model.load_state() else "not loaded.")

    print(f"Predicted label: {model.evaluate(args.path, show=args.show)}")


if __name__ == "__main__":
    test()