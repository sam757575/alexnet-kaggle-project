import unittest
from src.models.alexnet import AlexNet

class TestAlexNet(unittest.TestCase):

    def setUp(self):
        self.model = AlexNet()

    def test_model_initialization(self):
        self.assertIsNotNone(self.model)

    def test_model_output_shape(self):
        input_shape = (224, 224, 3)  # Example input shape for AlexNet
        alexnet = AlexNet(input_shape=input_shape, num_classes=1000)
        self.assertEqual(alexnet.model.output_shape, (None, 1000))

    def test_model_compile(self):
        try:
            self.model.compile_model()
        except Exception as e:
            self.fail(f"Model compilation failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
