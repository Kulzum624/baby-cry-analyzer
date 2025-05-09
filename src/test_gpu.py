import tensorflow as tf
from tensorflow.python.client import device_lib
import sys

def test_gpu():
    print("TensorFlow version:", tf.__version__)
    print("\nAvailable devices:")
    print(device_lib.list_local_devices())
    
    # Test GPU availability
    if tf.test.is_built_with_cuda():
        print("\nTensorFlow is built with CUDA")
    else:
        print("\nTensorFlow is NOT built with CUDA")
    
    # Test GPU memory
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print("\nFound GPU devices:", gpus)
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        else:
            print("\nNo GPU devices found")
    except Exception as e:
        print("\nError checking GPU:", str(e))
    
    # Force flush output
    sys.stdout.flush()

if __name__ == "__main__":
    test_gpu() 