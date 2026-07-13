from ai.core import get_device_info, get_logger, set_seed
from configs.settings import APP_NAME, MODEL_NAME, VERSION


def main() -> None:
    logger = get_logger(APP_NAME)

    set_seed(42)

    device_info = get_device_info()

    print("=" * 50)
    print(APP_NAME)
    print("=" * 50)
    print("Version :", VERSION)
    print("Model   :", MODEL_NAME)
    print("Device  :", device_info["device"])

    if "gpu_name" in device_info:
        print("GPU     :", device_info["gpu_name"])

    print("System Ready!")

    logger.info("%s version %s started.", APP_NAME, VERSION)
    logger.info("Selected device: %s", device_info["device"])


if __name__ == "__main__":
    main()