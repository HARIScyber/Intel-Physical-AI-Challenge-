import openvino as ov
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_device_info():
    """
    Detects and prints available OpenVINO devices and version.
    """
    core = ov.Core()
    version = ov.get_version()

    logger.info(f"OpenVINO Version: {version}")

    devices = core.available_devices
    logger.info(f"Available Devices: {devices}")

    device_details = {}
    for device in devices:
        try:
            # Get full device name/properties
            full_name = core.get_property(device, "FULL_DEVICE_NAME")
            device_details[device] = full_name
        except Exception as e:
            logger.warning(f"Could not get details for device {device}: {e}")
            device_details[device] = "Unknown"

    return {
        "version": version,
        "devices": devices,
        "details": device_details
    }

if __name__ == "__main__":
    info = get_device_info()
    print("\n--- OpenVINO Device Information ---")
    print(f"Version: {info['version']}")
    print(f"Available Devices: {', '.join(info['devices'])}")
    for dev, detail in info['details'].items():
        print(f"  - {dev}: {detail}")
    print("----------------------------------\n")
