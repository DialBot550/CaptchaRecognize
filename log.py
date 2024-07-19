import logging
import config

logger = logging.getLogger('CaptchaRecognitionLogger')
logger.setLevel(logging.DEBUG)

# fh = logging.FileHandler(config.log_path)
# fh.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(formatter)

console_handler.setFormatter(formatter)

# logger.addHandler(fh)
logger.addHandler(console_handler)
