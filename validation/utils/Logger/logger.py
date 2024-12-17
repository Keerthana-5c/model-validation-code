import logging
import os

class BaseLogger:

    def __init__(self, file_name):
        self.create_logger(file_name)


    def create_logger(self, file_name):
        self.create_logs_directory()

        self.logger = logging.getLogger("base_logger")
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            file_handler = logging.FileHandler(os.path.join('logs', file_name))
            file_handler.setLevel(logging.DEBUG)

            formatter = logging.Formatter('[%(asctime)s]  %(levelname)s - %(name)s | %(message)s')

            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

    def create_logs_directory(self):
        logs_dir = 'logs'
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

    def log_info(self, message):
        self.logger.info(message)

    def log_warning(self, message):
        self.logger.warning(message)

    def log_error(self, message):
        self.logger.error(message)