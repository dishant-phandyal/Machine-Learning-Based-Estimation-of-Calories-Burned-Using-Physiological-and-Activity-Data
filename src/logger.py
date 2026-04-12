import logging
import os
from datetime import datetime

# 1. Create a log file name based on the current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# 2. Define the path to the logs directory
logs_dir = os.path.join(os.getcwd(), "logs")

# 3. Create the logs directory if it doesn't exist
os.makedirs(logs_dir, exist_ok=True)

# 4. Define the full path to the specific log file
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# 5. Configure the basic logging setup
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)