import logging 
import os
from datetime import datetime

logs_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_dir, exist_ok=True)    

file_name = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(logs_dir, file_name)


logging.basicConfig(
    filename= log_path, 
    format = "[%(asctime)s %(lineno)d %(name)s - %(levelname)s - %(message)s]", 
    level = logging.INFO
)