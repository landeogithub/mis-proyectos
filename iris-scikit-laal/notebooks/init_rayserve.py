# init_rayserve.py
import ray
from ray import serve
import logging
import sys

# --------------------- Logging Setup ---------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('init_rayserve.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --------------------- Global Configuration ---------------------
GLOBAL_CONFIG = {
    'RAY_HEAD': "ray-head",
    'RAY_PORT': "10001",
    'SERVE_HOST': "0.0.0.0",
    'SERVE_PORT': 8000,
}

def init_ray_serve():
    """Initialize Ray and Serve."""
    try:
        ray.init(address=f"ray://{GLOBAL_CONFIG['RAY_HEAD']}:{GLOBAL_CONFIG['RAY_PORT']}")
        serve.start(detached=True, http_options={
            'host': GLOBAL_CONFIG["SERVE_HOST"],
            'port': GLOBAL_CONFIG["SERVE_PORT"]
        })
        logger.info("Ray and Serve initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Ray/Serve: {e}")
        raise

if __name__ == "__main__":
    init_ray_serve()
