import os
import time
import argparse
import moxing as mox

basename = os.path.basename(__file__)
parser = argparse.ArgumentParser(prog=basename, description='TensorBoard log file synchronizer for ModelArts')
parser.add_argument('-l', '--logdir', type=str, required=True, help='S3 path of tensorboard log file')
parser.add_argument('-i', '--interval', type=int, default=30, help='Sync interval seconds (default: 30s)')
parser.add_argument('-d', '--daemon', action='store_true', help='Whether to run in daemon mode (sync periodically)')
args, unknown = parser.parse_known_args()

local_cache_path = os.getenv('DLS_LOCAL_CACHE_PATH', "/cache") + args.logdir.split("s3:/")[1]

if __name__ == '__main__':
    if args.daemon:
        while True:
            mox.file.sync_copy(args.logdir, local_cache_path)
            time.sleep(args.interval)
    else:
        mox.file.copy_parallel(args.logdir, local_cache_path)
