import os
import random

from typing import Tuple

from google.cloud.storage import Client
from google.cloud.storage.blob import Blob
from google.cloud.storage.bucket import Bucket

def _get_client_bucket(name: str) -> Tuple[Client, Bucket]:
    client = Client()
    bucket = Bucket(client, name)
    return (client, bucket)

def _download(bucket: Bucket, data_path: str):
    blob = bucket.blob(data_path)
    blob.download_to_filename(data_path)


def download_config(bucket_name: str, data_path: str):
    (_, bucket) = _get_client_bucket(bucket_name)
    _download(bucket, data_path)


def preload_checkpoints(bucket_name: str, checkpoints: [ str ]):
    (_, bucket) = _get_client_bucket(bucket_name)

    for checkpoint in checkpoints:
        if os.path.isfile(checkpoint):
            print(f'Found:{checkpoint}')
            continue

        dirname = os.path.dirname(checkpoint)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print(f'Downloading:{checkpoint}')
        _download(bucket, checkpoint)


def preload_dataset(bucket_name: str, data_path: str, sample_threshold: int) -> str:
    # ensure_dir_exists
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    (storage_client, bucket) = _get_client_bucket(bucket_name)

    dataset = storage_client.list_blobs(bucket, prefix=data_path)

    if sample_threshold > 0:
        random.seed(123)
        dataset = random.sample(list(dataset), k=sample_threshold)

    for blob in dataset:
        blob.download_to_filename(blob.name)

    return os.path.abspath(data_path)


def _upload(bucket: Bucket, src: str, prefix = None):
    dest = src.replace(os.path.abspath(os.curdir) + '/', '')
    if prefix:
        dest = os.path.join(prefix, dest)

    blob = Blob(dest, bucket)
    print(f'Uploading:{dest}')
    blob.upload_from_filename(src)


def upload_model(bucket_name: str, save_path: str):
    (_, bucket) = _get_client_bucket(bucket_name)
    _upload(bucket, save_path)


def upload_recursive(bucket_name: str, save_path: str, prefix=None):
    def _upload_internal(bucket: Bucket, path: str, prefix=None):
        path = os.path.abspath(path)
        for src in os.listdir(path):
            src = os.path.join(path, src)
            if os.path.isdir(src):
                _upload_internal(bucket, src, prefix)
                continue

            _upload(bucket, os.path.join(path, src), prefix)

    (_, bucket) = _get_client_bucket(bucket_name)
    _upload_internal(bucket, save_path, prefix)
