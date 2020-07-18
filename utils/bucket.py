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


def download_config(bucket_name: str, data_path: str):
    (_, bucket) = _get_client_bucket(bucket_name)
    blob = bucket.blob(data_path)
    blob.download_to_filename(data_path)


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


def upload_model(bucket_name: str, save_path: str):
    name = save_path.replace(os.path.abspath(os.curdir) + '/', '')

    (_, bucket) = _get_client_bucket(bucket_name)
    blob = Blob(name, bucket)
    blob.upload_from_filename(save_path)
