import hashlib


def get_file_md5_checksum(file_path):
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5()
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            file_hash.update(chunk)

    return file_hash.hexdigest()
