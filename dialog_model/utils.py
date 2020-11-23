import hashlib


def iterate_on_parts_by_condition(iterable, condition):
    cur_chunk = []
    for elem in iterable:
        if not condition(elem):
            cur_chunk.append(elem)
        else:
            yield cur_chunk
            cur_chunk = []

    if cur_chunk:
        yield cur_chunk


def get_file_md5_checksum(file_path):
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()
