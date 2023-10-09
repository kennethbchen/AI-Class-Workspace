from os import PathLike, path
from typing import Callable
import csv

def read_csv_cached(file_path: PathLike, read_func: Callable, force_regenerate = False):

    if not path.exists(file_path) or force_regenerate:

        if force_regenerate:
            print("Force Regenerating", file_path)
        else:
            print("Creating", file_path)

        with open(file_path, 'w', newline='\n', encoding='utf-8') as file:
            writer = csv.writer(file)

            for row in read_func():

                if type(row) is not list:
                    writer.writerow(list([row]))
                else:
                    writer.writerow(list(row))

    print("Reading", file_path)
    file = open(file_path, "r", encoding='utf-8')
    data = list(csv.reader(file, delimiter=","))
    file.close()

    return data