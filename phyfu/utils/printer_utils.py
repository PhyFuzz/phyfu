import os


def list_to_str(arr):
    return '[' + ''.join([f"{i:<10.2e}" for i in arr]).rstrip() + ']'


def float_to_str(f):
    return f"{f:<10.2e}"


to_readable_mapping = {
    list: list_to_str,
    float: float_to_str
}


def to_readable(item):
    if item.__class__ in to_readable_mapping.keys():
        return to_readable_mapping[item.__class__](item)
    return f"{item}"


class BufferedWriter:
    def __init__(self, file_path, write_freq):
        self.strings = []
        self.file_path = file_path
        if os.path.exists(file_path):
            print(f"WARNING: The file {file_path} already exists. Going to overwrite it.")
            os.remove(file_path)
        self.write_freq = write_freq

    def write(self, string_to_write):
        """
        Append a line to the buffer
        :param string_to_write: the string to be appended to the buffer
        :return: None
        """
        self.strings.append(string_to_write)
        if len(self.strings) > self.write_freq:
            self.flush()

    def flush(self):
        with open(self.file_path, 'a') as f:
            f.write("\n".join(self.strings))
            f.write("\n")
        self.strings = []

    def close(self):
        """
        IMPORTANT: Always call this function if you do not want to log anymore!
        :return: None
        """
        self.flush()


class EnhancedPrinter:
    """
    A useful function to substitute print().
    Can support persistent logging or simply write to stdout.
    Can also enable/disable printing out temporarily.
    """
    def __init__(self, to_file, print_out, target_file=None, write_freq=100):
        if to_file and target_file is not None:
            self.writer = BufferedWriter(target_file, write_freq=write_freq)
        else:
            self.target_file = None
        self.__print_out = print_out

    def print(self, *items, print_out=None, write_to_file=None):
        """
        Log an item. Also provides an option to temporarily enable/disable printing out
        :param items: the items to log, just like python's print function
        :param print_out: whether to print out to stdout or not.
        None means retain the default one;
        True or False to override the default setting temporarily.
        :param write_to_file: similar to print_out
        :return: None
        """
        if print_out or (print_out is None and self.__print_out):
            print(*items)
        if hasattr(self, "writer") and (write_to_file is None or write_to_file):
            self.writer.write(" ".join([str(item) for item in items]))

    def close(self):
        """
        IMPORTANT: Always call this function if you do not want to log anymore!
        :return: None
        """
        if hasattr(self, "writer"):
            self.writer.close()
