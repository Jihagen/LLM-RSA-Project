import os

class FileManager:
    """
    Utility class to manage file paths and operations.
    """
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.getcwd()

    def get_full_path(self, *path_parts):
        """
        Construct the full path from base directory and path parts.
        """
        return os.path.join(self.base_dir, *path_parts)

    def ensure_dir_exists(self, path):
        """
        Ensure that a directory exists. If not, create it.
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def list_files(self, directory, extension=None):
        """
        List files in a directory with an optional file extension filter.
        """
        files = os.listdir(directory)
        if extension:
            files = [f for f in files if f.endswith(extension)]
        return files
