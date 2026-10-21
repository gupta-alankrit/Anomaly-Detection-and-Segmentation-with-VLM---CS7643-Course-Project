"""Read selected portions of a remote ZIP archive without storing the whole ZIP.

ZIP files keep a directory of member names and byte offsets near the end of the
archive. Python's ``zipfile.ZipFile`` needs to seek to that directory and then to
individual compressed members. ``RemoteZip`` provides those file operations by
requesting HTTP byte ranges, so ``zipfile`` can perform the actual ZIP parsing.

The reader fetches 4 MiB blocks by default and retains up to 12 blocks in memory
(about 48 MiB). A cache miss can transfer extra neighboring archive data; this is
not a one-request-per-image downloader. The server must honor Range requests.

Example
-------
    import zipfile
    remote = RemoteZip("https://example.org/dataset.zip")
    with zipfile.ZipFile(remote) as archive:
        annotations = archive.read("category/text_annos.json")

``read()`` without a size reads all remaining archive bytes. Use this reader
through ``zipfile`` when the intention is to fetch only selected members.
"""

import io

import requests


class RemoteZip(io.RawIOBase):
    """Expose a remote HTTP object as a seekable, read-only binary file.

    Attributes
    ----------
    - size: int = Total archive size in bytes, reported by Content-Range.
    - pos: int = Current logical offset, independent of the cached HTTP blocks.
    - cache: dict[int, bytes] = Insertion-ordered cache, keyed by block number.

    Raises
    ------
    - requests.HTTPError for unsuccessful HTTP requests.
    - RuntimeError when a server ignores ranges or returns an unexpected length.
    """

    def __init__(self, url, block_size=4 * 1024 * 1024):
        """Resolve the download URL and obtain its total size with a 1-byte read.

        Arguments
        ----------
        - url: str = Public archive URL; HTTP redirects are followed.
        - block_size: int = Positive number of archive bytes per cached block.

        Returns
        -------
        - None. The initialized reader starts at logical byte offset zero.
        """
        self.session = requests.Session()
        # Content-Length is only 1 for this response. Content-Range's denominator
        # gives the full object size, including for a redirected Hugging Face URL.
        self.response = self.session.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        self.response.raise_for_status()
        if self.response.status_code != 206:
            raise RuntimeError("Server does not support byte ranges")
        self.url = self.response.url
        self.size = int(self.response.headers["Content-Range"].split("/")[-1])
        self.pos = 0
        self.block_size = block_size
        self.cache = {}

    def readable(self):
        """Return True: this object supports the binary read() interface."""
        return True

    def seekable(self):
        """Return True: zipfile can reposition within the remote archive."""
        return True

    def tell(self):
        """Return the current logical byte offset as an int."""
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        """Move the logical cursor without requesting any bytes over the network.

        Arguments
        ----------
        - offset: int = Byte displacement from the selected origin.
        - whence: int = io.SEEK_SET (start), io.SEEK_CUR (cursor), or
          io.SEEK_END (archive end). Callers must supply one of these constants.

        Returns
        -------
        - int = New offset. Raises ValueError for a target outside [0, size].
        """
        self.pos = (0 if whence == io.SEEK_SET else self.pos if whence == io.SEEK_CUR else self.size) + offset
        if not 0 <= self.pos <= self.size:
            raise ValueError("Seek outside archive")
        return self.pos

    def _block(self, n):
        """Return block n from memory or fetch its inclusive HTTP byte range.

        Arguments
        ----------
        - n: int = Zero-based block number; byte offset is n * block_size.

        Returns
        -------
        - bytes = The block, shortened only at the end of the archive.
        """
        if n not in self.cache:
            start = n * self.block_size
            end = min(start + self.block_size, self.size) - 1
            response = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=90)
            response.raise_for_status()
            if response.status_code != 206 or len(response.content) != end - start + 1:
                raise RuntimeError("Invalid range response")
            # Evict the oldest inserted block to bound memory. Cache hits do not
            # change insertion order, so this is FIFO rather than an LRU cache.
            if len(self.cache) >= 12:
                self.cache.pop(next(iter(self.cache)))
            self.cache[n] = response.content
        return self.cache[n]

    def read(self, size=-1):
        """Read from the cursor, joining cached blocks when a read spans them.

        Arguments
        ----------
        - size: int = Requested bytes; a negative value means all remaining bytes.

        Returns
        -------
        - bytes = At most size bytes, or b'' at EOF. The cursor advances by the
          number of returned bytes. No padding is added beyond archive EOF.
        """
        if size < 0:
            size = self.size - self.pos
        size = min(size, self.size - self.pos)
        chunks = []
        while size:
            # divmod maps an absolute offset to a cached block and its local
            # offset. A read can cross any number of block boundaries.
            block_number, offset = divmod(self.pos, self.block_size)
            take = min(size, self.block_size - offset)
            chunks.append(self._block(block_number)[offset:offset + take])
            self.pos += take
            size -= take
        return b"".join(chunks)


