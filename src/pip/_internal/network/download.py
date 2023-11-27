"""Download files with progress indicators.
"""
import email.message
import logging
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Iterable, Optional, Tuple

from pip._vendor.requests.models import CONTENT_CHUNK_SIZE, Response
from pip._vendor.rich.progress import TaskID

from pip._internal.cli.progress_bars import PipProgress, get_download_progress_renderer
from pip._internal.exceptions import NetworkConnectionError
from pip._internal.models.index import PyPI
from pip._internal.models.link import Link
from pip._internal.network.cache import is_from_cache
from pip._internal.network.session import PipSession
from pip._internal.network.utils import HEADERS, raise_for_status, response_chunks
from pip._internal.utils.misc import format_size, redact_auth_from_url, splitext

logger = logging.getLogger(__name__)


def _get_http_response_size(resp: Response) -> Optional[int]:
    try:
        return int(resp.headers["content-length"])
    except (ValueError, KeyError, TypeError):
        return None


def get_logged_url(link: Link) -> str:
    if link.netloc == PyPI.file_storage_domain:
        url = link.show_url
    else:
        url = link.url_without_fragment

    return redact_auth_from_url(url)


def get_log_message(resp: Response, link: Link, total_length: Optional[int]) -> str:
    logged_url = get_logged_url(link)
    if total_length:
        logged_url = f"{logged_url} ({format_size(total_length)})"

    if is_from_cache(resp):
        return f"Using cached {logged_url}"
    return f"Downloading {logged_url}"


def _progress_iterator(
    chunks: Iterable[bytes],
    progress_bar: PipProgress,
    task_id: TaskID,
    parallel: bool = False,
) -> Iterable[bytes]:
    if parallel:
        # Dont add a `with` for parallel downloads, since they are already in
        # a `with` block
        for chunk in chunks:
            progress_bar.update(task_id=task_id, advance=len(chunk))
            yield chunk
    else:
        with progress_bar:
            for chunk in chunks:
                progress_bar.update(task_id=task_id, advance=len(chunk))
                yield chunk


def _prepare_download(
    resp: Response, link: Link, progress_bar: PipProgress, parallel: bool = False
) -> Iterable[bytes]:
    total_length = _get_http_response_size(resp)

    chunks = response_chunks(resp, CONTENT_CHUNK_SIZE)
    description = get_log_message(resp, link, total_length)

    if is_from_cache(resp):
        hide_progress = True
    elif total_length and total_length <= (40 * 1000):
        hide_progress = True
    elif progress_bar.progress_disabled:
        hide_progress = True
    else:
        hide_progress = False

    task_id = progress_bar.add_task(
        description=description,
        total=total_length,
        hide_progress=hide_progress,
    )
    if hide_progress and not parallel:
        return chunks

    return _progress_iterator(chunks, progress_bar, task_id, parallel)


def sanitize_content_filename(filename: str) -> str:
    """
    Sanitize the "filename" value from a Content-Disposition header.
    """
    return os.path.basename(filename)


def parse_content_disposition(content_disposition: str, default_filename: str) -> str:
    """
    Parse the "filename" value from a Content-Disposition header, and
    return the default filename if the result is empty.
    """
    m = email.message.Message()
    m["content-type"] = content_disposition
    filename = m.get_param("filename")
    if filename:
        # We need to sanitize the filename to prevent directory traversal
        # in case the filename contains ".." path parts.
        filename = sanitize_content_filename(str(filename))
    return filename or default_filename


def _get_http_response_filename(resp: Response, link: Link) -> str:
    """Get an ideal filename from the given HTTP response, falling back to
    the link filename if not provided.
    """
    filename = link.filename  # fallback
    # Have a look at the Content-Disposition header for a better guess
    content_disposition = resp.headers.get("content-disposition")
    if content_disposition:
        filename = parse_content_disposition(content_disposition, filename)
    ext: Optional[str] = splitext(filename)[1]
    if not ext:
        ext = mimetypes.guess_extension(resp.headers.get("content-type", ""))
        if ext:
            filename += ext
    if not ext and link.url != resp.url:
        ext = os.path.splitext(resp.url)[1]
        if ext:
            filename += ext
    return filename


def _http_get_download(session: PipSession, link: Link) -> Response:
    target_url = link.url.split("#", 1)[0]
    resp = session.get(target_url, headers=HEADERS, stream=True)
    raise_for_status(resp)
    return resp


def _download(
    link: Link,
    location: str,
    session: PipSession,
    progress_bar: PipProgress,
    parallel: bool = False,
) -> Tuple[str, str]:
    try:
        resp = _http_get_download(session, link)
    except NetworkConnectionError as e:
        assert e.response is not None
        logger.critical("HTTP error %s while getting %s", e.response.status_code, link)
        raise

    filename = _get_http_response_filename(resp, link)
    filepath = os.path.join(location, filename)

    chunks = _prepare_download(resp, link, progress_bar, parallel)
    with open(filepath, "wb") as content_file:
        for chunk in chunks:
            content_file.write(chunk)
    content_type = resp.headers.get("Content-Type", "")
    return filepath, content_type


class Downloader:
    def __init__(
        self,
        session: PipSession,
        progress_bar: str,
    ) -> None:
        self._session = session
        self._progress_bar = progress_bar

    def __call__(self, link: Link, location: str) -> Tuple[str, str]:
        """Download the file given by link into location."""

        progress_bar = get_download_progress_renderer(
            progress_type=self._progress_bar, logger=logger
        )
        return _download(link, location, self._session, progress_bar, parallel=False)


class BatchDownloader:
    def __init__(
        self,
        session: PipSession,
        progress_bar: str,
    ) -> None:
        self._session = session
        self._progress_bar = progress_bar

    def _sequential_download(
        self,
        link: Link,
        location: str,
        progress_bar: PipProgress,
        parallel: bool = False,
    ) -> Tuple[Link, Tuple[str, str]]:
        filepath, content_type = _download(
            link, location, self._session, progress_bar, parallel=parallel
        )
        return link, (filepath, content_type)

    def _download_parallel(
        self,
        links: Iterable[Link],
        location: str,
        max_workers: int,
        progress_bar: PipProgress,
    ) -> Iterable[Tuple[Link, Tuple[str, str]]]:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            _download_wrapper = partial(
                self._sequential_download,
                location=location,
                progress_bar=progress_bar,
                parallel=True,
            )
            results = list(pool.map(_download_wrapper, links))
        return results

    def download_parallel(
        self,
        links: Iterable[Link],
        location: str,
        max_workers: int,
        progress_bar: PipProgress,
    ) -> Iterable[Tuple[Link, Tuple[str, str]]]:
        # This is required because `with Progress` statements add a \n even if
        # nothing is explicitly printed. So in case the user specifies -q a `with`
        # block shouldn't be used
        if logger.getEffectiveLevel() > logging.INFO:
            return self._download_parallel(links, location, max_workers, progress_bar)
        with progress_bar:
            results = self._download_parallel(
                links, location, max_workers, progress_bar
            )
            return results

    def __call__(
        self, links: Iterable[Link], location: str
    ) -> Iterable[Tuple[Link, Tuple[str, str]]]:
        """Download the files given by links into location."""
        links = list(links)
        max_workers = self._session.parallel_downloads
        if max_workers == 1 or len(links) == 1:
            for link in links:
                # New progress bar is needed for each link to prevent artifacts
                # in Docker and jupyter
                progress_bar = get_download_progress_renderer(
                    progress_type=self._progress_bar, logger=logger
                )
                yield self._sequential_download(
                    link, location, progress_bar, parallel=False
                )
        else:
            progress_bar = get_download_progress_renderer(
                progress_type=self._progress_bar, parallel=True
            )
            results = self.download_parallel(links, location, max_workers, progress_bar)
            for result in results:
                yield result
