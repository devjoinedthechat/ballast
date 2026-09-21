"""Where bytes live.

A backend stores opaque blobs by tenant and digest. It knows nothing about
tensors, blocks or compression; that is the chunk store's job, one layer up.
Keeping the boundary this narrow is what makes a second backend a small file
rather than a fork, and what makes the eventual systems-language rewrite a
matter of replacing one class.

Two ship. `LocalBackend` writes files, atomically and durably: a per-writer
temporary name, fsync of the file, rename, fsync of the directory, so a crash at
any point leaves either the old state or the new one and never a torn blob.
`S3Backend` puts objects; durability is the service's. Both are exercised by the
same tests, the latter against an in-process mock.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator


@runtime_checkable
class Backend(Protocol):
    def exists(self, tenant: str, digest: str) -> bool: ...

    def put(self, tenant: str, digest: str, data: bytes | memoryview) -> bool:
        """Store a blob. Returns False if it was already present."""

    def get(self, tenant: str, digest: str) -> bytes | memoryview:
        """The blob's bytes. Raises FileNotFoundError if absent."""

    def delete(self, tenant: str, digest: str) -> int:
        """Remove a blob. Returns the bytes freed, 0 if it was absent."""

    def delete_tenant(self, tenant: str) -> int:
        """Remove every blob a tenant has. Returns the bytes freed."""

    def list_tenant(self, tenant: str) -> Iterator[str]:
        """Every digest a tenant has, for reconciliation."""

    def describe(self) -> str: ...


class LocalBackend:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def path(self, tenant: str, digest: str) -> Path:
        return self.root / tenant / digest[:2] / digest

    def exists(self, tenant: str, digest: str) -> bool:
        return self.path(tenant, digest).exists()

    def put(self, tenant: str, digest: str, data: bytes | memoryview) -> bool:
        target = self.path(tenant, digest)
        if target.exists():
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            with tmp.open("wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(target)
            _fsync_dir(target.parent)
        finally:
            tmp.unlink(missing_ok=True)
        return True

    def get(self, tenant: str, digest: str) -> bytes | memoryview:
        target = self.path(tenant, digest)
        if not target.exists():
            raise FileNotFoundError(f"blob {digest[:12]} missing for tenant {tenant!r}")
        return target.read_bytes()

    def mmap(self, tenant: str, digest: str) -> Any:
        """A read-only memory map of the blob, for zero-copy reads of raw blocks."""
        import numpy as np  # noqa: PLC0415

        target = self.path(tenant, digest)
        if not target.exists():
            raise FileNotFoundError(f"blob {digest[:12]} missing for tenant {tenant!r}")
        if target.stat().st_size == 0:
            return np.empty(0, dtype=np.uint8)
        return np.memmap(target, mode="r", dtype=np.uint8)

    def delete(self, tenant: str, digest: str) -> int:
        target = self.path(tenant, digest)
        if not target.exists():
            return 0
        size = target.stat().st_size
        target.unlink()
        parent = target.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
        return size

    def delete_tenant(self, tenant: str) -> int:
        root = self.root / tenant
        if not root.exists():
            return 0
        freed = 0
        for file in root.rglob("*"):
            if file.is_file():
                freed += file.stat().st_size
                file.unlink()
        for folder in sorted(root.rglob("*"), reverse=True):
            if folder.is_dir():
                folder.rmdir()
        root.rmdir()
        return freed

    def list_tenant(self, tenant: str) -> Iterator[str]:
        root = self.root / tenant
        if not root.exists():
            return
        for file in root.rglob("*"):
            if file.is_file() and not file.name.endswith(".tmp"):
                yield file.name

    def describe(self) -> str:
        return f"local:{self.root}"


class S3Backend:
    """Blobs as objects under `s3://bucket/prefix/<tenant>/<digest>`.

    Needs boto3, imported when constructed. Pass a `client` to use a configured
    or mocked one; otherwise boto3's default resolution applies.
    """

    def __init__(self, bucket: str, prefix: str = "", client: Any = None) -> None:
        if client is None:
            import boto3  # noqa: PLC0415

            client = boto3.client("s3")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = client

    def key(self, tenant: str, digest: str) -> str:
        return f"{self.prefix}/{tenant}/{digest}" if self.prefix else f"{tenant}/{digest}"

    def _tenant_prefix(self, tenant: str) -> str:
        return f"{self.prefix}/{tenant}/" if self.prefix else f"{tenant}/"

    def exists(self, tenant: str, digest: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self.key(tenant, digest))
        except self.client.exceptions.ClientError:
            return False
        return True

    def put(self, tenant: str, digest: str, data: bytes | memoryview) -> bool:
        if self.exists(tenant, digest):
            return False
        self.client.put_object(Bucket=self.bucket, Key=self.key(tenant, digest), Body=bytes(data))
        return True

    def get(self, tenant: str, digest: str) -> bytes:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.key(tenant, digest))
        except self.client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"blob {digest[:12]} missing for tenant {tenant!r}") from None
        return bytes(response["Body"].read())

    def delete(self, tenant: str, digest: str) -> int:
        key = self.key(tenant, digest)
        try:
            size = int(self.client.head_object(Bucket=self.bucket, Key=key)["ContentLength"])
        except self.client.exceptions.ClientError:
            return 0
        self.client.delete_object(Bucket=self.bucket, Key=key)
        return size

    def delete_tenant(self, tenant: str) -> int:
        freed = 0
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._tenant_prefix(tenant)):
            objects = page.get("Contents") or []
            if not objects:
                continue
            freed += sum(int(o["Size"]) for o in objects)
            self.client.delete_objects(
                Bucket=self.bucket, Delete={"Objects": [{"Key": o["Key"]} for o in objects], "Quiet": True}
            )
        return freed

    def list_tenant(self, tenant: str) -> Iterator[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        prefix = self._tenant_prefix(tenant)
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                yield obj["Key"][len(prefix) :]

    def describe(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}" if self.prefix else f"s3://{self.bucket}"


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def from_url(url: str, local_root: Path) -> Backend:
    """`s3://bucket/prefix` or a path; anything else is an error."""
    if url.startswith("s3://"):
        rest = url[len("s3://") :]
        bucket, _, prefix = rest.partition("/")
        if not bucket:
            raise ValueError(f"no bucket in {url!r}")
        return S3Backend(bucket, prefix)
    if "://" in url:
        raise ValueError(f"unsupported backend {url!r}; use a path or s3://bucket/prefix")
    return LocalBackend(Path(url) if url else local_root)
