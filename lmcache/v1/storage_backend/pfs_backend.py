from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.memory_management import MemoryAllocatorInterface, MemoryObj
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.storage_backend.gds_backend \
    import get_fstype, get_extra_config_bool, \
        pack_metadata, unpack_metadata, UnsupportedMetadataVersion, \
        rand_suffix, save_metadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, DiskCacheMetadata

import asyncio
import ctypes
import os
from collections import OrderedDict
from typing import Optional, List
from concurrent.futures import Future
from readerwriterlock import rwlock
import torch
import mmap
import numpy as np
import random

logger = init_logger(__name__)

_METADATA_FILE_SUFFIX = ".metadata"
_DATA_FILE_SUFFIX = ".kvcache.safetensors"
_METADATA_VERSION = 1
_METADATA_MAX_SIZE = 4096  # reserve 4K for metadata.

class PfsBackend(StorageBackendInterface):
    """
    分布式文件系统存储后端实现
    """
    def __str__(self):
        return self.__class__.__name__

    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        memory_allocator: MemoryAllocatorInterface,
        dst_device: str = "cuda",
    ):
        assert dst_device.startswith("cuda"), "PfsBackend only supports CUDA devices"
        super().__init__(dst_device)

        self.config = config
        self.loop = loop
        self.memory_allocator = memory_allocator
        self.dst_device = dst_device

        assert config.pfs_path is not None, "PFS path must be set in the configuration"
        self.pfs_path = config.pfs_path
        self.fstype = get_fstype(config.pfs_path)

        logger.info(f"Using PFS path: {self.pfs_path} with filesystem type: {self.fstype}")

        # cuFile & Direct IO 设置

        self.use_cufile = True
        self.use_direct_io = False
        if config.extra_config is not None:
            use_cufile = get_extra_config_bool("use_cufile", config)
            if use_cufile is not None:
                self.use_cufile = use_cufile
            use_direct_io = get_extra_config_bool("use_direct_io", config)
            if use_direct_io is not None:
                self.use_direct_io = use_direct_io

        if self.fstype in ["tmpfs", "overlayfs"]:
            self.use_cufile = False
            logger.warning(f"cuFile is not supported for {self.fstype}, disabling cuFile support")

        if self.use_cufile:
            logger.info("cuFile API is enabled")
            import cufile
            self.cudart = None
            self.cufile = cufile
            self._cufile_driver = self.cufile.CuFileDriver()
        else:
            logger.info("cuFile API is disabled")
            self.cufile = None
            self.cudart = ctypes.CDLL("libcudart.so")

        if not os.path.exists(self.pfs_path):
            logger.info(f"PFS path {self.pfs_path} does not exist, creating it")
            os.makedirs(self.pfs_path, exist_ok=True)

        # TODO: implement monitor
        self.stats = None

        self.rand = random.Random(self.dst_device)

        # 缓存结构

        # self.hot_lock = asyncio.Lock()
        self.hot_rwlock = rwlock.RWLockRead()
        self.hot_rlock = self.hot_rwlock.gen_rlock()
        self.hot_wlock = self.hot_rwlock.gen_wlock()
        self.hot_cache : OrderedDict[CacheEngineKey, DiskCacheMetadata] = OrderedDict() # 元数据

        # self.put_lock = asyncio.Lock()
        self.put_rwlock = rwlock.RWLockWrite()
        self.put_rlock = self.put_rwlock.gen_rlock()
        self.put_wlock = self.put_rwlock.gen_wlock()
        self.put_tasks : set[CacheEngineKey] = set()  # 正在进行的put任务

        if hasattr(self.memory_allocator, "base_pointer"):
            logger.debug(f"Using base pointer {self.memory_allocator.base_pointer}")
            self.cufile_base_pointer = self.memory_allocator.base_pointer
        else:
            logger.info("No base pointer found, cufile will use bounce buffers")
            self.cufile_base_pointer = None

        self.save_metadata_tasks: set[asyncio.Task] = set()

        # TODO: load挂载点已存在的数据
        # asyncio.run_coroutine_threadsafe(self._scan_metadata(), self.loop)

    def contains(self, key, pin = False):
        logger.debug(f"Checking if key {key} exists in PFS backend with pin={pin}")
        res : bool = False
        with self.hot_rlock:
            res = key in self.hot_cache
        # 避免争用hot_lock
        if res != True:
            res = bool(self._read_metadata(key))

        if pin:
            if res:
                self.hot_cache[key].pin()
            else:
                logger.warning(f"Key {key} not found in hot cache or metadata, cannot pin")

        logger.debug(f"Key {key} exists in PFS backend: {res}")
        return res

    def _read_metadata(self, key) -> Optional[DiskCacheMetadata]:
        """
        从元数据文件中读取指定key的元数据
        """
        logger.debug(f"Reading metadata for key {key} from PFS backend")
        path = self._key_to_path(key) + _METADATA_FILE_SUFFIX
        metadata_path = os.path.join(self.pfs_path, path)
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata file {metadata_path} does not exist")
            return None

        try:
            with os.open(metadata_path, os.O_DIRECT | os.O_RDONLY) as fd:
                buf = os.read(fd, _METADATA_MAX_SIZE)
                if not buf:
                    logger.warning(f"Metadata file {metadata_path} is empty")
                    return None

                shape, dtype, size, extra_metadata = unpack_metadata(buf)
                if extra_metadata["lmcache_version"] != str(_METADATA_VERSION):
                    raise UnsupportedMetadataVersion("unhandled lmcache metadata")

                # TODO(extra_metadata)
                metadata = DiskCacheMetadata(
                    metadata_path.removesuffix(_METADATA_FILE_SUFFIX), size, shape, dtype
                )
                with self.hot_wlock:
                    self.hot_cache[key] = metadata

                return metadata
        except UnsupportedMetadataVersion as e:
            logger.error(f"Unsupported metadata version for key {key}: {e}")
        except Exception as e:
            logger.error(f"Failed to read metadata for key {key}: {e}")

        return None

    def _key_to_path(self, key: CacheEngineKey) -> str:
        """
        Convert a CacheEngineKey to a file path in PFS.
        """
        return os.path.join(self.pfs_path, key.to_string().replace('/', '_') + _DATA_FILE_SUFFIX)

    def exists_in_put_tasks(self, key):
        logger.debug(f"Checking if key {key} exists in put tasks")
        with self.put_rlock:
            return key in self.put_tasks

    def batched_submit_put_task(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ) -> Optional[List[Future]]:
        logger.debug(f"Submitting batched put tasks for keys: {keys}")
        return [
            self.submit_put_task(key, memory_obj)
            for key, memory_obj in zip(keys, memory_objs, strict=False)
        ]

    def submit_put_task(
        self, key: CacheEngineKey, memory_obj: MemoryObj
    ) -> Optional[Future]:
        logger.debug(f"Submitting put task for key {key}")
        assert memory_obj is not None

        memory_obj.ref_count_up()

        with self.put_wlock:
            # if key in self.put_tasks:
            #     logger.warning(f"Key {key} is already in put tasks, skipping")
            #     return None
            self.put_tasks.add(key)

        # 异步提交put任务
        return asyncio.run_coroutine_threadsafe(
            self._async_save_bytes_to_disk(key, memory_obj), self.loop
        )

    async def _async_save_bytes_to_disk(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> None:
        """
        Convert KV to bytes and async store bytes to disk.
        """
        kv_chunk = memory_obj.tensor
        assert kv_chunk is not None

        path = self._key_to_path(key)
        tmp = ".tmp" + rand_suffix(self.rand, 8)
        metadata = await asyncio.to_thread(
            self._save_pfs,
            path,
            tmp,
            kv_chunk,
            self.cufile_base_pointer,
            memory_obj.metadata.address,
        )

        self.insert_key(key, memory_obj)
        memory_obj.ref_count_down()

        task = asyncio.create_task(
            save_metadata(path + _METADATA_FILE_SUFFIX, tmp, metadata)
        )
        self.save_metadata_tasks.add(task)
        task.add_done_callback(self.save_metadata_tasks.discard)
        with self.put_wlock:
            self.put_tasks.discard(key)

    @torch.inference_mode()
    def _save_pfs(
        self,
        path: str, tmp: str,
        kv_chunk: torch.Tensor,
        base_pointer: int,
        device_offset: int,
    ) -> DiskCacheMetadata:
        """
        Save the KV chunk to PFS and return metadata.
        """
        assert kv_chunk.is_cuda, "KV chunk must be a CUDA tensor"
        assert torch.device(self.dst_device) == torch.device(kv_chunk.device)

        metadata = pack_metadata(kv_chunk, lmcache_version=_METADATA_VERSION)

        # 分配临时文件路径
        tmp_path = path + tmp
        # 文件首部写入4KB元数据
        with open(tmp_path, "wb") as f:
            f.write(metadata)

        # 使用cuFile API或POSIX IO保存数据
        offset = _METADATA_MAX_SIZE
        if self.cufile_base_pointer is None:
            dev_addr = ctypes.c_void_p(kv_chunk.data_ptr())
            dev_offset = 0
        else:
            dev_addr = ctypes.c_void_p(base_pointer)
            dev_offset = device_offset

        if self.use_cufile:
            logger.debug(f"Saving data to PFS file {tmp_path} using cuFile API")
            with self.cufile.CuFile(tmp_path, "r+", use_direct_io=self.use_direct_io) as f:
                # f is the file descriptor
                ret = f.write(
                    dev_addr,
                    kv_chunk.nbytes,
                    file_offset=offset,
                    dev_offset=dev_offset,
                )
        else:
            logger.debug(f"Saving data to PFS file {tmp_path} using POSIX IO")
            nbytes = kv_chunk.nbytes
            fd = os.open(tmp_path, os.O_RDWR | os.O_DIRECT)
            os.ftruncate(fd, nbytes + offset) # 预分配文件空间
            mm = mmap.mmap(
                fd,
                nbytes + offset,
                prot=mmap.PROT_WRITE,
                flags=mmap.MAP_SHARED,
            )
            os.close(fd)

            # Save disk tensor
            arr = np.frombuffer(mm, dtype=np.uint8)
            buf_addr = arr.__array_interface__["data"][0]
            self.cudart.cudaMemcpy(
                ctypes.c_void_p(buf_addr + offset),
                ctypes.c_void_p(int(dev_addr.value) + dev_offset),
                ctypes.c_size_t(nbytes),
                ctypes.c_int(1),
            )

        os.rename(tmp_path, path)

        return metadata

    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        path = self._key_to_path(key)
        size = memory_obj.get_size()
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        with self.hot_wlock:
            self.hot_cache[key] = DiskCacheMetadata(path, size, shape, dtype)

    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        logger.debug(f"Submitting prefetch task for key {key}")
        with self.hot_rlock:
            entry = self.hot_cache.get(key)
        if entry is None:
            return None

        path = entry.path
        dtype = entry.dtype
        shape = entry.shape
        assert dtype is not None
        assert shape is not None
        return asyncio.run_coroutine_threadsafe(
            self._async_load_bytes_from_disk(key, path, dtype, shape), self.loop
        )

    async def _async_load_bytes_from_disk(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
    ):
        logger.debug(f"Loading bytes from disk for key {key} at path {path}")
        # TODO: 基于cuFileBatchIOSubmit或cuFileReadAsync实现异步IO
        # cuFileReadAsync需要使用基于CUstream的流式IO
        return self._load_bytes_from_disk(key, path, dtype, shape)

    def _load_bytes_from_disk(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> Optional[MemoryObj]:
        logger.debug(f"Loading bytes from disk for key {key} at path {path}")
        # 分配内存对象
        memory_obj = self.memory_allocator.allocate(shape, dtype)
        if memory_obj is None:
            logger.debug("Memory allocation failed during sync disk load.")
            return None
        assert memory_obj.tensor is not None
        assert memory_obj.tensor.is_cuda
        assert torch.device(self.dst_device) == torch.device(memory_obj.tensor.device)

        # 使用cuFile API或POSIX IO加载数据
        offset = _METADATA_MAX_SIZE
        if self.cufile_base_pointer is None:
            addr = ctypes.c_void_p(memory_obj.tensor.data_ptr())
            dev_offset = 0
        else:
            addr = ctypes.c_void_p(self.cufile_base_pointer)
            dev_offset = memory_obj.metadata.address
        # 若存在，cufile_base_pointer与memory_allocator.base_pointer相等
        # addr指向GPU显存的cuFile首地址，或memory_obj的首地址
        ret = self._load_pfs(path, addr, memory_obj.get_size(), offset, dev_offset)
        if ret != memory_obj.get_size():
            if ret < 0:
                logger.error(
                    f"Error loading {path}: ret: {ret} removing entry from cache"
                )
                with self.hot_wlock:
                    self.hot_cache.pop(key)
            else:
                # TODO: we should probably count errors and
                # remove the entry if it's a persistent problem.
                logger.error(
                    f"Error loading {path}: got only {ret} bytes "
                    f"out of {memory_obj.get_size()}, ignoring"
                )
            memory_obj.ref_count_down()
            return None
        return memory_obj

    def _load_pfs(
        self,
        file_path: str,
        gpu_pointer: ctypes.c_void_p,
        size_in_bytes: int,
        file_offset: int,
        dev_offset: int,
    ) -> int:
        # Read data from disk into a GPU buffer
        logger.debug(f"Loading data from PFS file {file_path} into GPU memory at {gpu_pointer.value}, size {size_in_bytes}, file_offset {file_offset}, dev_offset {dev_offset}")
        if self.cufile:
            with self.cufile.CuFile(
                file_path, "r", use_direct_io=self.use_direct_io
            ) as f:
                return f.read(
                    gpu_pointer,
                    size_in_bytes,
                    file_offset=file_offset,
                    dev_offset=dev_offset,
                )
        else:
            fd = os.open(file_path, os.O_RDONLY)
            file_size = os.fstat(fd).st_size
            mm = mmap.mmap(
                fd,
                file_size,
                prot=mmap.PROT_READ,
                flags=mmap.MAP_PRIVATE | mmap.MAP_POPULATE,
            )
            os.close(fd)

            arr = np.frombuffer(mm, dtype=np.uint8)
            addr = arr.__array_interface__["data"][0]

            res = self.cudart.cudaMemcpy(
                ctypes.c_void_p(int(gpu_pointer.value) + dev_offset),
                ctypes.c_void_p(addr + file_offset),
                ctypes.c_size_t(size_in_bytes),
                ctypes.c_int(1),
            )

            if res != 0:
                raise RuntimeError(f"cudaMemcpy failed with code {res}")
            del arr
            mm.close()
            return size_in_bytes

    def get_blocking(self, key):
        logger.debug(f"Blocking-get data for key {key} from PFS backend")
        with self.hot_rlock:
            entry = self.hot_cache.get(key)
        if entry is None:
            logger.debug(f"Key {key} not found in hot cache, reading from metadata")
            entry = self._read_metadata(key)
            if entry is None:
                logger.warning(f"Key {key} not found in hot cache or metadata")
                return None

        # 从磁盘加载数据
        return self._load_bytes_from_disk(
            key, entry.path, entry.dtype, entry.shape
        )

    def get_non_blocking(self, key):
        return self.submit_prefetch_task(key)

    # NOTE: 默认实现为每个key串行调用get_blocking
    # def batched_get_blocking(self, keys):
    #     return super().batched_get_blocking(keys)

    def pin(self, key):
        # TODO: How is metadata.pin() working?
        # metadata.pin() prevents it from being evicted
        logger.debug(f"Pinning key {key} in PFS backend")
        with self.hot_rlock:
            if key not in self.hot_cache:
                logger.warning(f"Key {key} not found in hot cache, cannot pin")
                return False
            self.hot_cache[key].pin()
        return True

    def unpin(self, key):
        logger.debug(f"Unpinning key {key} in PFS backend")
        with self.hot_rlock:
            if key not in self.hot_cache:
                logger.warning(f"Key {key} not found in hot cache, cannot unpin")
                return False
            self.hot_cache[key].unpin()
        return True

    def close(self):
        logger.info("Closing PfsBackend")
        for task in self.save_metadata_tasks:
            asyncio.wait(task)
        # TODO: Join the event loop
