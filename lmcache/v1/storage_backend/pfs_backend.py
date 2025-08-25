from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.memory_management import MemoryAllocatorInterface, MemoryObj, MemoryFormat
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
from dataclasses import dataclass, field
import time
import threading
import json
import aiofiles

logger = init_logger(__name__)

_METADATA_FILE_SUFFIX = ".metadata"
_DATA_FILE_SUFFIX = ".kvcache.safetensors"
_METADATA_VERSION = 1
_METADATA_MAX_SIZE = 4096  # reserve 4K for metadata.

def singleton(cls):
    instances = {}
    def getinstance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return getinstance

@singleton
@dataclass
class Profiler:
    # Profiler for the connector
    from lmcache.utils import thread_safe
    layerwise_load_time: dict[str, float] = field(default_factory=dict)
    layerwise_save_time: dict[str, float] = field(default_factory=dict)
    metric_dict: dict[str, float] = field(default_factory=dict)
    lock_dict: dict[str, threading.Lock] = field(default_factory=dict)
    is_running: bool = True
    @thread_safe
    def update_metric(self, metric_name: str, value: float):
        if metric_name not in self.metric_dict:
            self.metric_dict[metric_name] = 0.0
            self.lock_dict[metric_name] = threading.Lock()
        with self.lock_dict[metric_name]:
            self.metric_dict[metric_name] += value
    def summary(self):
        """Print the summary of the Profiler()."""
        res = {**self.metric_dict}
        for layer_name, load_time in self.layerwise_load_time.items():
            res[f"load_time_{layer_name}"] = load_time
        for layer_name, save_time in self.layerwise_save_time.items():
            res[f"save_time_{layer_name}"] = save_time
        logger.info("Profiler summary: %s", res)
        return res
    def print(self):
        while self.is_running:
            time.sleep(60)
            self.summary()
    def shutdown(self):
        self.is_running = False
        with open("/home/qiuan/workspace/pfs_backend_Profiler.json", "w+") as f:
            json.dump(self.summary(), f)


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
        # TODO(pqa): memory_allocator
        self.memory_allocator = memory_allocator
        self.dst_device = dst_device

        assert config.pfs_path is not None, "PFS path must be set in the configuration"
        self.pfs_path = config.pfs_path

        # layerwise
        self.use_layerwise = config.use_layerwise

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

        # TODO: load挂载点已存在的数据
        # asyncio.run_coroutine_threadsafe(self._scan_metadata(), self.loop)

        # Profiling
        self.profile_thread = threading.Thread(target=Profiler().print, daemon=True)
        self.profile_thread.start()

    def contains(self, key, pin = False):
        logger.debug(f"Checking if key {key} exists in PFS backend with pin={pin}")
        # start_time = time.perf_counter()
        with self.hot_rlock:
            if key not in self.hot_cache:
                return False
            if pin:
                self.hot_cache[key].pin()
            return True
        # Profiler().update_metric("lookup", time.perf_counter() - start_time)
        logger.debug(f"Key {key} exists in PFS backend: {res}")
        return res

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

        start_time = time.perf_counter()
        memory_obj.ref_count_up()

        with self.put_wlock:
            if key in self.put_tasks:
                logger.warning(f"Key {key} is already in put tasks, skipping")
                return None
            self.put_tasks.add(key)
        Profiler().update_metric("put_overhead", time.perf_counter() - start_time)

        # 异步提交put任务
        return asyncio.run_coroutine_threadsafe(
            self._async_save_bytes_to_disk(key, memory_obj), self.loop
        )

    @torch.inference_mode()
    async def _async_save_bytes_to_disk(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> None:
        """
        Convert KV to bytes and async store bytes to disk.
        """
        assert memory_obj.tensor is not None

        start_time = time.perf_counter()
        path = self._key_to_path(key)
        Profiler().update_metric("put_overhead", time.perf_counter() - start_time)

        start_time = time.perf_counter()
        async with aiofiles.open(path, "wb") as f:
            await f.write(memory_obj.byte_array)
        Profiler().update_metric("put_async_write", time.perf_counter() - start_time)

        start_time = time.perf_counter()
        self.insert_key(key, memory_obj)
        memory_obj.ref_count_down()
        Profiler().update_metric("insert", time.perf_counter() - start_time)

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

        start_time = time.perf_counter()
        metadata = pack_metadata(kv_chunk, lmcache_version=_METADATA_VERSION)
        Profiler().update_metric("put_gen_metadata", time.perf_counter() - start_time)

        start_time = time.perf_counter()
        # 分配临时文件路径
        tmp_path = path + tmp

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
            # 文件首部写入4KB元数据
            with open(tmp_path, "wb") as f:
                f.write(metadata)
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
            # # fd = os.open(tmp_path, os.O_RDWR | os.O_DIRECT)
            # fd = os.open(tmp_path, os.O_RDWR | os.O_CREAT)
            # os.ftruncate(fd, nbytes + offset) # 预分配文件空间
            # mm = mmap.mmap(
            #     fd,
            #     nbytes + offset,
            #     prot=mmap.PROT_WRITE,
            #     flags=mmap.MAP_SHARED,
            # )
            # os.close(fd)

            # Save disk tensor
            with open(tmp_path, "wb") as f:
                f.write(metadata)
                # arr = np.frombuffer(f, dtype=np.uint8)
                # buf_addr = arr.__array_interface__["data"][0]
                # self.cudart.cudaMemcpy(
                #     ctypes.c_void_p(buf_addr + offset),
                #     ctypes.c_void_p(int(dev_addr.value) + dev_offset),
                #     ctypes.c_size_t(nbytes),
                #     ctypes.c_int(2),
                # )
                # 3fs file can't be mmaped
                try:
                    kv_chunk.detach().to(torch.float16).cpu().numpy().tofile(f)
                except Exception as e:
                    logger.error(f"Failed to write tensor data to file {tmp_path}: {e}")
                    raise
        Profiler().update_metric("put_write", time.perf_counter() - start_time)

        start_time = time.perf_counter()
        os.rename(tmp_path, path)
        Profiler().update_metric("put_rename", time.perf_counter() - start_time)

        logger.info(f"Saved data to PFS file {path}")

        return metadata

    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        path = self._key_to_path(key)
        size = memory_obj.get_size()
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        # layerwise-optimize needs fmt to be KV_T2D
        fmt = memory_obj.metadata.fmt
        with self.hot_wlock:
            self.hot_cache[key] = DiskCacheMetadata(path, size, shape, dtype, fmt)

    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        logger.debug(f"Submitting prefetch task for key {key}")
        with self.hot_rlock:
            if key not in self.hot_cache:
                logger.warning(f"Key {key} not found in hot cache, cannot prefetch")
                return None
            path = self.hot_cache[key].path
            dtype = self.hot_cache[key].dtype
            shape = self.hot_cache[key].shape
            fmt = self.hot_cache[key].fmt
        assert dtype is not None
        assert shape is not None
        return asyncio.run_coroutine_threadsafe(
            self._async_load_bytes_from_disk(path, dtype, shape, fmt), self.loop
        )

    @torch.inference_mode()
    async def _async_load_bytes_from_disk(
        self,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
        fmt: MemoryFormat
    ):
        logger.debug(f"Async loading bytes from disk at path {path}")
        # start = time.perf_counter()

        memory_obj = self.memory_allocator.allocate(shape, dtype, fmt)
        if memory_obj is None:
            logger.warning("Memory allocation failed during async disk load.")
            return None

        # Profiler().update_metric("async-load-allocation", time.perf_counter() - start)
        # start = time.perf_counter()

        buffer = memory_obj.byte_array
        async with aiofiles.open(path, "rb") as f:
            await f.readinto(buffer)
        
        # Profiler().update_metric("async-load-read", time.perf_counter() - start)

        return memory_obj

    @torch.inference_mode()
    def _load_bytes_from_disk(
        self,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
        fmt: MemoryFormat
    ) -> Optional[MemoryObj]:
        # logger.debug(f"Loading bytes from disk for key {key} at path {path}")
        # 分配内存对象
        # start_time = time.perf_counter()
        # if self.use_layerwise:
        #     memory_obj = self.memory_allocator.allocate(shape, dtype, fmt=MemoryFormat.KV_T2D)
        # else:
        #     memory_obj = self.memory_allocator.allocate(shape, dtype)
        # if memory_obj is None:
        #     logger.debug("Memory allocation failed during sync disk load.")
        #     return None
        # assert memory_obj.tensor is not None
        # assert memory_obj.tensor.is_cuda
        # assert torch.device(self.dst_device) == torch.device(memory_obj.tensor.device)
        # Profiler().update_metric("allocation", time.perf_counter() - start_time)

        logger.debug(f"Loading data from PFS file {path} into memory")

        memory_obj = self.memory_allocator.allocate(shape, dtype, fmt)
        if memory_obj is None:
            logger.debug("Memory allocation failed during sync disk load.")
            return None

        buffer = memory_obj.byte_array
        with open(path, "rb") as f:
            f.readinto(buffer)

        return memory_obj

    @torch.inference_mode()
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
                flags=mmap.MAP_PRIVATE | mmap.MAP_POPULATE,
                prot=mmap.PROT_READ,
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
        start_time = time.perf_counter()
        with self.hot_rlock:
            entry = self.hot_cache.get(key)
        if entry is None:
            logger.debug(f"Key {key} not found in hot cache, reading from metadata")
            entry = self._read_metadata(key)
            if entry is None:
                logger.warning(f"Key {key} not found in hot cache or metadata")
                return None

        # 从磁盘加载数据
        res = self._load_bytes_from_disk(
            key, entry.path, entry.dtype, entry.shape
        )
        Profiler().update_metric("blocking_get", time.perf_counter() - start_time)
        return res

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
        # for task in self.save_metadata_tasks:
        #     asyncio.wait(task)
        # TODO: Join the event loop
        # import json
        # with open("pfs_backend_Profiler.json", "w") as f:
        #     json.dump(Profiler().print_summary(), f)
        Profiler().shutdown()
        self.profile_thread.join()
