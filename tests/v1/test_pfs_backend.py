# /usr/bin/env python

# Standard
from pathlib import Path
import os
import asyncio
import threading
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.storage_backend import CreateStorageBackends
from lmcache.v1.memory_management import CuFileMemoryAllocator

# Third Party
import pytest

def test_pfs_backend_sanity():
    # Hardcoded backend config arguments
    BASE_DIR = Path(__file__).parent
    # PFS_PATH = "/tmp/pfs/test-cache"
    BACKEND_NAME = "PfsBackend"
    # Generate a CacheEngineKey
    TEST_KEY = CacheEngineKey(
        fmt="vllm",
        model_name="meta-llama/Llama-3.1-70B-Instruct",
        world_size=8,
        worker_id=0,
        chunk_hash="e3229141e680fb413d2c5d3ebb416c4ad300d381e309fc9e417757b91406c157",
    )

    try:
        # 0 create PFS directory
        # os.makedirs(PFS_PATH, exist_ok=True)

        # 1. create backends (PFS and CPU),
        # since lmcache defaultly creates a CPU backend as buffer allocator

        # 1.a create backend config
        config = LMCacheEngineConfig.from_file(BASE_DIR/"data"/"pfs.yaml")

        # 1.b create the event loop
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever)
        thread.start()

        # 1.c finally create backends,
        # P.S. we create a cufile memory allocator
        # before creating backends
        backends = CreateStorageBackends(
            config,
            None, # NixlBackend and RemoteBackend need metadata
            loop,
            LMCacheEngineBuilder._Create_memory_allocator(config, None),
            "cuda",
        )
        assert BACKEND_NAME in backends

        # 1.d check if the backend is created
        # and if the memory allocator is set
        backend = backends[BACKEND_NAME]
        assert backend is not None
        assert backend.memory_allocator is not None
        assert isinstance(backend.memory_allocator, CuFileMemoryAllocator)

        # 2. query the key
        # and check if it does not exist
        # assert not backend.contains(TEST_KEY, False) # as the metatadata file wasn't deleted
        assert not backend.exists_in_put_tasks(TEST_KEY)

        # 3. create a tensor
        memory_obj = backend.memory_allocator.allocate(
            [2048, 2048], dtype=torch.uint8
        )
        assert memory_obj is not None

        # 4. insert the tensor and check the status of the key
        future = backend.submit_put_task(TEST_KEY, memory_obj)
        assert future is not None
        assert backend.exists_in_put_tasks(TEST_KEY)
        # assert not backend.contains(TEST_KEY, False)
        future.result()  # wait for the task to complete
        assert backend.contains(TEST_KEY, False)
        assert not backend.exists_in_put_tasks(TEST_KEY)

        # 5. query the key again and check if it exists

        # 5.a synchronously
        returned_memory_obj = backend.get_blocking(TEST_KEY)
        assert returned_memory_obj is not None
        assert returned_memory_obj.get_size() == memory_obj.get_size()
        assert returned_memory_obj.get_shape() == memory_obj.get_shape()
        assert returned_memory_obj.get_dtype() == memory_obj.get_dtype()

        # 5.b asynchronously
        future = backend.get_non_blocking(TEST_KEY)
        assert future is not None
        returned_memory_obj = future.result()
        assert returned_memory_obj is not None
        assert returned_memory_obj.get_size() == memory_obj.get_size()
        assert returned_memory_obj.get_shape() == memory_obj.get_shape()
        assert returned_memory_obj.get_dtype() == memory_obj.get_dtype()
    # except OSError as e:
    #     pytest.fail(f"Failed to create PFS directory: {e}")
    finally:
        # with the backend closed, the loop should stop as well
        if loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread.is_alive():
            thread.join()
        # backend.close()

        # if os.path.exists(PFS_PATH):
        #     os.rmdir(PFS_PATH)
