"""
created by yansiyu01@baidu.com at 2025/07/31
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import logging
import os
import queue
import socket
import time
import threading
import requests
import zmq
import ctypes
import struct
import numpy as np
import numpy.typing as npt

from aiohttp import web
from collections import defaultdict
from functools import cache
from typing import Dict, List, Optional, Union, Tuple

from sglang.srt.disaggregation.base.conn import (
    BaseKVBootstrapServer,
    BaseKVManager,
    BaseKVReceiver,
    BaseKVSender,
    KVArgs,
    KVPoll,
)
from sglang.srt.disaggregation.common.utils import (
    FastQueue,
    group_concurrent_contiguous,
)
from sglang.srt.disaggregation.p2p.transfer_engine import P2PTransferEngine
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    format_tcp_address,
    get_free_port,
    get_int_env_var,
    get_ip,
    get_local_ip_auto,
    is_valid_ipv6_address,
)

logger = logging.getLogger(__name__)


class P2PTransferError(Exception):
    """
    P2PTransferError 异常类，表示 P2P 传输过程中发生的错误
    """
    def __init__(self, bootstrap_room: int, failure_reason: str):
        """
        初始化
        """
        super().__init__(failure_reason)
        self.bootstrap_room = bootstrap_room
        self.failure_reason = failure_reason

    def __str__(self):
        """
        返回 P2PTransferError 对象的字符串表示形式
        """
        return f"P2PTransferError(bootstrap_room={self.bootstrap_room}): {self.failure_reason}"


# prefill
@dataclasses.dataclass
class TransferKVChunk:
    """
    KVCache 数据的元信息
    """
    room: int
    prefill_kv_indices: npt.NDArray[np.int32]
    index_slice: slice
    is_last: bool
    prefill_aux_index: Optional[int]


# decode
@dataclasses.dataclass
class TransferInfo:
    """
    Decode 接收 KVCache 所需的目标传输信息
    """
    room: int
    endpoint: str
    dst_port: int
    p2p_session_id: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    required_dst_info_num: int
    is_dummy: bool

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        """
        从 ZMQ 消息中解析并创建一个新的实例
        """
        if msg[4] == b"" and msg[5] == b"":
            is_dummy = True
            dst_kv_indices = np.array([], dtype=np.int32)
            dst_aux_index = None
        else:
            dst_kv_indices = np.frombuffer(msg[4], dtype=np.int32)
            dst_aux_index = int(msg[5].decode("ascii"))
            is_dummy = False
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            p2p_session_id=msg[3].decode("ascii"),
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
            required_dst_info_num=int(msg[6].decode("ascii")),
            is_dummy=is_dummy,
        )


# decode
@dataclasses.dataclass
class KVArgsRegisterInfo:
    """
    KVCache 注册信息
    """
    room: str
    endpoint: str
    dst_port: int
    p2p_session_id: str
    dst_kv_ptrs: list[int] 
    dst_aux_ptrs: list[int]

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        """
        从 ZMQ 消息中解析并创建一个新的实例
        """
        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            p2p_session_id=msg[3].decode("ascii"),
            dst_kv_ptrs=[msg[4][i*64:(i+1)*64] for i in range(len(msg[4])//64)],
            dst_aux_ptrs=[msg[5][i*64:(i+1)*64] for i in range(len(msg[5])//64)],
        )


class AuxDataCodec:
    """
    AuxDataCodec 序列化和反序列化辅助数据缓冲区的工具类
    """

    @staticmethod
    def serialize_data_from_buffer(src_addr, data_length):
        """
        从指定的内存地址读取指定长度的数据，并将其转换为字节串
        """
        buffer = (ctypes.c_byte * data_length).from_address(src_addr)
        return bytes(buffer)

    @staticmethod
    def deserialize_data_to_buffer(kv_args, buffer_index, aux_index, data):
        """
        将指定的数据写入到指定的内存地址中
        """
        dst_aux_ptr = kv_args.aux_data_ptrs[buffer_index]
        item_len = kv_args.aux_item_lens[buffer_index]
        dst_addr = dst_aux_ptr + item_len * aux_index
        buffer = (ctypes.c_byte * len(data)).from_address(dst_addr)
        buffer[:] = data
        return


class P2PKVManager(BaseKVManager):
    """
    P2PKVManager 类
    """
    AUX_DATA_HEADER = b"AUX_DATA"

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        """
        初始化 P2PKVManager 
        """
        self.kv_args = args
        self.disaggregation_mode = disaggregation_mode
        self.server_args = server_args
        self.is_mla_backend = is_mla_backend
        self.local_ip = get_local_ip_auto()

        self.p2p_engine = P2PTransferEngine(self.local_ip, self.kv_args.gpu_id)
        self.buffer_handles = {}
        self.decode_physical_gpu_ids = {}
        
        self.bootstrap_port = server_args.disaggregation_bootstrap_port
        self.dist_init_addr = server_args.dist_init_addr
        self.tp_size = server_args.tp_size
        self.dp_size = server_args.dp_size
        self.enable_dp_attention = server_args.enable_dp_attention
        
        if not server_args.enable_dp_attention and server_args.dp_size != 1:
            raise ValueError(
                "If dp_attention is not enabled, dp size must be 1 in disaggregation mode."
            )
        
        self.request_status = {}
        self.rank_port = get_free_port()
        self.server_socket = zmq.Context().socket(zmq.PULL)
        self.register_buffer_to_engine()
        
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.transfer_infos: Dict[int, Dict[str, TransferInfo]] = {}
            self.decode_kv_args_table: Dict[str, KVArgsRegisterInfo] = {} 
            self.start_prefill_thread()
            self._register_to_bootstrap()
            self.session_failures = defaultdict(int) 
            self.failed_sessions = set()
            self.session_lock = threading.Lock()
            cpu_count = os.cpu_count() 
            transfer_thread_pool_size = get_int_env_var(
                "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE", 
                min(max(4, int(0.75 * cpu_count) // 8), 12), 
            )
            transfer_queue_size = get_int_env_var("SGLANG_DISAGGREGATION_QUEUE_SIZE", 4) 
            self.transfer_queues: List[FastQueue] = [ 
                FastQueue() for _ in range(transfer_queue_size) 
            ]
            assert transfer_thread_pool_size >= transfer_queue_size, (
                f"The environment variable SGLANG_DISAGGREGATION_THREAD_POOL_SIZE={transfer_thread_pool_size} must be " 
                f"greater than or equal to SGLANG_DISAGGREGATION_QUEUE_SIZE={transfer_queue_size}." 
            )
            self.executors = [ 
                concurrent.futures.ThreadPoolExecutor(
                    transfer_thread_pool_size // transfer_queue_size 
                )
                for _ in range(transfer_queue_size)
            ]
            for queue, executor in zip(self.transfer_queues, self.executors): 
                threading.Thread(
                    target=self.transfer_worker, args=(queue, executor), daemon=True
                ).start()

            self.bootstrap_time_out = get_int_env_var(
                "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT", 30 
            )
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.heartbeat_failures = {}
            self.session_pool = defaultdict(requests.Session)
            self.session_pool_lock = threading.Lock()
            self.addr_to_rooms_tracker = defaultdict(set) 
            self.connection_lock = threading.Lock() 
            self.heartbeat_interval = max( 
                float(os.getenv("SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL", 5.0)), 2.0 
            )
            self.max_failures = max( 
                get_int_env_var("SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE", 2), 1
            )
            self.start_decode_thread()
            self.connection_pool: Dict[str, Dict[str, Union[str, int]]] = {}
            self.prefill_tp_size_table: Dict[str, int] = {} 
            self.prefill_dp_size_table: Dict[str, int] = {} 
            
        else:
            raise ValueError(f"Unsupported DisaggregationMode: {self.disaggregation_mode}")
        
        self.failure_records: Dict[int, str] = {}
        self.failure_lock = threading.Lock()
        self.handle_size = 64
        # KVCache 传输统计，以 room (每个请求) 为 key，保存 { "kv_bytes": int, "kv_time_ms": float, "kv_tokens": int }
        self.kv_transfer_stats: Dict[int, Dict[str, float | int]] = {}
        self.kv_transfer_stats_lock = threading.Lock()
        self.enable_kvcache_log = get_int_env_var("SGLANG_KVCACHE_LOG", 0) 

    def register_buffer_to_engine(self):
        """
        将KV缓冲区注册到引擎中
        """
        logger.info("Start registering KV buffers to P2P engines.")
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            self.kv_handles = []
            for ptr in self.kv_args.kv_data_ptrs:
                kv_handle = self.p2p_engine.register_buffer(ptr)
                self.kv_handles.append(kv_handle)
        
    def _connect(self, endpoint: str, is_ipv6: bool = False):
        """
        连接到指定的ZeroMQ端点
        """
        socket = zmq.Context().socket(zmq.PUSH)
        if is_ipv6:
            socket.setsockopt(zmq.IPV6, 1)
        socket.connect(endpoint)
        return socket
    
    def _bind_server_socket(self):
        """
        绑定服务器socket
        """
        self.server_socket.bind(format_tcp_address(self.local_ip, self.rank_port))

    def start_prefill_thread(self):
        """
        启动预填充线程
        """
        self.rank_port = get_free_port()
        self._bind_server_socket()

        def bootstrap_thread():
            """
            预填充线程函数
            """
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                room = waiting_req_bytes[0].decode("ascii")
                p2p_session_id = waiting_req_bytes[3].decode("ascii")
                if room == "None":
                    physical_gpu_id = int(waiting_req_bytes[6].decode("ascii"))
                    self.decode_physical_gpu_ids[p2p_session_id] = physical_gpu_id

                    handles_bytes = waiting_req_bytes[4]
                    num_handles = len(handles_bytes) // self.handle_size

                    dst_kv_ptrs = [
                        handles_bytes[i * self.handle_size:(i + 1) * self.handle_size]
                        for i in range(num_handles)
                    ]

                    for kv_handle in dst_kv_ptrs:
                        result = self.p2p_engine.register_d_handle(kv_handle)

                    aux_ptrs = list(struct.unpack(f"{len(waiting_req_bytes[5]) // 8}Q", waiting_req_bytes[5]))

                    self.decode_kv_args_table[p2p_session_id] = KVArgsRegisterInfo(
                        room=room,
                        endpoint=waiting_req_bytes[1].decode("ascii"),
                        dst_port=int(waiting_req_bytes[2].decode("ascii")),
                        p2p_session_id=p2p_session_id,
                        dst_kv_ptrs=dst_kv_ptrs,
                        dst_aux_ptrs=aux_ptrs,
                    )
                else:
                    # 传输请求
                    required_dst_info_num = int(waiting_req_bytes[6].decode("ascii"))
                    room = int(waiting_req_bytes[0].decode("ascii"))
                    if room not in self.transfer_infos:
                        self.transfer_infos[room] = {}
                    
                    key = waiting_req_bytes[3].decode("ascii")
                    self.transfer_infos[room][key] = TransferInfo.from_zmq(waiting_req_bytes)
                    if len(self.transfer_infos[room]) == required_dst_info_num:
                        self.update_status(room, KVPoll.WaitingForInput)
        
        threading.Thread(target=bootstrap_thread, daemon=True).start()

    def send_kvcache(
        self,
        req: TransferInfo, 
        p2p_session_id: str,  
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: List[bytes],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        """
        发送 KVCache
        更新：使用 p2p_engine.transfer_many 批量发送 KVCache（SGLANG_P2P_BATCH_LIMIT设置batch大小）
        """
        try:
            dst_physical_gpu_id = self.decode_physical_gpu_ids.get(p2p_session_id)
        
            if dst_physical_gpu_id is None:
                logger.error(f"Physical GPU ID not found for session {p2p_session_id}")
                return 1

            src_physical_gpu_id = self.kv_args.gpu_id # 源 GPU

            batch_limit = int(os.getenv("SGLANG_P2P_BATCH_LIMIT", "512") or "512")
            if batch_limit <= 0:
                batch_limit = 512
            timeout_s = float(os.getenv("SGLANG_P2P_TRANSFER_TIMEOUT", "60") or "60")
            
            prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
                prefill_kv_indices, dst_kv_indices
            )

            batch_src_ptrs: list[int] = []
            batch_src_devs: list[int] = []
            batch_dst_handles: list[bytes] = []
            batch_dst_devs: list[int] = []
            batch_offsets: list[int] = []
            batch_lengths: list[int] = []

            total_tokens = 0
            start_time = time.perf_counter()

            num_layers = len(self.kv_args.kv_data_ptrs)
            for layer_id in range(num_layers):
                base_ptr = int(self.kv_args.kv_data_ptrs[layer_id])
                item_len = int(self.kv_args.kv_item_lens[layer_id])  # 一个block的字节数
                dst_handle = dst_kv_ptrs[layer_id]

                for prefill_block, dst_block in zip(prefill_kv_blocks, dst_kv_blocks):
                    if not prefill_block or not dst_block:
                        continue
                    first_src_block = int(prefill_block[0])
                    first_dst_block = int(dst_block[0])
                    length_bytes = int(len(prefill_block) * item_len)

                    src_ptr = base_ptr + first_src_block * item_len
                    dst_off = first_dst_block * item_len

                    batch_src_ptrs.append(src_ptr)
                    batch_src_devs.append(int(src_physical_gpu_id))
                    batch_dst_handles.append(dst_handle)
                    batch_dst_devs.append(int(dst_physical_gpu_id))
                    batch_offsets.append(dst_off)
                    batch_lengths.append(length_bytes)

                    total_tokens += len(prefill_block)

            if not batch_src_ptrs:
                logger.debug("No KV transfer tasks to send.")
                return 0

            # batch 提交（transfer_many），返回一个句柄
            handles = []
            total_bytes = 0
            for i in range(0, len(batch_src_ptrs), batch_limit):
                j = i + batch_limit
                sub_src_ptrs = batch_src_ptrs[i:j]
                sub_src_devs = batch_src_devs[i:j]
                sub_dst_handles = batch_dst_handles[i:j]
                sub_dst_devs = batch_dst_devs[i:j]
                sub_offsets = batch_offsets[i:j]
                sub_lengths = batch_lengths[i:j]

                if hasattr(self.p2p_engine, "transfer_many"):
                    h = self.p2p_engine.transfer_many(
                        src_ptrs=sub_src_ptrs,
                        src_devs=sub_src_devs,
                        dst_handles=sub_dst_handles,
                        dst_devs=sub_dst_devs,
                        dst_offsets=sub_offsets,
                        lengths=sub_lengths,
                    )
                    handles.append(h)
                else:
                    sub_handles = []
                    for sp, sd, dh, dd, off, ln in zip(
                        sub_src_ptrs, sub_src_devs, sub_dst_handles, sub_dst_devs, sub_offsets, sub_lengths
                    ):
                        sub_handles.append(self.p2p_engine.transfer(sp, sd, dh, dd, off, ln))

                    class _BatchHandle:
                        def __init__(self, hs): self._hs = hs
                        def is_done(self): return all(h.is_done() for h in self._hs)
                        def wait(self):
                            for h in self._hs:
                                if hasattr(h, "wait"): h.wait()
                                else:
                                    while not h.is_done(): time.sleep(0.001)
                    handles.append(_BatchHandle(sub_handles))

                total_bytes += sum(sub_lengths)

            # 统一等待所有batch完成（轮询）
            deadline = time.perf_counter() + timeout_s
            while True:
                remaining = [h for h in handles if not h.is_done()]
                if not remaining:
                    break
                if time.perf_counter() > deadline:
                    logger.error(f"P2P transfer_many timeout after {timeout_s}s")
                    return 1
                time.sleep(0.002)

            # 统计信息
            total_time_ms = (time.perf_counter() - start_time) * 1000.0
            with self.kv_transfer_stats_lock:
                self.kv_transfer_stats[req.room] = {
                    "kv_bytes": int(total_bytes),
                    "kv_time_ms": float(total_time_ms),
                    "kv_tokens": int(total_tokens),
                }

            return 0

        except Exception as e:
            logger.exception(f"P2P KV cache transfer failed: {e}")
            return 1

    def send_aux(
        self,
        req: TransferInfo,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
    ):
        """
        发送AUX
        """
        overall_ret = 0
        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        for i in range(len(prefill_aux_ptrs)):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            data = AuxDataCodec.serialize_data_from_buffer(src_addr, length)

            ret = self.send_aux_data_to_endpoint(
                remote=req.endpoint,
                dst_port=req.dst_port,
                room=req.room,
                buffer_index=i,
                aux_index=req.dst_aux_index,
                data=data,
            )
        if ret != 0:
            overall_ret = 1
        return overall_ret

    def send_aux_data_to_endpoint(
        self,
        remote: str,
        dst_port: int,
        room: int,
        buffer_index: int,
        aux_index: int,
        data: bytes,
    ):
        """
        发送AUX数据 
        """
        try:
            socket = self._connect(
                format_tcp_address(remote, dst_port), is_ipv6=is_valid_ipv6_address(remote)
            )

            socket.send_multipart(
                [
                    P2PKVManager.AUX_DATA_HEADER,
                    str(room).encode("ascii"),
                    str(buffer_index).encode("ascii"),
                    str(aux_index).encode("ascii"),
                    struct.pack(">I", len(data)),
                    data,
                ]
            )
            return 0
        except Exception as e:
            logger.exception(f"Failed to send aux: {e}")
            return 1

    def sync_status_to_decode_endpoint(
        self, remote: str, dst_port: int, room: int, status: int
    ):
        """
        同步状态
        """
        self._connect(
            format_tcp_address(remote, dst_port), is_ipv6=is_valid_ipv6_address(remote)
        ).send_multipart(
            [
                str(room).encode("ascii"),
                str(status).encode("ascii"),
            ]
        )

    def transfer_worker(
        self, queue: FastQueue, executor: concurrent.futures.ThreadPoolExecutor
    ):        
        """
        数据传输
        """
        while True:
            try:
                kv_chunk: TransferKVChunk = queue.get()
                reqs_to_be_processed = (
                    self.transfer_infos[kv_chunk.room].values()
                    if kv_chunk.room in self.transfer_infos
                    else []
                )
                polls = []
                dst_ranks_infos = []
                for req in reqs_to_be_processed:
                    if not req.is_dummy:
                        # Early exit if the request has failed
                        with self.session_lock:
                            if req.p2p_session_id in self.failed_sessions:
                                self.record_failure(
                                    kv_chunk.room,
                                    f"Decode instance could be dead, remote p2p session {req.p2p_session_id} is not alive",
                                )
                                self.update_status(kv_chunk.room, KVPoll.Failed)
                                self.sync_status_to_decode_endpoint(
                                    req.endpoint,
                                    req.dst_port,
                                    req.room,
                                    KVPoll.Failed,
                                )
                                break

                        chunked_dst_kv_indice = req.dst_kv_indices[kv_chunk.index_slice]

                        if len(chunked_dst_kv_indice) < len(
                            kv_chunk.prefill_kv_indices
                        ):
                            kv_chunk.prefill_kv_indices = kv_chunk.prefill_kv_indices[
                                : len(chunked_dst_kv_indice)
                            ]
                            logger.warning(
                                f"len(chunked_dst_kv_indice) = {len(chunked_dst_kv_indice)}, len(kv_chunk.prefill_kv_indices) = {len(kv_chunk.prefill_kv_indices)}"
                            )

                        ret = self.send_kvcache(
                            req,
                            req.p2p_session_id,
                            kv_chunk.prefill_kv_indices,
                            self.decode_kv_args_table[
                                req.p2p_session_id
                            ].dst_kv_ptrs,
                            chunked_dst_kv_indice,
                            executor,
                        )
                        if ret != 0:
                            with self.session_lock:
                                self.session_failures[req.p2p_session_id] += 1
                                if self.session_failures[req.p2p_session_id] >= 1:
                                    self.failed_sessions.add(req.p2p_session_id)
                                    logger.error(
                                        f"Session {req.p2p_session_id} failed."
                                    )
                            self.record_failure(
                                kv_chunk.room,
                                f"Failed to send kv chunk of {kv_chunk.room} to {req.endpoint}:{req.dst_port}",
                            )
                            self.update_status(kv_chunk.room, KVPoll.Failed)
                            self.sync_status_to_decode_endpoint(
                                req.endpoint, req.dst_port, req.room, KVPoll.Failed
                            )
                            break

                        if kv_chunk.is_last:
                            # Only the last chunk we need to send the aux data
                            ret = self.send_aux(
                                req,
                                kv_chunk.prefill_aux_index,
                                self.decode_kv_args_table[
                                    req.p2p_session_id
                                ].dst_aux_ptrs,
                            )
                            polls.append(True if ret == 0 else False)
                            dst_ranks_infos.append(
                                (req.endpoint, req.dst_port, req.room)
                            )

                            # Only sync status when all the dst ranks have received the kvcache
                            if len(polls) == req.required_dst_info_num:
                                status = KVPoll.Success if all(polls) else KVPoll.Failed
                                self.update_status(req.room, status)
                                for endpoint, dst_port, room in dst_ranks_infos:
                                    self.sync_status_to_decode_endpoint(
                                        endpoint, dst_port, room, status
                                    )
                            
                            # 打印单条请求 KVCache 传输统计
                            with self.kv_transfer_stats_lock:
                                stats = self.kv_transfer_stats.pop(kv_chunk.room, None)

                            if self.enable_kvcache_log and stats is not None:
                                kv_bytes = stats.get("kv_bytes", 0)
                                kv_time_ms = stats.get("kv_time_ms", 0.0)
                                kv_tokens = stats.get("kv_tokens", 0)
                                logger.info(
                                    f"[KVCACHE_TRANSFER] room={kv_chunk.room} "
                                    f"bytes={kv_bytes} tokens={kv_tokens} time_ms={kv_time_ms:.2f}"
                                )
                    else:
                        # Dummy request means the decode instance is not used, so its status can be marked as success directly
                        if kv_chunk.is_last and req.room in self.request_status:
                            self.update_status(req.room, KVPoll.Success)

                if (
                    kv_chunk.room not in self.request_status
                    or self.check_status(kv_chunk.room) == KVPoll.Success
                ):
                    if kv_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(kv_chunk.room)

            except Exception as e:
                raise RuntimeError(
                    f"Transfer thread failed because of {e}. Prefill instance with bootstrap_port={self.bootstrap_port} is dead."
                )
    
    def _handle_aux_data(self, msg: List[bytes]):
        """
        处理AUX数据 
        """
        room = int(msg[1].decode("ascii"))
        buffer_index = int(msg[2].decode("ascii"))
        aux_index = int(msg[3].decode("ascii"))
        data_length = struct.unpack(">I", msg[4])[0]
        data = msg[5]

        if len(data) != data_length:
            logger.error(f"AUX_DATA length mismatch for bootstrap_room {room}: expected {data_length}, got {len(data)}")
            return
        AuxDataCodec.deserialize_data_to_buffer(
            self.kv_args, buffer_index, aux_index, data
        )
        logger.debug(
            f"Received AUX_DATA for bootstrap_room {room} with length:{len(data)}"
        )

    def start_decode_thread(self):
        """
        启动解码线程 
        """
        self.rank_port = get_free_port()
        self._bind_server_socket()

        def decode_thread():
            """
            解码线程 
            """
            while True:
                msg = self.server_socket.recv_multipart()
                if msg[0] == P2PKVManager.AUX_DATA_HEADER:
                    self._handle_aux_data(msg)
                    continue

                (bootstrap_room, status) = msg
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                if status == KVPoll.Failed:
                    self.record_failure(
                        bootstrap_room,
                        f"Failed to get kvcache from prefill instance, it might be dead",
                    )
                self.update_status(bootstrap_room, status)

        def heartbeat_checker():
            """
            心跳检测
            """
            while True:
                time.sleep(self.heartbeat_interval)
                with self.connection_lock:
                    addresses = list(self.prefill_dp_size_table.keys())

                for bootstrap_addr in addresses:
                    session = None
                    try:
                        with self.session_pool_lock:
                            session = self.session_pool[bootstrap_addr]
                        response = session.get(
                            f"http://{bootstrap_addr}/health",
                            timeout=(2, 3),
                            headers={"Connection": "keep-alive"},
                        )
                        if response.status_code == 200:
                            self.heartbeat_failures[bootstrap_addr] = 0

                            current_rooms = self.addr_to_rooms_tracker[
                                bootstrap_addr
                            ].copy()

                            for bootstrap_room in current_rooms:
                                if bootstrap_room not in self.request_status:
                                    self.addr_to_rooms_tracker[bootstrap_addr].discard(
                                        bootstrap_room
                                    )
                        else:
                            logger.info(
                                f"Attempting to reconnect to {bootstrap_addr}..."
                            )
                            self.heartbeat_failures[bootstrap_addr] = (
                                self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                            )
                            with self.session_pool_lock:
                                if bootstrap_addr in self.session_pool:
                                    del self.session_pool[bootstrap_addr]
                    except Exception:
                        logger.info(f"Attempting to reconnect to {bootstrap_addr}...")
                        self.heartbeat_failures[bootstrap_addr] = (
                            self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                        )

                    if (
                        self.heartbeat_failures.get(bootstrap_addr, 0)
                        >= self.max_failures
                    ):
                        self._handle_node_failure(bootstrap_addr)
                        with self.session_pool_lock:
                            if bootstrap_addr in self.session_pool:
                                del self.session_pool[bootstrap_addr]

        threading.Thread(target=decode_thread).start()
        threading.Thread(target=heartbeat_checker).start()

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last: bool,
        aux_index: Optional[int] = None,
    ):
        """
        向传输队列中添加传输请求 
        """
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last or (is_last and aux_index is not None)

        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            logger.debug(
                "Request with bootstrap_room=%s already failed", bootstrap_room
            )
            return

        if bootstrap_room not in self.transfer_infos:
            return

        dst_infos = self.transfer_infos[bootstrap_room].keys()
        session_port_sum = sum(int(session.split(":")[1]) for session in dst_infos)
        shard_idx = session_port_sum % len(self.transfer_queues)

        self.transfer_queues[shard_idx].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last=is_last,
                prefill_aux_index=aux_index,
            )
        )

    def check_status(self, bootstrap_room: int):
        """
        检查状态
        """
        return self.request_status.get(bootstrap_room, KVPoll.Bootstrapping)

    def update_status(self, bootstrap_room: int, status: KVPoll):
        """
        更新状态
        """
        if bootstrap_room not in self.request_status:
            self.request_status[bootstrap_room] = status
        else:
            if status == KVPoll.Failed:
                self.request_status[bootstrap_room] = KVPoll.Failed
            else:
                self.request_status[bootstrap_room] = max(
                    self.request_status[bootstrap_room], status
                )

    def record_failure(self, bootstrap_room: int, failure_reason: str):
        """
        记录失败原因
        """
        with self.failure_lock:
            self.failure_records[bootstrap_room] = failure_reason

    def get_session_id(self): 
        """
        获取会话ID
        """
        return self.p2p_engine.get_session_id() 

    def _register_to_bootstrap(self):
        """
        注册到 bootstarp 服务器
        """
        if self.dist_init_addr:
            ip_address = socket.gethostbyname(self.dist_init_addr.split(":")[0])
        else:
            ip_address = get_ip()

        bootstrap_server_url = f"{ip_address}:{self.bootstrap_port}"
        url = f"http://{bootstrap_server_url}/route"
        payload = {
            "role": "Prefill",
            "tp_size": self.tp_size,
            "dp_size": self.dp_size,
            "rank_ip": self.local_ip,
            "rank_port": self.rank_port,
            "engine_rank": self.kv_args.engine_rank,
        }

        try:
            response = requests.put(url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.debug("Prefill successfully registered to bootstrap server.")
            else:
                logger.error(
                    f"Prefill instance failed to connect to bootstrap server: {response.status_code}, {response.text}"
                )
        except Exception as e:
            logger.error(
                f"Prefill instance failed to register to bootstrap server: {e}"
            )

    def _handle_node_failure(self, failed_bootstrap_addr):
        """
        处理节点失败的情况 
        """
        with self.connection_lock:
            keys_to_remove = [
                k for k in self.connection_pool if k.startswith(failed_bootstrap_addr)
            ]
            for k in keys_to_remove:
                del self.connection_pool[k]
            if failed_bootstrap_addr in self.prefill_tp_size_table:
                del self.prefill_tp_size_table[failed_bootstrap_addr]
            if failed_bootstrap_addr in self.prefill_dp_size_table:
                del self.prefill_dp_size_table[failed_bootstrap_addr]

            possible_affected_rooms = self.addr_to_rooms_tracker.get(
                failed_bootstrap_addr, []
            )
            if failed_bootstrap_addr in self.addr_to_rooms_tracker:
                del self.addr_to_rooms_tracker[failed_bootstrap_addr]

        affected_rooms = []
        for room in possible_affected_rooms:
            if (
                room in self.request_status
                and self.check_status(room) != KVPoll.Success
            ):
                self.record_failure(
                    room,
                    f"Losing connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr})",
                )
                self.update_status(room, KVPoll.Failed)
                affected_rooms.append(room)
        logger.error(
            f"Losing connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr}), affected {len(affected_rooms)} requests"
        )


class P2PKVSender(BaseKVSender):
    """
    P2PKVSender 类
    """
    def __init__(
        self,
        mgr: P2PKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        """
        初始化函数
        """
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.kv_mgr.update_status(bootstrap_room, KVPoll.Bootstrapping)
        self.aux_index = None
        self.bootstrap_server_url = bootstrap_addr
        self.conclude_state = None
        self.init_time = None
        self.curr_idx = 0

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        """
        初始化函数
        """
        self.num_kv_indices = num_kv_indices
        self.aux_index = aux_index
        self.init_time = time.time()

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
    ):
        """
        向kvmgr发送键值对索引
        """
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last = self.curr_idx == self.num_kv_indices

        if not is_last:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room, kv_indices, index_slice, False
            )
        else:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                True,
                aux_index=self.aux_index,
            )

    def poll(self) -> KVPoll:
        """
        轮询函数
        """
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
            elif status == KVPoll.Bootstrapping:
                if self.init_time is not None:
                    now = time.time()
                    elapsed = now - self.init_time
                    if elapsed >= self.kv_mgr.bootstrap_time_out:
                        self.kv_mgr.record_failure(
                            self.bootstrap_room,
                            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.Bootstrapping",
                        )
                        self.conclude_state = KVPoll.Failed
                        return KVPoll.Failed

            return status
        else:
            return self.conclude_state

    def clear(self) -> None:
        """
        清除函数
        """
        if self.bootstrap_room in self.kv_mgr.request_status:
            self.kv_mgr.request_status.pop(self.bootstrap_room)
    
    def failure_exception(self):
        """
        异常处理
        """
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, 
                "P2P transfer failed due to an unknown reason"
            )
        raise P2PTransferError(self.bootstrap_room, failure_reason) 


class P2PKVReceiver(BaseKVReceiver):
    """
    P2PKVReceiver 类
    """
    _ctx = zmq.Context()
    _socket_cache = {}
    _socket_locks = {}
    _global_lock = threading.Lock()

    def __init__(
        self,
        mgr: P2PKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        data_parallel_rank: Optional[int] = None,
    ):
        """
        初始化函数
        """
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.kv_mgr = mgr
        self.session_id = self.kv_mgr.get_session_id()
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)
        self.conclude_state = None
        self.data_parallel_rank = data_parallel_rank

        if self.bootstrap_addr not in self.kv_mgr.prefill_dp_size_table:
            self.prefill_tp_size, self.prefill_dp_size = (
                self._get_prefill_parallel_info_from_server()
            )
            if self.prefill_tp_size is None or self.prefill_dp_size is None:
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
                )
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return
            else:
                logger.debug(
                    f"Fetch prefill parallel info from [{self.bootstrap_addr}]: DP size:{self.prefill_dp_size}, TP size:{self.prefill_tp_size}"
                )
                self.kv_mgr.prefill_tp_size_table[self.bootstrap_addr] = (
                    self.prefill_tp_size
                )
                self.kv_mgr.prefill_dp_size_table[self.bootstrap_addr] = (
                    self.prefill_dp_size
                )
        else:
            self.prefill_tp_size = self.kv_mgr.prefill_tp_size_table[
                self.bootstrap_addr
            ]
            self.prefill_dp_size = self.kv_mgr.prefill_dp_size_table[
                self.bootstrap_addr
            ]

        local_tp_size_per_dp_rank = self.kv_mgr.tp_size // self.kv_mgr.dp_size
        prefill_tp_size_per_dp_rank = self.prefill_tp_size // self.prefill_dp_size
        if local_tp_size_per_dp_rank == prefill_tp_size_per_dp_rank:
            self.target_tp_rank = (
                self.kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank
            )
            self.required_dst_info_num = 1
            self.target_tp_ranks = [self.target_tp_rank]
        elif local_tp_size_per_dp_rank > prefill_tp_size_per_dp_rank:
            assert (
                self.kv_mgr.is_mla_backend
            ), "PD with different TP sizes per DP rank is not yet supported for non-MLA models"
            self.target_tp_rank = (
                self.kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank
            ) // (local_tp_size_per_dp_rank // prefill_tp_size_per_dp_rank)
            self.required_dst_info_num = (
                local_tp_size_per_dp_rank // prefill_tp_size_per_dp_rank
            )
            self.target_tp_ranks = [self.target_tp_rank]
        else:
            assert (
                self.kv_mgr.is_mla_backend
            ), "PD with different TP sizes per DP rank is not yet supported for non-MLA models"

            self.target_tp_ranks = [
                rank
                for rank in range(
                    (self.kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank)
                    * (prefill_tp_size_per_dp_rank // local_tp_size_per_dp_rank),
                    (self.kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank + 1)
                    * (prefill_tp_size_per_dp_rank // local_tp_size_per_dp_rank),
                )
            ]

            self.target_tp_rank = self.target_tp_ranks[0]
            self.required_dst_info_num = 1

        if self.data_parallel_rank is not None:
            logger.debug(f"Targeting DP rank: {self.data_parallel_rank}")
            self.target_dp_group = self.data_parallel_rank
        else:
            self.target_dp_group = bootstrap_room % self.prefill_dp_size

        bootstrap_key = (
            f"{self.bootstrap_addr}_{self.target_dp_group}_{self.target_tp_rank}"
        )

        if bootstrap_key not in self.kv_mgr.connection_pool:
            bootstrap_infos = []
            for target_tp_rank in self.target_tp_ranks:
                bootstrap_info = self._get_bootstrap_info_from_server(
                    target_tp_rank,
                    self.target_dp_group,
                )
                if bootstrap_info is not None:
                    bootstrap_info["is_dummy"] = not bool(
                        target_tp_rank == self.target_tp_rank
                        or self.target_tp_rank is None
                    )
                    logger.debug(
                        f"Fetched bootstrap info: {bootstrap_info} for DP {self.target_dp_group} TP {target_tp_rank}"
                    )
                    bootstrap_infos.append(bootstrap_info)
                else:
                    self.kv_mgr.record_failure(
                        self.bootstrap_room,
                        f"Could not fetch bootstrap info for engine rank: {self.kv_mgr.kv_args.engine_rank} and target_dp_group: {self.target_dp_group}",
                    )
                    self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                    return

            self.bootstrap_infos = bootstrap_infos
            self.kv_mgr.connection_pool[bootstrap_key] = self.bootstrap_infos

            self._register_kv_args()
        else:
            self.bootstrap_infos = self.kv_mgr.connection_pool[bootstrap_key]

        assert len(self.bootstrap_infos) > 0
        self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].add(self.bootstrap_room)
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def _get_bootstrap_info_from_server(self, engine_rank, target_dp_group):
        """
        从服务器获取引导信息
        """
        try:
            url = f"http://{self.bootstrap_addr}/route?engine_rank={engine_rank}&target_dp_group={target_dp_group}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                bootstrap_info = response.json()
                return bootstrap_info
            else:
                logger.error(
                    f"Failed to get prefill server info: {response.status_code}, {response.text}"
                )
                return None
        except Exception as e:
            logger.error(f"Error fetching prefill info from bootstrap: {e}")
            return None

    def _get_prefill_parallel_info_from_server(self) -> Tuple[int, int]:
        """
        从服务器获取预填充并行信息
        """
        try:
            url = f"http://{self.bootstrap_addr}/route?engine_rank={-1}&target_dp_group={-1}"
            response = requests.get(url)
            if response.status_code == 200:
                prefill_parallel_info = response.json()
                return int(prefill_parallel_info["prefill_tp_size"]), int(
                    prefill_parallel_info["prefill_dp_size"]
                )
            else:
                logger.error(
                    f"Failed to get prefill parallel info: {response.status_code}, {response.text}"
                )
                return None, None
        except Exception as e:
            logger.error(f"Error fetching prefill parallel info from bootstrap: {e}")
            return None, None

    def _register_kv_args(self):
        """
        注册 KV 参数
        """
        tp_rank = self.kv_mgr.kv_args.engine_rank
        physical_gpu_id = (
            self.kv_mgr.server_args.base_gpu_id +
            tp_rank * self.kv_mgr.server_args.gpu_id_step
        )
        for bootstrap_info in self.bootstrap_infos:
            self.prefill_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
            logger.info(f"Connecting to prefill server at {self.prefill_server_url}")
            
            packed_kv_handles = b"".join(self.kv_mgr.kv_handles)
            packed_aux_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
            )

            gpu_id_str = str(physical_gpu_id) 
            
            sock, lock = self._connect("tcp://" + self.prefill_server_url)
            logger.info(f"Connected to {self.prefill_server_url} with socket: {sock}")
            with lock:
                sock.send_multipart(
                    [
                        "None".encode("ascii"), 
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        packed_kv_handles,
                        packed_aux_data_ptrs,
                        gpu_id_str.encode("ascii")  # 发送GPU ID
                    ]
                )
    
    @classmethod
    def _connect(cls, endpoint: str):
        """
        连接到指定的ZMQ端点
        """
        with cls._global_lock:
            if endpoint not in cls._socket_cache:
                sock = cls._ctx.socket(zmq.PUSH)
                sock.connect(endpoint)
                cls._socket_cache[endpoint] = sock
                cls._socket_locks[endpoint] = threading.Lock()
            return cls._socket_cache[endpoint], cls._socket_locks[endpoint]

    def init(self, kv_indices: npt.NDArray[np.int32], aux_index: Optional[int] = None):
        """
        初始化函数，用于连接到预填充服务器并发送初始化请求
        """
        for bootstrap_info in self.bootstrap_infos:
            self.prefill_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
            is_dummy = bootstrap_info["is_dummy"]

            sock, lock = self._connect("tcp://" + self.prefill_server_url)
            with lock:
                sock.send_multipart(
                    [
                        str(self.bootstrap_room).encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        kv_indices.tobytes() if not is_dummy else b"",
                        str(aux_index).encode("ascii") if not is_dummy else b"",
                        str(self.required_dst_info_num).encode("ascii"),
                    ]
                )

    def poll(self) -> KVPoll:
        """
        轮询检查状态
        """
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status

            return status
        else:
            return self.conclude_state

    def clear(self) -> None:
        """
        清除
        """
        if self.bootstrap_room in self.kv_mgr.request_status:
            self.kv_mgr.request_status.pop(self.bootstrap_room)

    def failure_exception(self):
        """
        失败异常
        """
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "P2P transfer failed due to an unknown reason"
            )
        raise P2PTransferError(self.bootstrap_room, failure_reason)


class P2PKVBootstrapServer(BaseKVBootstrapServer):
    """
    P2PKVBootstrapServer 类
    """
    def __init__(self, port: int):
        """
        初始化
        """
        self.port = port
        self.app = web.Application()
        self.store = dict()
        self.lock = asyncio.Lock()
        self._setup_routes()
        self.tp_size = None
        self.dp_size = None
        self.tp_size_per_dp_rank = None
        self.prefill_port_table: Dict[int, Dict[int, Dict[str, Union[str, int]]]] = {}

        # Start bootstrap server
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.run()

    def run(self):
        """
        运行线程 
        """
        self.thread.start()

    def _setup_routes(self):
        """
        设置路由
        """
        self.app.router.add_route("*", "/route", self._handle_route)
        self.app.router.add_get("/health", self._handle_health_check)

    async def _handle_health_check(self, request):
        """
        处理健康检查请求
        """
        return web.Response(text="OK", status=200)

    async def _handle_route(self, request: web.Request):
        """
        处理HTTP请求
        """
        method = request.method
        if method == "PUT":
            return await self._handle_route_put(request)
        elif method == "GET":
            return await self._handle_route_get(request)
        else:
            return web.Response(
                text="Method not allowed", status=405, content_type="application/json"
            )

    async def _handle_route_put(self, request: web.Request):
        """
        处理 PUT 请求的异步函数
        """
        data = await request.json()
        role = data["role"]
        tp_size = data["tp_size"]
        dp_size = data["dp_size"]
        rank_ip = data["rank_ip"]
        rank_port = int(data["rank_port"])
        engine_rank = int(data["engine_rank"])

        if self.tp_size is None:
            self.tp_size = tp_size

        if self.dp_size is None:
            self.dp_size = dp_size

        tp_size_per_dp_rank = tp_size // dp_size
        if self.tp_size_per_dp_rank is None:
            self.tp_size_per_dp_rank = tp_size_per_dp_rank

        if role == "Prefill":
            dp_group = engine_rank // tp_size_per_dp_rank
            tp_rank_in_dp_group = engine_rank % tp_size_per_dp_rank

            async with self.lock:
                if dp_group not in self.prefill_port_table:
                    self.prefill_port_table[dp_group] = {}

            self.prefill_port_table[dp_group][tp_rank_in_dp_group] = {
                "rank_ip": rank_ip,
                "rank_port": rank_port,
            }
            logger.debug(
                f"Register prefill bootstrap: {engine_rank} with rank_ip: {rank_ip} and rank_port: {rank_port}"
            )

        return web.Response(text="OK", status=200)

    async def _handle_route_get(self, request: web.Request):
        """
        处理 GET 请求的异步函数
        """
        engine_rank = request.query.get("engine_rank")
        target_dp_group = request.query.get("target_dp_group")
        if not engine_rank or not target_dp_group:
            return web.Response(text="Missing inputs for bootstrap server.", status=400)

        if int(engine_rank) == -1 and int(target_dp_group) == -1:
            prefill_parallel_info = {
                "prefill_tp_size": self.tp_size,
                "prefill_dp_size": self.dp_size,
            }
            return web.json_response(prefill_parallel_info, status=200)

        async with self.lock:
            bootstrap_info = self.prefill_port_table[int(target_dp_group)][
                int(engine_rank)
            ]

        if bootstrap_info is not None:
            return web.json_response(bootstrap_info, status=200)
        else:
            return web.Response(text="Bootstrap info not Found", status=404)

    def _run_server(self):
        """
        启动服务
        """
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            access_log = None
            if logging.getLogger(__name__).getEffectiveLevel() <= logging.DEBUG:
                access_log = self.app.logger

            self._runner = web.AppRunner(self.app, access_log=access_log)
            self._loop.run_until_complete(self._runner.setup())

            site = web.TCPSite(self._runner, port=self.port)
            self._loop.run_until_complete(site.start())
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
        finally:
            self._loop.run_until_complete(self._runner.cleanup())
            self._loop.close()

    def close(self):
        """
        关闭服务
        """
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            logger.info("Stopping server loop...")

        if self.thread.is_alive():
            self.thread.join(timeout=2)
            logger.info("Server thread stopped")

    def poll(self) -> KVPoll: ...