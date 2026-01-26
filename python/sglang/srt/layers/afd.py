# AF disaggregation

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========

import itertools
from collections import deque
from enum import Enum, auto
from typing import Any, Dict, Optional, List, Tuple
from abc import ABC, abstractmethod
from functools import cache

import sys
import os
import time
import logging

logger = logging.getLogger(__name__)

import torch
from torch import nn
import torch.distributed as dist
import zmq

from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.layers.communicator import LayerCommunicator, ScatterMode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.afd_type import AFDPerspective

from sglang.srt.layers.communicator import (
    CommunicateContext,
    CommunicateSummableTensorPairFn,
    ScatterMode,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import BumpAllocator


class AFDForwardStage(Enum):
    AFD_FORWARD_STAGE_A = auto()
    AFD_FORWARD_STAGE_F = auto()

class AFDStageScheduleGenerator:
    Schedule = List[Tuple[AFDForwardStage, int, int]]
    @staticmethod
    def ffn_stage(num_layers: int, m_stage: int) -> Schedule:
        schedule = []
        for l, m in itertools.product(range(num_layers), range(m_stage)):
            schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_A, l, m))
            schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_F, l, m))
        return schedule
    @staticmethod
    def attn_stage(num_layers: int, m_stage: int) -> Schedule:
        schedule = []
        if num_layers == 1:
            return (
                [(AFDForwardStage.AFD_FORWARD_STAGE_A, 0, m) for m in range(m_stage)]
                +
                [(AFDForwardStage.AFD_FORWARD_STAGE_F, 0, m) for m in range(m_stage)]
            )
        for l, m in itertools.product(range(num_layers + 1), range(m_stage)):
            if l > 0:
                schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_F, l - 1, m))
            if l < num_layers:
                schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_A, l, m))
        return schedule

class FifoTensorCommunicator(ABC):
    @abstractmethod
    def recv_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def send_tensor(self, x: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def init(self, dp_rank, dp_size, tp_rank, tp_size):
        raise NotImplementedError

class ZMQSimpleTensorCommunicator(FifoTensorCommunicator):
    def __init__(self):
        super().__init__()
        self.zmq_context = zmq.Context()

        self.start_lport = self.get_ffn_port() if afd_is_attn() else self.get_attn_port()
        self.start_dport = self.get_attn_port() if afd_is_attn() else self.get_ffn_port()

    def init(self, dp_rank, dp_size, tp_rank, tp_size):
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.tp_rank = tp_rank
        self.tp_size = tp_size

    def get_ffn_port(self) -> int:
        return 40000

    def get_attn_port(self) -> int:
        return 50000

    def get_lport(self) -> int:
        return self.start_lport + 1 + dist.get_rank()

    def get_dport(self) -> int:
        return self.start_dport + 1 + dist.get_rank()

    def get_current_cuda_device(self) -> torch.device:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot get current CUDA device.")
        device_index = torch.cuda.current_device()
        return torch.device(f"cuda:{device_index}")

    @cache
    def get_push_socket(self) -> zmq.Socket:
        socket = self.zmq_context.socket(zmq.PUSH)
        socket.connect(f"tcp://localhost:{self.get_dport()}")
        return socket

    @cache
    def get_pull_socket(self) -> zmq.Socket:
        socket = self.zmq_context.socket(zmq.PULL)
        socket.bind(f"tcp://*:{self.get_lport()}")
        return socket

    def recv_tensor(self) -> torch.Tensor:
        socket = self.get_pull_socket()
        x = socket.recv_pyobj()
        assert isinstance(x, torch.Tensor)
        return x.to(self.get_current_cuda_device())

    def send_tensor(self, x: torch.Tensor):
        socket = self.get_push_socket()
        socket.send_pyobj(x)

class StepMeshTensorCache(object):
    def __init__(self, ten=None, key=0):
        self.push_tensor = ten
        self.pull_tensor = ten

        self.push_key = key
        self.pull_key = key + 1

        self.h = None

def stepmesh_scheduler():
    import setproctitle
    setproctitle.setproctitle("stepmesh_scheduler")

    os.environ['DMLC_ROLE'] = 'scheduler'

    logger.info("StepMesh scheduler: DMLC_PS_ROOT_URI=%s" % os.environ["DMLC_NODE_HOST"])
    import fserver_lib as f

    f.init()
    logger.info("StepMesh scheduler init done.")

    while True:
        time.sleep(10000)


class StepMeshTensorCommunicator(FifoTensorCommunicator):
    def __init__(self):
        super().__init__()

        self.key = 0
        self.comm_ids = []
        self.waits = []
        self.free_tensors = {}
        self.register_buf = {}
        self.buf_size_history = []

    def init(self, dp_rank, dp_size, tp_rank, tp_size):
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        import fserver_lib as f

        self.start_stepmesh_scheduler()
        if afd_is_attn():
            time.sleep(10) # wait scheduler
        logger.info("%s init..." % os.environ['DMLC_ROLE'])
        f.init()
        logger.info("%s init done." % os.environ['DMLC_ROLE'])
        self.f = f

    def env_def(self, env, v):
        if os.environ.get(env) == None:
            os.environ[env] = v

    def get_node_ip(self):
        if os.environ.get("DMLC_NODE_HOST") != None:
            return

        import psutil

        interface_name = os.environ.get("DMLC_INTERFACE")

        interfaces = psutil.net_if_addrs()

        if interface_name not in interfaces:
            logger.info("Invalid DMLC_INTERFACE %s" % interface_name)
            return

        for addr in interfaces[interface_name]:
            if addr.family == 2:  # socket.AF_INET
                os.environ["DMLC_NODE_HOST"] = addr.address
                break

    def start_stepmesh_scheduler(self):
        self.get_node_ip()

        self.env_def('DMLC_NODE_RANK',    str(self.dp_rank if self.dp_rank else 0))
        self.env_def('DMLC_NUM_SERVER',   '1')
        self.env_def('DMLC_NUM_WORKER',   '1')
        self.env_def('DMLC_GROUP_SIZE',   '1')
        self.env_def('DMLC_PS_ROOT_PORT', '8123')
        self.env_def('DMLC_ENABLE_RDMA',  'ibverbs')
        self.env_def('STEPMESH_GPU',      str(torch.cuda.current_device()))
        self.env_def('DMLC_INSTANCE_ID',  str(self.tp_rank if self.tp_rank else 0))

        if afd_is_attn():
            os.environ['DMLC_ROLE'] = 'worker'
        else:
            os.environ['DMLC_ROLE'] = 'server'

        if os.environ['DMLC_ROLE'] != 'worker':
            return

        if os.environ.get('DMLC_NODE_RANK') != '0':
            return

        if os.environ['STEPMESH_GPU'] != '0':
            return

        if os.environ.get('STEPMESH_SCHEDULER_STARTED') == '1':
            return

        os.environ['STEPMESH_SCHEDULER_STARTED'] = '1'
        os.environ["DMLC_NODE_HOST"] = os.environ["DMLC_PS_ROOT_URI"]

        import multiprocessing

        p = multiprocessing.Process(target=stepmesh_scheduler)
        p.daemon = True
        p.start()

    def attn_send(self, x):
        if x.size(0) == 0:
            x = torch.empty([1], dtype=x.dtype, device=x.device)

        free = self.free_tensors.get(x.shape, [])
        self.free_tensors[x.shape] = free

        if free:
            t = free.pop()
            t.push_tensor.copy_(x)
        else:
            self.key += 2
            t = StepMeshTensorCache(x, self.key)

            if (len(self.buf_size_history) > 3):
                oldest_size = self.buf_size_history.pop(0)
                _tensors = self.free_tensors.get(oldest_size)
                if (len(_tensors) > 1):
                    del _tensors[-1]
                else:
                    del self.free_tensors[oldest_size]
            self.buf_size_history.append(x.shape)

        t.h = self.f.push_pull(
                [t.push_tensor],
                [t.push_key],
                [t.pull_tensor],
                [t.pull_key])

        self.waits.append(t)

    def attn_recv(self):
        t = self.waits.pop(0)
        self.f.wait(t.h, timeout_ms=100000)

        free = self.free_tensors.get(t.push_tensor.shape, [])
        self.free_tensors[t.push_tensor.shape] = free
        free.append(t)

        if t.pull_tensor.ndim == 1:
            return torch.empty(0, dtype=t.pull_tensor.dtype, device=t.pull_tensor.device)

        return t.pull_tensor

    def ffn_send(self, x):
        free = self.free_tensors.get(x.shape, [])
        self.free_tensors[x.shape] = free

        if free:
            t = free.pop()
            t.copy_(x)
        else:
            t = torch.empty_like(x)
            t.copy_(x)

            if (len(self.buf_size_history) > 3):
                oldest_size = self.buf_size_history.pop(0)
                _tensors = self.free_tensors.get(oldest_size)
                if (len(_tensors) > 1):
                    del _tensors[-1]
                else:
                    del self.free_tensors[oldest_size]
            self.buf_size_history.append(x.shape)

        c = self.comm_ids.pop(0)
        for _id, comm_id in enumerate(c[0]):
            _need_event = (_id == 0)
            idx_range = c[1][_id]
            _res_tensor = t[idx_range[0]:idx_range[1]]
            if _res_tensor.size(0) == 0:
                _res_tensor = torch.empty([1], dtype=_res_tensor.dtype, device=x.device)
            self.f.respond([_res_tensor], comm_id, _need_event)

        free.append(t)

    def ffn_recv(self):
        batches = self.f.get_batch()

        ## batches [
        #     [comm_id, push_tensor_list, key_list],
        #     [comm_id, push_tensor_list, key_list],
        # ]
        # assert len(batches) == 1, "just handle for one worker"

        _comm_ids = []
        _idx_ranges = []
        _tensor_list = []
        _begin_idx = 0

        for batch in batches:
            if batch[1][0].ndim == 1:
                _length = 0
            else:
                _length = batch[1][0].size(0)
                _tensor_list.append(batch[1][0])
            _end_idx = _begin_idx + _length
            _idx_ranges.append([_begin_idx, _end_idx])
            _comm_ids.append(batch[0])
            _begin_idx = _end_idx
        self.comm_ids.append([_comm_ids, _idx_ranges])
        if len(_tensor_list) == 0:
            return torch.empty(0, dtype=batches[0][1][0].dtype, device=batches[0][1][0].device)
        elif len(_tensor_list) == 1:
            return _tensor_list[0]
        else:
            return torch.cat(_tensor_list, dim=0)

    def recv_tensor(self) -> torch.Tensor:
        if afd_is_attn():
            return self.attn_recv()
        else:
            return self.ffn_recv()

    def send_tensor(self, x: torch.Tensor):
        if afd_is_attn():
            self.attn_send(x)
        else:
            self.ffn_send(x)

@cache
def get_tensor_communicator() -> FifoTensorCommunicator:
    if os.environ.get("DMLC_INTERFACE"):
        return StepMeshTensorCommunicator()
    else:
        return ZMQSimpleTensorCommunicator()

def get_afd_mirco_batch() -> int:
    afd_mirco_batch = global_server_args_dict.get("afd_mirco_batch")
    if afd_mirco_batch is None:
        raise ValueError("afd_mirco_batch is not set")
    return afd_mirco_batch

def get_afd_perspective() -> Optional[AFDPerspective]:
    afd_perspective = global_server_args_dict.get("afd_perspective")
    return afd_perspective

def afd_is_ffn():
    return get_afd_perspective() == AFDPerspective.FFN

def afd_is_attn():
    return get_afd_perspective() == AFDPerspective.ATTN

def model_forward_afd_split_inputs(
    layers,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    input_data_scatter_mode: ScatterMode,
):
    def _model_forward_afd_split_inputs_raw(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        ) -> List[Dict]:
        return [
            dict(
                **_model_forward_filter_inputs(
                    hidden_states=hidden_states,
                    residual=residual,
                    positions=positions,
                    output_forward_batch=output_forward_batch,
                    afd_subbatch_index=afd_subbatch_index,
                ),
                **({}),
            )
            for afd_subbatch_index, output_forward_batch in enumerate(
                forward_batch.afd_children
            )
        ]

    def _model_forward_filter_inputs(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        positions: torch.Tensor,
        output_forward_batch: ForwardBatch,
        afd_subbatch_index: int,
    ) -> Dict:
        token_slice = slice(*output_forward_batch.afd_parent_token_range)
        return dict(
            hidden_states=hidden_states[token_slice],
            residual=None if residual is None else residual[token_slice],
            positions=positions[token_slice],
            forward_batch=output_forward_batch,
            afd_subbatch_index=afd_subbatch_index,
        )

    layer_input_scatter_mode = layers[0].layer_scatter_modes.layer_input_mode
    afd_splitter_scatter_mode = ScatterMode.TP_ATTN_FULL
    context = CommunicateContext.init_new()

    hidden_states, residual = CommunicateSummableTensorPairFn.execute(
        hidden_states_input_mode=input_data_scatter_mode,
        residual_input_mode=input_data_scatter_mode,
        output_mode=afd_splitter_scatter_mode,
        hidden_states=hidden_states,
        residual=residual,
        forward_batch=forward_batch,
        context=context,
    )

    inputs_arr = _model_forward_afd_split_inputs_raw(
        hidden_states=hidden_states,
        residual=residual,
        positions=positions,
        forward_batch=forward_batch,
    )

    def _post_transform(hidden_states, residual, forward_batch, **kwargs):
        hidden_states, residual = CommunicateSummableTensorPairFn.execute(
            hidden_states_input_mode=afd_splitter_scatter_mode,
            residual_input_mode=afd_splitter_scatter_mode,
            output_mode=layer_input_scatter_mode,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            context=context,
        )
        return dict(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            **kwargs,
        )

    return [_post_transform(**inputs) for inputs in inputs_arr]

def model_forward_afd(
    layers,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    input_data_scatter_mode: ScatterMode,
):
    num_layers = len(layers)
    m_stage = get_afd_mirco_batch()

    input_arrs = model_forward_afd_split_inputs(
        layers=layers,
        hidden_states=hidden_states,
        residual=residual,
        positions=positions,
        forward_batch=forward_batch,
        input_data_scatter_mode=input_data_scatter_mode
    )

    stage_outputs: Dict[AFDForwardStage, deque[dict[Any, Any]]] = {
        AFDForwardStage.AFD_FORWARD_STAGE_A: deque(),
        AFDForwardStage.AFD_FORWARD_STAGE_F: deque(),
    }

    stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].extend(input_arrs)

    def forward_A(layer_id: int, mirco_batch_idx: int):
        inputs_args = stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].popleft()
        hidden_states, residual = layers[layer_id].forward_afd_A(
            input_arrs[mirco_batch_idx]["positions"],
            inputs_args["hidden_states"],
            input_arrs[mirco_batch_idx]["forward_batch"],
            inputs_args["residual"],
        )
        stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_A].append(
            dict (
            hidden_states = hidden_states,
            residual = residual,
        ))

    def forward_F(layer_id: int, mirco_batch_idx: int):
        inputs_args = stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_A].popleft()
        hidden_states, residual = layers[layer_id].forward_afd_F(
            inputs_args["hidden_states"],
            input_arrs[mirco_batch_idx]["forward_batch"],
            inputs_args["residual"],
        )
        stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].append(
            dict (
            hidden_states = hidden_states,
            residual = residual,
        ))

    stage_executors = {
        AFDForwardStage.AFD_FORWARD_STAGE_A : forward_A,
        AFDForwardStage.AFD_FORWARD_STAGE_F : forward_F,
    }

    pipeline_stages = None
    if afd_is_attn():
        pipeline_stages = AFDStageScheduleGenerator.attn_stage(num_layers, m_stage)
    elif afd_is_ffn():
        pipeline_stages = AFDStageScheduleGenerator.ffn_stage(num_layers, m_stage)
    else:
        raise NotImplementedError()

    for type, *args in pipeline_stages:
        stage_executors.get(type)(*args)

    try:
        results = [stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].popleft() for _ in range(m_stage)]
    except IndexError:
        raise ValueError("model_forward_afd: impossible path, a potential implementation bug?")

    all_hidden_states, all_residual = zip(
        *((res["hidden_states"], res["residual"]) for res in results)
    )

    all_residual = [res for res in all_residual if res is not None]

    return (
        torch.cat(all_hidden_states, dim=0),
        torch.cat(all_residual, dim=0) if afd_is_attn() and (len(all_residual) > 0) else None,
    )


def deepseek_v2_forward_afd(
    layers,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    input_data_scatter_mode: ScatterMode,
    zero_allocator: Optional[BumpAllocator] = None,
):
    num_layers = len(layers)
    m_stage = get_afd_mirco_batch()

    input_arrs = model_forward_afd_split_inputs(
        layers=layers,
        hidden_states=hidden_states,
        residual=residual,
        positions=positions,
        forward_batch=forward_batch,
        input_data_scatter_mode=input_data_scatter_mode,
    )

    stage_outputs: Dict[AFDForwardStage, deque[dict[Any, Any]]] = {
        AFDForwardStage.AFD_FORWARD_STAGE_A: deque(),
        AFDForwardStage.AFD_FORWARD_STAGE_F: deque(),
    }

    stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].extend(input_arrs)

    def forward_A(layer_id: int, mirco_batch_idx: int):
        inputs_args = stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].popleft()
        hidden_states, residual = layers[layer_id].forward_afd_A(
            input_arrs[mirco_batch_idx]["positions"],
            inputs_args["hidden_states"],
            input_arrs[mirco_batch_idx]["forward_batch"],
            inputs_args["residual"],
            zero_allocator,
        )
        stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_A].append(
            dict(
                hidden_states=hidden_states,
                residual=residual,
            )
        )

    def forward_F(layer_id: int, mirco_batch_idx: int):
        inputs_args = stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_A].popleft()
        hidden_states, residual = layers[layer_id].forward_afd_F(
            inputs_args["hidden_states"],
            input_arrs[mirco_batch_idx]["forward_batch"],
            inputs_args["residual"],
            zero_allocator,
        )
        stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].append(
            dict(
                hidden_states=hidden_states,
                residual=residual,
            )
        )

    stage_executors = {
        AFDForwardStage.AFD_FORWARD_STAGE_A: forward_A,
        AFDForwardStage.AFD_FORWARD_STAGE_F: forward_F,
    }

    pipeline_stages = (
        AFDStageScheduleGenerator.attn_stage(num_layers, m_stage)
        if afd_is_attn()
        else AFDStageScheduleGenerator.ffn_stage(num_layers, m_stage)
    )

    for stage in pipeline_stages:
        type, *args = stage
        stage_executors.get(type)(*args)

    try:
        results = [
            stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].popleft()
            for _ in range(m_stage)
        ]
    except IndexError:
        raise ValueError(
            "model_forward_afd: impossible path, a potential implementation bug?"
        )

    all_hidden_states, all_residual = zip(
        *((res["hidden_states"], res["residual"]) for res in results)
    )

    return (
        torch.cat(all_hidden_states, dim=0),
        torch.cat(all_residual, dim=0) if afd_is_attn() else None,
    )

class AFDCommunicator:
    def __init__(self):
        self.empty_tensor = torch.empty(0, dtype=global_server_args_dict.get("dtype"), device=global_server_args_dict.get("device"))

    @abstractmethod
    def attn_transmit(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def ffn_transmit(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AFDCommunicatorATTN(AFDCommunicator):
    def __init__(self):
        super().__init__()

    def attn_transmit(self, hidden_states: torch.Tensor) -> torch.Tensor:
        get_tensor_communicator().send_tensor(hidden_states)
        return self.empty_tensor

    def ffn_transmit(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return get_tensor_communicator().recv_tensor()


class AFDCommunicatorFFN(AFDCommunicator):
    def __init__(self):
        super().__init__()

    def attn_transmit(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return get_tensor_communicator().recv_tensor()

    def ffn_transmit(self, hidden_states: torch.Tensor) -> torch.Tensor:
        get_tensor_communicator().send_tensor(hidden_states)
        return self.empty_tensor
