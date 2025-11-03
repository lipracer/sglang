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

class ZMQSimpleTensorCommunicator(FifoTensorCommunicator):
    def __init__(self, afd_perspective: AFDPerspective):
        super().__init__()
        self.zmq_context = zmq.Context()

        self.start_lport = (
            self.get_ffn_port() if afd_perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN
            else self.get_attn_port()
        )

        self.start_dport = (
            self.get_attn_port() if afd_perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN
            else self.get_ffn_port()
        )

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
    def __init__(self):
        self.push_tensor = None
        self.pull_tensor = None

        self.push_key = 0
        self.pull_key = 0

        self.h = None

def stepmesh_scheduler():
    os.environ['DMLC_ROLE'] = 'scheduler'

    print("StepMesh scheduler: DMLC_PS_ROOT_URI=%s" % os.environ["DMLC_NODE_HOST"])
    import fserver_lib as f

    f.init()
    print("StepMesh scheduler init done.")

    while True:
        time.sleep(10000)

    f.stop()

class StepMeshTensorCommunicator(FifoTensorCommunicator):
    def __init__(self, afd_perspective: AFDPerspective):
        self.perspective = afd_perspective

        super().__init__()

        import fserver_lib as f

        self.start_stepmesh_scheduler()

        time.sleep(10) # wait scheduler

        logger.info("%s init..." % os.environ['DMLC_ROLE'])
        f.init()
        logger.info("%s init done." % os.environ['DMLC_ROLE'])

        self.worker_num = int(os.environ['DMLC_NUM_WORKER'])

        self.key = 0
        self.gpu = torch.cuda.current_device()
        self.tensor_shape = None

        self.f = f

        self.comm_ids = []
        self.waits = []
        self.free_tensors = {}
        self.register_buf = {}

    def env_def(self, env, v):
        if os.environ.get(env) == None:
            os.environ[env] = v

    def get_node_ip(self):
        if os.environ.get("DMLC_NODE_HOST") != None:
            return

        import psutil

        interface_name = os.environ.get("MLC_INTERFACE")

        interfaces = psutil.net_if_addrs()

        if interface_name not in interfaces:
            print("Invalid MLC_INTERFACE %s" % interface_name)
            return

        for addr in interfaces[interface_name]:
            if addr.family == 2:  # socket.AF_INET
                os.environ["DMLC_NODE_HOST"] = addr.address
                break

    def start_stepmesh_scheduler(self):
        self.get_node_ip()

        gpu = '%s' % torch.cuda.current_device()

        self.env_def('DMLC_NODE_RANK',    '0')
        self.env_def('DMLC_NUM_SERVER',   '1')
        self.env_def('DMLC_NUM_WORKER',   '1')
        self.env_def('DMLC_GROUP_SIZE',   '1')
        self.env_def('DMLC_PS_ROOT_PORT', '8123')
        self.env_def('DMLC_ENABLE_RDMA',  'ibverbs')
        self.env_def('STEPMESH_GPU',      gpu)

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
        os.environ["DMLC_PS_ROOT_URI"] = os.environ["DMLC_NODE_HOST"]

        import multiprocessing

        p = multiprocessing.Process(target=stepmesh_scheduler)
        p.daemon = True
        p.start()

    def attn_send(self, x):
        free = self.free_tensors.get(x.shape)
        if free == None:
            self.free_tensors[x.shape] = []
            free = self.free_tensors[x.shape]

        if len(free) < 15:
            self.key += 2

            t = StepMeshTensorCache()

            t.push_tensor = torch.empty_like(x)
            t.pull_tensor = torch.empty_like(x)
            t.pull_tensor.zero_()

            t.push_key = self.key
            t.pull_key = self.key + 1

        else:
            t = free.pop(0)

        t.push_tensor.copy_(x)

        h = self.f.push_pull(
                [t.push_tensor],
                [t.push_key],
                [t.pull_tensor],
                [t.pull_key])

        t.h = h

        self.waits.append(t)

    def attn_recv(self):
        t = self.waits.pop(0)
        self.f.wait(t.h)

        self.free_tensors[t.push_tensor.shape].append(t)

        return t.pull_tensor.clone()

    def ffn_send(self, x):
        free = self.free_tensors.get(x.shape)
        if free == None:
            self.free_tensors[x.shape] = []
            free = self.free_tensors[x.shape]

        if len(free) < 15:
            t = torch.empty_like(x)
        else:
            t = free.pop(0)

        t.copy_(x)

        c = self.comm_ids.pop(0)
        self.f.respond([t], c, True)

        free.append(t)

    def ffn_recv(self):
        batches = self.f.get_batch()

        ## batches [
        #     [comm_id, push_tensor_list, key_list],
        #     [comm_id, push_tensor_list, key_list],
        # ]
        assert len(batches) == 1, "just handle for one worker"

        x = batches[0][1][0]
        key = batches[0][2][0]
        self.comm_ids.append(batches[0][0])

        if self.register_buf.get(key) == None:
            y = torch.empty_like(x)

            self.f.register_recv_buffer(y, [0], [key])

            self.register_buf[key] = y

        return x.clone()

    def recv_tensor(self) -> torch.Tensor:
        if self.perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN:
            return self.attn_recv()
        else:
            return self.ffn_recv()

    def send_tensor(self, x: torch.Tensor):
        if self.perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN:
            self.attn_send(x)
        else:
            self.ffn_send(x)

@cache
def get_tensor_communicator() -> FifoTensorCommunicator:
    afd_perspective = get_afd_perspective()
    if afd_perspective is not None:
        if os.environ.get("MLC_INTERFACE"):
            return StepMeshTensorCommunicator(afd_perspective)
        else:
            return ZMQSimpleTensorCommunicator(afd_perspective)
    else:
        raise NotImplementedError

def get_afd_mirco_batch() -> int:
    afd_mirco_batch = global_server_args_dict.get("afd_mirco_batch")
    return afd_mirco_batch

def get_afd_perspective() -> Optional[AFDPerspective]:
    afd_perspective = global_server_args_dict.get("afd_perspective")
    return afd_perspective

def afd_is_ffn():
    return get_afd_perspective() == AFDPerspective.AFD_PERSPECTIVE_FFN

def afd_is_attn():
    return get_afd_perspective() == AFDPerspective.AFD_PERSPECTIVE_ATTN

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

    pipeline_stages = (
        AFDStageScheduleGenerator.attn_stage(num_layers, m_stage)
        if afd_is_attn()
        else AFDStageScheduleGenerator.ffn_stage(num_layers, m_stage)
    )

    for stage in pipeline_stages:
        type, *args = stage
        stage_executors.get(type)(*args)

    try:
        results = [stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].popleft() for _ in range(m_stage)]
    except IndexError:
        raise ValueError("model_forward_afd: impossible path, a potential implementation bug?")

    all_hidden_states, all_residual = zip(
        *((res["hidden_states"], res["residual"]) for res in results)
    )

    return (
        torch.cat(all_hidden_states, dim=0),
        torch.cat(all_residual, dim=0) if afd_is_attn() else None,
    )

class AFDCommunicator(LayerCommunicator):
    def __init__(self, layer_communicator: LayerCommunicator, perspective: AFDPerspective, layer_id: int):
        self.perspective = perspective
        self.layer_communicator = layer_communicator
        self.layer_id = layer_id
        pass
    def prepare_attn(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        # just pass through
        if self.perspective == AFDPerspective.AFD_PERSPECTIVE_FFN:
            return hidden_states, residual

        return self.layer_communicator.prepare_attn(hidden_states, residual, forward_batch)

    def prepare_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if self.perspective == AFDPerspective.AFD_PERSPECTIVE_FFN:
            hidden_states = get_tensor_communicator().recv_tensor()
            return hidden_states, residual

        hidden_states, residual = self.layer_communicator.prepare_mlp(hidden_states, residual, forward_batch)
        get_tensor_communicator().send_tensor(hidden_states)

        return hidden_states, residual

    def postprocess_layer(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if self.perspective == AFDPerspective.AFD_PERSPECTIVE_FFN:
            get_tensor_communicator().send_tensor(hidden_states)
            return hidden_states, residual

        hidden_states = get_tensor_communicator().recv_tensor()

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        return hidden_states, residual

class AFDProxyAttention(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch) -> torch.Tensor:
        return hidden_states

class AFDProxyMLP(nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                forward_batch: Optional[ForwardBatch] = None) -> torch.Tensor:
        return hidden_states
