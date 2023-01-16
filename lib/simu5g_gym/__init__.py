#
# Copyright (C) 2022 Anjie Qiu, <qiu@eit.uni-kl.de>
#
# Documentation for these modules is at
# http://veins.car2x.org/
# http://simu5g.org/
#
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#

"""
Simu5G-Gym base structures to create gym environments from simu5g simulations.
"""

import atexit
import logging
import os
import signal
import subprocess
import sys
from typing import Any, Dict, NamedTuple
import warnings

import gym
import numpy as np
import zmq
from gym import error, spaces, utils
from gym.utils import seeding

from . import simu5g_gym_pb2

SENTINEL_EMPTY_SPACE = gym.spaces.Space()
SENTINEL_NO_SEED_GIVEN = "NO SEED GIVEN"  # replace with pep 661 once ready


class StepResult(NamedTuple):
    """Result record from one step in the environment."""

    observation: Any
    reward: np.float32
    done: bool
    info: Dict


def ensure_valid_scenario_dir(scenario_dir):
    """
    Raise an exception if path is not a valid scenario directory.
    """
    if scenario_dir is None:
        raise ValueError("No scenario_dir given.")
    if not os.path.isdir(scenario_dir):
        raise ValueError("The scenario_dir does not point to a directory.")
    if not os.path.exists(os.path.join(scenario_dir, "omnetpp.ini")):
        raise FileNotFoundError(
            "The scenario_dir needs to contain an omnetpp.ini file."
        )
    return True


def add_simu5g_root_dir_to_path(simu5g_root_dir):
    """
    Add simu5g_root_dir to path
    """
    # check if simu5g root dir exists already in PATH
    if 'simu5g/bin' in os.environ['PATH']:
        warnings.warn("simu5g executable is found in PATH.")
    else:
        warnings.warn("simu5g executable is not found in PATH, trying to add simu5g_root_dir to PATH")
        if simu5g_root_dir is None:
            raise ValueError("No simu5g_root_dir given.")

        sys.path.append(simu5g_root_dir)

        if not os.path.isdir(simu5g_root_dir):
            raise ValueError("The simu5g_root_dir does not point to a directory.")
        # if not os.path.exists(os.path.join(simu5g_root_dir, "simu5g")):
        #     raise FileNotFoundError(
        #         "The simu5g_root_dir needs to contain an simu5g executable."
        #     )


    return True


def launch_simu5g(
        scenario_dir,
        seed,
        port,
        print_stdout=True,
        extra_args=None,
        user_interface="Cmdenv",
        config="General",
):
    """
    Launch a simu5g experiment and return the process instance.

    All extra_args keys need to contain their own -- prefix.
    The respective values need to be correctly quouted.
    """
    command = [
        "./run",
        f"-u{user_interface}",
        f"-c{config}",
        f"--seed-set={seed}",
        f"--*.manager.seed={seed}",
        f"--*.gym_connection.port={port}",
    ]
    extra_args = dict() if extra_args is None else extra_args
    for key, value in extra_args.items():
        command.append(f"{key}={value}")
    # DEBUG:
    # debug = True
    # if debug:
    #    command.append('--debug-on-erros=true')

    logging.debug("Launching simu5g experiment using command `%s`", command)
    stdout = sys.stdout if print_stdout else subprocess.DEVNULL
    process = subprocess.Popen(command, stdout=stdout, cwd=scenario_dir)
    logging.debug("Simu5g process launched with pid %d", process.pid)
    return process


def shutdown_simu5g(process, gracetime_s=1.0):
    """
    Shut down simu5g if it still runs.
    """
    process.poll()
    if process.poll() is not None:
        logging.debug(
            "Simu5G process %d was shut down already with return code %d.",
            process.pid,
            process.returncode,
        )
        return
    process.terminate()
    try:
        process.wait(gracetime_s)
    except subprocess.TimeoutExpired as _exc:
        logging.warning(
            "Simu5g process %d did not shut down gracefully, sennding kill.",
            process.pid,
        )
        process.kill()
        try:
            process.wait(gracetime_s)
        except subprocess.TimeoutExpired as _exc2:
            logging.error(
                "Simu5g process %d could not even be killed!", process.pid
            )
    assert (
            process.poll() and process.returncode is not None
    ), "Simu5g could not be killed."


def serialize_action_discete(action):
    """Serialize a single discrete action into protobuf wire format."""
    reply = simu5g_gym_pb2.Reply()
    reply.action.discrete.value = action
    return reply.SerializeToString()


def parse_space(space):
    """Parse a Gym.spaces.Space from a protobuf request into python types."""
    if space.HasField("discrete"):
        return space.discrete.value
    if space.HasField("box"):
        return np.array(space.box.values, dtype=np.float32)
    if space.HasField("multi_discrete"):
        return np.array(space.multi_discrete.values, dtype=int)
    if space.HasField("multi_binary"):
        return np.array(space.multi_binary.values, dtype=bool)
    if space.HasField("tuple"):
        return tuple(parse_space(subspace) for subspace in space.tuple.values)
    if space.HasField("dict"):
        return {
            item.key: parse_space(item.space) for item in space.dict.values
        }
    raise RuntimeError("Unknown space type")


class Simu5gEnv(gym.Env):
    metadata = {"render.modes": []}

    default_scenario_dir = None
    default_simu5g_root_dir = None
    """
    Default scenario_dir argument for constructor.
    """

    # Anjie: add a new variable for simu5g_root_dir
    def __init__(
            self,
            scenario_dir=None,
            simu5g_root_dir=None,
            run_simu5g=True,
            port=None,
            timeout=10.0,
            print_simu5g_stdout=True,
            action_serializer=serialize_action_discete,
            simu5g_kwargs=None,
            user_interface="Cmdenv",
            config="General",
    ):
        if scenario_dir is None:
            scenario_dir = self.default_scenario_dir
        assert ensure_valid_scenario_dir(scenario_dir)
        if simu5g_root_dir is None:
            simu5g_root_dir = self.default_simu5g_root_dir
        assert add_simu5g_root_dir_to_path(simu5g_root_dir)
        self.scenario_dir = scenario_dir
        self._action_serializer = action_serializer

        self.action_space = SENTINEL_EMPTY_SPACE
        self.observation_space = SENTINEL_EMPTY_SPACE

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.port = port
        self.bound_port = None
        self._timeout = timeout
        self.print_simu5g_stdout = print_simu5g_stdout

        self.run_simu5g = run_simu5g
        self._passed_args = (
            simu5g_kwargs if simu5g_kwargs is not None else dict()
        )
        self._user_interface = user_interface
        self._config = config
        self._seed = 0
        self.simu5g = None
        self._simu5g_shutdown_handler = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    # Gym Env interface

    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        """
        self.socket.send(self._action_serializer(action))
        step_result = self._parse_request(self._recv_request())
        if step_result.done:
            self.socket.send(
                self._action_serializer(self.action_space.sample())
            )
            logging.debug("Episode ended")
            if self.simu5g:
                logging.debug("Waiting for simu5g to finish")
                self.simu5g.wait()
        assert self.observation_space.contains(step_result.observation)
        return step_result

    def reset(self, seed=SENTINEL_NO_SEED_GIVEN, return_info=False, options=None):
        """
        Start and connect to a new simu5g experiment, return first observation.

        Shut down existing simu5g experiment processes and connections.
        Waits until first request from simu5g experiment has been received.
        """
        del options  # currently not used/implemented

        self.close()
        self.socket = self.context.socket(zmq.REP)
        if self.port is None:
            self.bound_port = self.socket.bind_to_random_port(
                "tcp://127.0.0.1"
            )
            logging.debug("Listening on random port %d", self.bound_port)
        else:
            self.socket.bind(f"tcp://127.0.0.1:{self.port}")
            self.bound_port = self.port
            logging.debug("Listening on configured port %d", self.bound_port)

        if seed is not SENTINEL_NO_SEED_GIVEN:
            self.seed(seed)

        if self.run_simu5g:
            self.simu5g = launch_simu5g(
                self.scenario_dir,
                self._seed,
                self.bound_port,
                self.print_simu5g_stdout,
                self._passed_args,
                self._user_interface,
                self._config,
            )
            logging.info("Launched simu5g experiment, waiting for request.")

            def simu5g_shutdown_handler(signum=None, stackframe=None):
                """
                Ensure that simu5g always gets shut down on python exit.

                This is implemented as a local function on purpose.
                There could be more than one Simu5gEnv in one python process.
                So calling atexit.unregister(shutdown_simu5g) could cause leaks.
                """
                shutdown_simu5g(self.simu5g)
                if signum is not None:
                    sys.exit()

            atexit.register(simu5g_shutdown_handler)
            signal.signal(signal.SIGTERM, simu5g_shutdown_handler)
            self._simu5g_shutdown_handler = simu5g_shutdown_handler

        initial_request = self._parse_request(self._recv_request())[0]
        logging.info("Received first request from Simu5g, ready to run.")
        if return_info:
            logging.warning("return info not yet implemented for reset()")
            initial_info = None  # not implemented
            return initial_request, initial_info

        return initial_request

    def render(self, mode="human"):
        """
        Render current environment (not supported by Simu5gEnv right now).
        """
        raise NotImplementedError(
            "Rendering is not implemented for this Simu5g_Gym"
        )

    def close(self):
        """
        Close the episode and shut down simu5g scenario and connection.
        """
        logging.info("Closing Simu5gEnv.")
        if self._simu5g_shutdown_handler is not None:
            atexit.unregister(self._simu5g_shutdown_handler)

        if self.simu5g:
            # TODO: send shutdown message (which needs to be implemented in simu5g code)
            shutdown_simu5g(self.simu5g)
            self.simu5g = None

        if self.bound_port:
            logging.debug("Closing Simu5gEnv server socket.")
            self.socket.unbind(f"tcp://127.0.0.1:{self.bound_port}")
            self.socket.close()
            self.socket = None
            self.bound_port = None
            self.simu5g = None

    def seed(self, seed=None):
        """
        Set and return seed for the next episode.

        Will generate a random seed if None is passed.
        """
        if seed is not None:
            logging.debug("Setting given seed %d", seed)
            self._seed = seed
        else:
            random_seed = gym.utils.seeding.create_seed(max_bytes=4)
            logging.debug("Setting random seed %d", random_seed)
            self._seed = seed
        return [self._seed]

    # Internal helpers

    def _recv_request(self):
        rlist, _, _ = zmq.select([self.socket], [], [], timeout=self._timeout)
        if not rlist:
            logging.error(
                "Simu5g instance with PID %d timed out after %.2f seconds",
                self.simu5g.pid,
                self._timeout,
            )
            raise TimeoutError(
                f"Simu5g instance did not send a request within {self._timeout}"
                " seconds"
            )
        assert rlist == [self.socket]
        return self.socket.recv()

    def _parse_request(self, data):
        request = simu5g_gym_pb2.Request()
        request.ParseFromString(data)
        if request.HasField("shutdown"):
            return StepResult(self.observation_space.sample(), 0.0, True, {})
        if request.HasField("init"):
            # parse spaces
            self.action_space = eval(request.init.action_space_code)
            self.observation_space = eval(request.init.observation_space_code)
            # sent empty reply
            init_msg = simu5g_gym_pb2.Reply()
            self.socket.send(init_msg.SerializeToString())
            # request next request (actual request with content)
            real_data = self._recv_request()
            real_request = simu5g_gym_pb2.Request()
            real_request.ParseFromString(real_data)
            # continue processing the real request
            request = real_request
        # the gym needs to be initialized at this point!
        assert self.action_space is not SENTINEL_EMPTY_SPACE
        assert self.observation_space is not SENTINEL_EMPTY_SPACE
        observation = parse_space(request.step.observation)
        reward = parse_space(request.step.reward)
        assert len(reward) == 1
        return StepResult(observation, reward[0], False, {})
