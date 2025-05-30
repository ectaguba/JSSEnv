import bisect
import datetime
import random

import pandas as pd
import gymnasium as gym
import numpy as np
import plotly.figure_factory as ff
from pathlib import Path


class JssEnv(gym.Env):
    def __init__(self, env_config=None):
        """
        This environment model the job shop scheduling problem as a single agent problem:

        -The actions correspond to a job allocation + one action for no allocation at this time step (NOPE action)

        -We keep a time with next possible time steps

        -Each time we allocate a job, the end of the job is added to the stack of time steps

        -If we don't have a legal action (i.e. we can't allocate a job),
        we automatically go to the next time step until we have a legal action

        MDP components:
            - States (s)
            - Actions (a): allocate a job's next operation, or NO-OP to advance time
            - Transitions (P): deterministic updates of job progress and machine availability
            - Rewards (R): positive for processing time, negative for idle planning, scaled
            - Episode termination: when all jobs finish (no legal actions remain)

        -
        :param env_config: Ray dictionary of config parameter
        """
        # Load instance data (jobs, machines, operation times)
        if env_config is None:
            print(f"Getting instance data from: {Path(__file__).parent.absolute() / 'instances' / 'ta80'}")
            env_config = {
                "instance_path": Path(__file__).parent.absolute() / "instances" / "ta80"
            }
        instance_path = env_config["instance_path"]

        # initial values for variables used for instance
        self.jobs = 0
        self.machines = 0
        self.instance_matrix = None
        self.jobs_length = None
        self.max_time_op = 0
        self.max_time_jobs = 0
        self.nb_legal_actions = 0
        self.nb_machine_legal = 0
        # initial values for variables used for solving (to reinitialize when reset() is called)
        self.solution = None
        self.last_solution = None
        self.last_time_step = float("inf")
        self.current_time_step = float("inf")
        self.next_time_step = list()
        self.next_jobs = list()
        self.legal_actions = None
        self.time_until_available_machine = None
        self.time_until_finish_current_op_jobs = None
        self.todo_time_step_job = None
        self.total_perform_op_time_jobs = None
        self.needed_machine_jobs = None
        self.total_idle_time_jobs = None
        self.idle_time_jobs_last_op = None
        self.state = None
        self.illegal_actions = None
        self.action_illegal_no_op = None
        self.machine_legal = None
        # initial values for variables used for representation
        self.start_timestamp = datetime.datetime.now().timestamp()
        self.sum_op = 0
        with open(instance_path, "r") as instance_file:
            for line_cnt, line_str in enumerate(instance_file, start=1):
                split_data = list(map(int, line_str.split()))

                if line_cnt == 1:
                    self.jobs, self.machines = split_data
                    self.instance_matrix = np.zeros((self.jobs, self.machines), dtype=(int, 2))
                    self.jobs_length = np.zeros(self.jobs, dtype=int)
                else:
                    assert len(split_data) % 2 == 0 and len(split_data) // 2 == self.machines
                    job_nb = line_cnt - 2
                    for i in range(0, len(split_data), 2):
                        machine, time = split_data[i], split_data[i + 1]
                        self.instance_matrix[job_nb][i // 2] = (machine, time)
                        self.max_time_op = max(self.max_time_op, time)
                        self.jobs_length[job_nb] += time
                        self.sum_op += time
        self.max_time_jobs = max(self.jobs_length)
        # check the parsed data are correct
        assert self.max_time_op > 0
        assert self.max_time_jobs > 0
        assert self.jobs > 0
        assert self.machines > 1, "We need at least 2 machines"
        assert self.instance_matrix is not None
        
        # ACTION SPACE: J jobs + 1 NO-OP action
        # a in {0..J-1}: start next op of job a; a=J: fast-forward time
        # allocate a job + one to wait
        self.action_space = gym.spaces.Discrete(self.jobs + 1)
        self.action_space = gym.spaces.Discrete(self.jobs + 1)
        
        # used for plotting
        self.colors = [
            tuple([random.random() for _ in range(3)]) for _ in range(self.machines)
        ]
        """
        matrix with the following attributes for each job:
            -Legal job
            -Left over time on the current op
            -Current operation %
            -Total left over time
            -When next machine available
            -Time since IDLE: 0 if not available, time otherwise
            -Total IDLE time in the schedule
        """
        # OBSERVATION SPACE:
        #  - action_mask: binary vector length J+1
        #  - real_obs: normalized features matrix J x 7
        
        self.observation_space = gym.spaces.Dict(
            {
                "action_mask": gym.spaces.Box(0, 1, shape=(self.jobs + 1,)),
                "real_obs": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(self.jobs, 7), dtype=float
                ),
            }
        )

    # def seed(self, seed=None):
    #     self.np_random, seed = gym.utils.seeding.np_random(seed)
    #     return [seed]
    
    def _get_current_state_representation(self):
        self.state[:, 0] = self.legal_actions[:-1]
        return {
            "real_obs": self.state,
            "action_mask": self.legal_actions,
        }

    def get_legal_actions(self):
        return self.legal_actions
    
    def action_masks(self):
        """Return a 1-D bool array of length `action_space.n`.
        `True`  → action is legal
        `False` → action is illegal  (will be masked)"""
        return self.legal_actions

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
    
        :param seed: Optional seed for the environment's random number generator.
        :param options: Additional options for resetting the environment (not used here).
        :return: A tuple (observation, info), where observation is the initial state and info is an empty dictionary.
        """
        # Set the seed if provided
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
    
        # Initialize scheduling state variables
        self.current_time_step = 0
        self.next_time_step = list()
        self.next_jobs = list()
        self.nb_legal_actions = self.jobs
        self.nb_machine_legal = 0
        
        # All job-actions legal; NO-OP illegal initially
        self.legal_actions = np.ones(self.jobs + 1, dtype=bool)
        self.legal_actions[self.jobs] = False
        
        # Used to represent the solution
        self.solution = np.full((self.jobs, self.machines), -1, dtype=int)
        
        self.time_until_available_machine = np.zeros(self.machines, dtype=int)
        self.time_until_finish_current_op_jobs = np.zeros(self.jobs, dtype=int)
        self.todo_time_step_job = np.zeros(self.jobs, dtype=int)
        self.total_perform_op_time_jobs = np.zeros(self.jobs, dtype=int)
        self.needed_machine_jobs = np.zeros(self.jobs, dtype=int)
        self.total_idle_time_jobs = np.zeros(self.jobs, dtype=int)
        self.idle_time_jobs_last_op = np.zeros(self.jobs, dtype=int)
        self.illegal_actions = np.zeros((self.machines, self.jobs), dtype=bool)
        self.action_illegal_no_op = np.zeros(self.jobs, dtype=bool)
        self.machine_legal = np.zeros(self.machines, dtype=bool)
        
        for job in range(self.jobs):
            needed_machine = self.instance_matrix[job][0][0]
            self.needed_machine_jobs[job] = needed_machine
            if not self.machine_legal[needed_machine]:
                self.machine_legal[needed_machine] = True
                self.nb_machine_legal += 1
                
        # Feature state matrix (zeros)
        self.state = np.zeros((self.jobs, 7), dtype=float)
        
        # Return initial MDP state and empty info
        return self._get_current_state_representation(), {}
    
        # self.current_time_step = 0
        # self.next_time_step = list()
        # self.next_jobs = list()
        # self.nb_legal_actions = self.jobs
        # self.nb_machine_legal = 0
        # # represent all the legal actions
        # self.legal_actions = np.ones(self.jobs + 1, dtype=bool)
        # self.legal_actions[self.jobs] = False
        # # used to represent the solution
        # self.solution = np.full((self.jobs, self.machines), -1, dtype=int)
        # self.time_until_available_machine = np.zeros(self.machines, dtype=int)
        # self.time_until_finish_current_op_jobs = np.zeros(self.jobs, dtype=int)
        # self.todo_time_step_job = np.zeros(self.jobs, dtype=int)
        # self.total_perform_op_time_jobs = np.zeros(self.jobs, dtype=int)
        # self.needed_machine_jobs = np.zeros(self.jobs, dtype=int)
        # self.total_idle_time_jobs = np.zeros(self.jobs, dtype=int)
        # self.idle_time_jobs_last_op = np.zeros(self.jobs, dtype=int)
        # self.illegal_actions = np.zeros((self.machines, self.jobs), dtype=bool)
        # self.action_illegal_no_op = np.zeros(self.jobs, dtype=bool)
        # self.machine_legal = np.zeros(self.machines, dtype=bool)
        # for job in range(self.jobs):
        #     needed_machine = self.instance_matrix[job][0][0]
        #     self.needed_machine_jobs[job] = needed_machine
        #     if not self.machine_legal[needed_machine]:
        #         self.machine_legal[needed_machine] = True
        #         self.nb_machine_legal += 1
        # self.state = np.zeros((self.jobs, 7), dtype=float)
        # return self._get_current_state_representation()

    def _prioritization_non_final(self):
        if self.nb_machine_legal >= 1:
            for machine in range(self.machines):
                if self.machine_legal[machine]:
                    final_job = list()
                    non_final_job = list()
                    min_non_final = float("inf")
                    for job in range(self.jobs):
                        if (
                            self.needed_machine_jobs[job] == machine
                            and self.legal_actions[job]
                        ):
                            if self.todo_time_step_job[job] == (self.machines - 1):
                                final_job.append(job)
                            else:
                                current_time_step_non_final = self.todo_time_step_job[
                                    job
                                ]
                                time_needed_legal = self.instance_matrix[job][
                                    current_time_step_non_final
                                ][1]
                                machine_needed_nextstep = self.instance_matrix[job][
                                    current_time_step_non_final + 1
                                ][0]
                                if (
                                    self.time_until_available_machine[
                                        machine_needed_nextstep
                                    ]
                                    == 0
                                ):
                                    min_non_final = min(
                                        min_non_final, time_needed_legal
                                    )
                                    non_final_job.append(job)
                    if len(non_final_job) > 0:
                        for job in final_job:
                            current_time_step_final = self.todo_time_step_job[job]
                            time_needed_legal = self.instance_matrix[job][
                                current_time_step_final
                            ][1]
                            if time_needed_legal > min_non_final:
                                self.legal_actions[job] = False
                                self.nb_legal_actions -= 1

    def _check_no_op(self):
        self.legal_actions[self.jobs] = False
        if (
            len(self.next_time_step) > 0
            and self.nb_machine_legal <= 3
            and self.nb_legal_actions <= 4
        ):
            machine_next = set()
            next_time_step = self.next_time_step[0]
            max_horizon = self.current_time_step
            max_horizon_machine = [
                self.current_time_step + self.max_time_op for _ in range(self.machines)
            ]
            for job in range(self.jobs):
                if self.legal_actions[job]:
                    time_step = self.todo_time_step_job[job]
                    machine_needed = self.instance_matrix[job][time_step][0]
                    time_needed = self.instance_matrix[job][time_step][1]
                    end_job = self.current_time_step + time_needed
                    if end_job < next_time_step:
                        return
                    max_horizon_machine[machine_needed] = min(
                        max_horizon_machine[machine_needed], end_job
                    )
                    max_horizon = max(max_horizon, max_horizon_machine[machine_needed])
            for job in range(self.jobs):
                if not self.legal_actions[job]:
                    if (
                        self.time_until_finish_current_op_jobs[job] > 0
                        and self.todo_time_step_job[job] + 1 < self.machines
                    ):
                        time_step = self.todo_time_step_job[job] + 1
                        time_needed = (
                            self.current_time_step
                            + self.time_until_finish_current_op_jobs[job]
                        )
                        while (
                            time_step < self.machines - 1 and max_horizon > time_needed
                        ):
                            machine_needed = self.instance_matrix[job][time_step][0]
                            if (
                                max_horizon_machine[machine_needed] > time_needed
                                and self.machine_legal[machine_needed]
                            ):
                                machine_next.add(machine_needed)
                                if len(machine_next) == self.nb_machine_legal:
                                    self.legal_actions[self.jobs] = True
                                    return
                            time_needed += self.instance_matrix[job][time_step][1]
                            time_step += 1
                    elif (
                        not self.action_illegal_no_op[job]
                        and self.todo_time_step_job[job] < self.machines
                    ):
                        time_step = self.todo_time_step_job[job]
                        machine_needed = self.instance_matrix[job][time_step][0]
                        time_needed = (
                            self.current_time_step
                            + self.time_until_available_machine[machine_needed]
                        )
                        while (
                            time_step < self.machines - 1 and max_horizon > time_needed
                        ):
                            machine_needed = self.instance_matrix[job][time_step][0]
                            if (
                                max_horizon_machine[machine_needed] > time_needed
                                and self.machine_legal[machine_needed]
                            ):
                                machine_next.add(machine_needed)
                                if len(machine_next) == self.nb_machine_legal:
                                    self.legal_actions[self.jobs] = True
                                    return
                            time_needed += self.instance_matrix[job][time_step][1]
                            time_step += 1

    def step(self, action: int):
        """
        Apply action a_t:
          - If a < J: start job a's next operation -> reward = processing time
          - If a = J (NO-OP): advance time to next completion -> reward = -idle penalty
        Then update state s_{t+1}, compute scaled reward, check termination.
        Returns (obs, reward, terminated, truncated, info).
        """
        if action < self.jobs:
            current_op = self.todo_time_step_job[action]
            if current_op >= self.machines:
                # job has no more operations → convert to NO-OP
                action = self.jobs
                
        reward = 0.0
        terminated = False
        truncated = False
        
        if action == self.jobs:
            # NO-OP: fast-forward time until at least one machine is free
            # Penalize idle "hole planning"
            self.nb_machine_legal = 0
            self.nb_legal_actions = 0
            
            for job in range(self.jobs):
                if self.legal_actions[job]:
                    self.legal_actions[job] = False
                    needed_machine = self.needed_machine_jobs[job]
                    self.machine_legal[needed_machine] = False
                    self.illegal_actions[needed_machine][job] = True
                    self.action_illegal_no_op[job] = True

            # NO-OP: penalize agent, only fast-forward while there are pending completions
            while self.nb_machine_legal == 0 and len(self.next_time_step) > 0:
                reward -= self.increase_time_step()
            
            # if nb_machine_legal==0 here AND no pending events,
            # we simply exit the loop without crashing

            # scale reward
            scaled_reward = self._reward_scaler(reward)
            
            # update masks, prioritization, etc.
            self._prioritization_non_final()
            self._check_no_op()
            
            # termination check
            terminated = self._is_done()
            
            return (
                self._get_current_state_representation(),
                scaled_reward,
                terminated,
                truncated,
                {},
            )
        else:
            # Schedule next operation of job 'action'
            # Immediate reward = operation processing time
            current_time_step_job = self.todo_time_step_job[action]
            machine_needed = self.needed_machine_jobs[action]

            a_to_im = self.instance_matrix[action]
            time_needed = a_to_im[current_time_step_job][1]
            
            reward += time_needed # reward
            
            self.time_until_available_machine[machine_needed] = time_needed
            self.time_until_finish_current_op_jobs[action] = time_needed
            self.state[action][1] = time_needed / self.max_time_op
            to_add_time_step = self.current_time_step + time_needed
            if to_add_time_step not in self.next_time_step:
                index = bisect.bisect_left(self.next_time_step, to_add_time_step)
                self.next_time_step.insert(index, to_add_time_step)
                self.next_jobs.insert(index, action)
            self.solution[action][current_time_step_job] = self.current_time_step
            for job in range(self.jobs):
                if (
                    self.needed_machine_jobs[job] == machine_needed
                    and self.legal_actions[job]
                ):
                    self.legal_actions[job] = False
                    self.nb_legal_actions -= 1
                    
            if self.nb_machine_legal > 0:
                self.nb_machine_legal -= 1
                self.machine_legal[machine_needed] = False
            
            for job in range(self.jobs):
                if self.illegal_actions[machine_needed][job]:
                    self.action_illegal_no_op[job] = False
                    self.illegal_actions[machine_needed][job] = False
                    
            # If no machines free, advance time automatically
            while self.nb_machine_legal == 0 and len(self.next_time_step) > 0:
                reward -= self.increase_time_step()
               
            # Determine which actions are legal next 
            self._prioritization_non_final()
            self._check_no_op()
            
            # Scale reward to [-1,1] range by max operation time
            scaled_reward = self._reward_scaler(reward)
            
            # Check for terminal state (all jobs done)
            terminated = self._is_done()
            
            # Return new observation, reward, done flags
            return (
                self._get_current_state_representation(),
                scaled_reward,
                terminated,
                truncated,
                {},
            )
    
    def _reward_scaler(self, reward):
        """Scale raw reward by the maximum operation time."""
        return reward / self.max_time_op

    def increase_time_step(self):
        """
        Internal time-advance:
          - Pop next completion event
          - Advance current_time_step
          - Update remaining-times and idle metrics for jobs & machines
        Returns total idle time across machines ("hole planning" penalty).
        """
        hole_planning = 0
        
        next_time_step_to_pick = self.next_time_step.pop(0)
        self.next_jobs.pop(0)
        
        difference = next_time_step_to_pick - self.current_time_step
        self.current_time_step = next_time_step_to_pick
        
        for job in range(self.jobs):
            was_left_time = self.time_until_finish_current_op_jobs[job]
            if was_left_time > 0:
                performed_op_job = min(difference, was_left_time)
                self.time_until_finish_current_op_jobs[job] = max(
                    0, self.time_until_finish_current_op_jobs[job] - difference
                )
                self.state[job][1] = (
                    self.time_until_finish_current_op_jobs[job] / self.max_time_op
                )
                self.total_perform_op_time_jobs[job] += performed_op_job
                self.state[job][3] = (
                    self.total_perform_op_time_jobs[job] / self.max_time_jobs
                )
                if self.time_until_finish_current_op_jobs[job] == 0:
                    self.total_idle_time_jobs[job] += difference - was_left_time
                    self.state[job][6] = self.total_idle_time_jobs[job] / self.sum_op
                    self.idle_time_jobs_last_op[job] = difference - was_left_time
                    self.state[job][5] = self.idle_time_jobs_last_op[job] / self.sum_op
                    self.todo_time_step_job[job] += 1
                    self.state[job][2] = self.todo_time_step_job[job] / self.machines
                    if self.todo_time_step_job[job] < self.machines:
                        self.needed_machine_jobs[job] = self.instance_matrix[job][
                            self.todo_time_step_job[job]
                        ][0]
                        self.state[job][4] = (
                            max(
                                0,
                                self.time_until_available_machine[
                                    self.needed_machine_jobs[job]
                                ]
                                - difference,
                            )
                            / self.max_time_op
                        )
                    else:
                        self.needed_machine_jobs[job] = -1
                        # this allow to have 1 is job is over (not 0 because, 0 strongly indicate that the job is a good candidate)
                        self.state[job][4] = 1.0
                        if self.legal_actions[job]:
                            self.legal_actions[job] = False
                            self.nb_legal_actions -= 1
            elif self.todo_time_step_job[job] < self.machines:
                self.total_idle_time_jobs[job] += difference
                self.idle_time_jobs_last_op[job] += difference
                self.state[job][5] = self.idle_time_jobs_last_op[job] / self.sum_op
                self.state[job][6] = self.total_idle_time_jobs[job] / self.sum_op
        for machine in range(self.machines):
            if self.time_until_available_machine[machine] < difference:
                empty = difference - self.time_until_available_machine[machine]
                hole_planning += empty
            self.time_until_available_machine[machine] = max(
                0, self.time_until_available_machine[machine] - difference
            )
            if self.time_until_available_machine[machine] == 0:
                for job in range(self.jobs):
                    if (
                        self.needed_machine_jobs[job] == machine
                        and not self.legal_actions[job]
                        and not self.illegal_actions[machine][job]
                    ):
                        self.legal_actions[job] = True
                        self.nb_legal_actions += 1
                        if not self.machine_legal[machine]:
                            self.machine_legal[machine] = True
                            self.nb_machine_legal += 1
        return hole_planning

    def _is_done(self):
        """Episode terminates when no legal job-actions remain (all jobs completed)."""
        if self.nb_legal_actions == 0:
            self.last_time_step = self.current_time_step
            self.last_solution = self.solution
            return True
        return False

    def render(self, mode="human"):
        """Optional: Gantt chart of the current schedule solution."""
        df = []
        # set the env_config instance path or else it won't render - Chris
        for job in range(self.jobs):
            i = 0
            while i < self.machines and self.solution[job][i] != -1:
                dict_op = dict()
                dict_op["Task"] = "Job {}".format(job)
                start_sec = self.start_timestamp + self.solution[job][i]
                finish_sec = start_sec + self.instance_matrix[job][i][1]
                dict_op["Start"] = datetime.datetime.fromtimestamp(start_sec)
                dict_op["Finish"] = datetime.datetime.fromtimestamp(finish_sec)
                dict_op["Resource"] = "Machine {}".format(
                    self.instance_matrix[job][i][0]
                )
                df.append(dict_op)
                i += 1
        fig = None
        if len(df) > 0:
            df = pd.DataFrame(df)
            fig = ff.create_gantt(
                df,
                index_col="Resource",
                colors=self.colors,
                show_colorbar=True,
                group_tasks=True,
            )
            fig.update_yaxes(
                autorange="reversed"
            )  # otherwise tasks are listed from the bottom up
            if hasattr(self, "_last_iteration"):
                fig.update_layout(
                    title_text=f"Iteration {self._last_iteration}",
                    title_x=0.5,
                )
        return fig


if __name__ == '__main__':
    env = JssEnv()
    obs = env.reset(seed=42)
    done = False
    cum_reward = 0
    while not done:
        legal_actions = obs["action_mask"]
        actions = np.random.choice(
            len(legal_actions), 1, p=(legal_actions / legal_actions.sum())
        )[0]
        observation, reward, terminated, truncated, info = env.step(actions)
        cum_reward += rewards
    print(f"Cumulative reward: {cum_reward}")

