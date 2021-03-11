import csv
from typing import Optional
import gym
from gym import spaces
import numpy as np
from sympy import lambdify
from scipy.integrate import odeint
from lifting_rl.n_linkage import kane
from scipy import interpolate
from gym.envs.classic_control import rendering


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

def get_coordinates(path):
    coordinates = []
    with open(path) as fr:
        reader = csv.reader(fr)
        for idx, row in enumerate(reader):
            coordinates.append([float(i) for i in row])
    return np.array(coordinates)


def get_interpolated(coords, timestamps, mode="spline"):
    interpolated_coords = []
    for i in range(coords.shape[1]):
        if mode == "default":
            y = coords[:, i]
            f = interpolate.interp1d(timestamps, y)
        if mode == "spline":
            y = coords[:, i]

            def f(x, y=y):
                tck = interpolate.splrep(timestamps, y)
                return np.float32(interpolate.splev(x, tck))

        interpolated_coords.append(f)
    return interpolated_coords


class LinkageEnvV2(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        path: str,
        w_params: dict,
        target_pos: Optional[np.ndarray] = None,
        verbose: bool = False
    ):
        self.viewer = None
        self.n_links = w_params["N_LINKS"]

        M, F, m_params = kane(n=w_params["N_LINKS"])
        self.M_func = lambdify(m_params, M)
        self.F_func = lambdify(m_params, F)

        self.state_space = spaces.Box(
            low=w_params["OBS_LOW"],
            high=w_params["OBS_HIGH"],
            dtype=np.float32,
        )
        print("state_space: ", self.state_space)

        self.action_space = spaces.Box(
            low=w_params["ACT_LOW"],
            high=w_params["ACT_HIGH"],
            shape=(w_params["N_LINKS"],),
            dtype=np.float32,
        )
        print("action_space: ", self.action_space)

        self.cur_step = 0
        self.cur_time_step = 0

        self.trajectory_points = get_coordinates(path)
        num_frames = self.trajectory_points.shape[0]
        self.trajectory_timestamps = np.array(
            [i * 1.0 / w_params["VIDEO_FPS"] for i in range(num_frames)]
        )

        self.dt = w_params["TIME_STEP"]
        end_time = round(num_frames / w_params["VIDEO_FPS"], 2)

        self.interpolated_trajectory = get_interpolated(
            self.trajectory_points, self.trajectory_timestamps
        )
        timestamps = [
            np.float32(i * self.dt)
            for i in range(int(end_time // self.dt))
        ]
        self.coordinates = [
            [
                self.interpolated_trajectory[i](t)
                for i in range(len(self.interpolated_trajectory))
            ]
            for t in timestamps
        ]

        self.param_vals = w_params["PARAM_VALS"]
        self.target_pos = target_pos
        self.u = None
        self.state = None
        self.verbose = verbose
        enc_init_state = self.reset()

        self.observation_space = np.zeros_like(enc_init_state)

    def reset(self):
        init_coords = self.trajectory_points[0][: self.n_links]
        init_vel = np.array([0] * init_coords.shape[0])
        init_state = np.concatenate((init_coords, init_vel))
        self.state = init_state
        self.cur_step = 0
        self.cur_time_step = 0
        self.u = None
        enc_state = self.encode_state(self.state)
        return enc_state

    def reward(self, state, target_pos, u):
        state_pos = state[:self.n_links]
        state_pos = angle_normalize(state_pos)
        target_pos = angle_normalize(target_pos)
        state_vel = state[self.n_links:]

        pos_reward = np.exp( 0.1 * ((state_pos - target_pos)**2).sum() )
        vel_reward = np.sqrt( (state_vel**2).sum() )
        control_reward = u**2

        reward = -(pos_reward + vel_reward + control_reward)
        return reward[0]

    def get_cur_target_pos(self):
        if self.target_pos is None:
            cur_target_pos = np.array(self.coordinates[self.cur_step])
        else:
            cur_target_pos = self.target_pos
        return cur_target_pos

    def step(self, u):
        self.u = u
        state = self.state
        x = state
        t = self.cur_time_step
        dt = self.dt
        args = self.param_vals
        f = self._rhs

        cur_target_pos = self.get_cur_target_pos()
        reward = self.reward(state, cur_target_pos, u)

        # RK-4th order integration step
        k1 = f(x, t, args)
        k2 = f(x + (dt / 2) * k1, t + dt / 2, args)
        k3 = f(x + (dt / 2) * k2, t + dt / 2, args)
        k4 = f(x + dt * k3, t + dt / 2, args)
        nx = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

        next_state = nx
        next_t = t + self.dt
        next_step = self.cur_step + 1

        if self.target_pos is None:
            is_end = next_step >= len(self.coordinates)-1
        else:
            is_end = False

        self.state = next_state
        self.cur_step = next_step
        self.cur_time_step = next_t

        enc_state = self.encode_state(self.state)
        return enc_state, reward, is_end, {}

    def _is_out_of_bounds(self):
        return not self.state_space.contains(self.state)

    def encode_state(self, state: np.ndarray) -> np.ndarray:
        state_pos = state[:self.n_links]
        state_vel = state[self.n_links:]
        state_encoded = []
        for i in range(state_pos.shape[0]):
            q = state_pos[i]
            state_encoded.append(np.cos(q))
            state_encoded.append(np.sin(q))
            # state_encoded.append(q)
            # state_encoded.append(q**2)
            # state_encoded.append(np.sqrt(abs(q)))

        for i in range(state_vel.shape[0]):
            dq = state_vel[i]
            state_encoded.append(dq)
            # state_encoded.append(dq**2)
            # state_encoded.append(np.sqrt(abs(dq)))

        return np.array(state_encoded)

    def _rhs(self, x, t, args):
        arguments = np.hstack((x, self.u, args))
        dx = np.array(
            np.linalg.solve(self.M_func(*arguments), self.F_func(*arguments))
        ).T[0]
        return dx

    def _draw_n_link(self, s, color = (0, 1, 0)):
        link_lengths = [0.4]*self.n_links
        start_p = np.array([0, 0])
        p1 = None
        p2 = None
        for i in range(self.n_links):
            q = s[i]
            l = link_lengths[i]
            if p1 is None:
                p1 = start_p
            p2 = np.array([p1[0] + l * np.cos(q), p1[1] + l * np.sin(q)])
            #
            # self.viewer.draw_line((p1[0], p1[1]), (p2[0], p2[1]))

            aline = rendering.Line((p1[0], p1[1]), (p2[0], p2[1]))
            aline.set_color(*color)
            self.viewer.add_onetime(aline)
            p1 = p2

    def render(self, mode="human"):
        s = self.state
        t_s = self.get_cur_target_pos()

        if self.viewer is None:
            self.viewer = rendering.Viewer(400, 400)
            bound = 0.4 + 0.4 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound, bound, -bound, bound)



        if s is None:
            return None

        link_lengths = [0.4]*self.n_links
        start_p = np.array([0, 0])

        self.viewer.draw_line((-2.2, 1), (2.2, 1))

        if self.u is not None:
            u = self.u[0]
            max_u = self.action_space.high
            u_norm = u / max_u

            control_line = rendering.Line((0, 0), (u_norm * 0.5, 0))
            control_line.set_color(0, 0, 1)
            self.viewer.add_onetime(control_line)

        self._draw_n_link(s, (0, 1, 0))
        self._draw_n_link(t_s, (1, 0, 0))
        #
        # p1 = None
        # p2 = None
        # for i in range(self.n_links):
        #     q = s[i]
        #     l = link_lengths[i]
        #     if p1 is None:
        #         p1 = start_p
        #     p2 = np.array([p1[0] + l * np.cos(q), p1[1] + l * np.sin(q)])
        #     self.viewer.draw_line((p1[0], p1[1]), (p2[0], p2[1]))
        #     p1 = p2

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

        #
        #
        #
        # p1 = [0.4 * np.cos(s[0]), 0.4 * np.sin(s[0])]
        #
        # # p2 = [p1[0] + 0.4 * np.cos(s[1]), p1[1] + 0.4 * np.sin(s[1])]
        #
        # xys = np.array([[0, 0], p1])
        # # thetas = [s[0] % (2 * np.pi), (s[1]) % (2 * np.pi)]
        # thetas = [s[0] % (2 * np.pi)]
        #
        # # link_lengths = [0.4, 0.4]
        # link_lengths = [0.4]
        #
        # self.viewer.draw_line((-2.2, 1), (2.2, 1))
        # for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
        #     l, r, t, b = 0, llen, 0.05, -0.05
        #     jtransform = rendering.Transform(rotation=th, translation=(x, y))
        #     link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        #     link.add_attr(jtransform)
        #     link.set_color(0.8, 0.3, 0.3)
        #     circ = self.viewer.draw_circle(0.05)
        #     circ.set_color(0.0, 0.0, 0)
        #     circ.add_attr(jtransform)
        #
        # return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
