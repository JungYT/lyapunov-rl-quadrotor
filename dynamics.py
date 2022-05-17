import numpy as np
import numpy.random as random
import gym
from fym.core import BaseEnv, BaseSystem
import fym.utils.rot as rot

def hat(v):
    v1, v2, v3 = v.ravel()
    return np.array([
        [0, -v3, v2],
        [v3, 0, -v1],
        [-v2, v1, 0]
    ])


def compute_test_init(num_sample=5):
    initials = []
    random.seed(0)
    while True:
        pos_init = 20 * (np.vstack([2, 2, 1])*random.rand(3, 1) \
            - np.vstack([1, 1, 0]))
        vel_init = 20 * (2*random.rand(3, 1) - 1)
        psi, theta, phi = np.pi * (2*random.rand(3) - 1)
        quat_init = rot.angle2quat(psi, theta, phi)
        omega_init = 10 * (2*random.rand(3, 1) - 1)
        tmp = np.vstack((pos_init, vel_init, quat_init, omega_init))
        initials.append(tmp)
        if len(initials) == num_sample-1:
            pos_init = np.vstack([10, 10, 5])
            vel_init = np.vstack([0, 0, 1])
            quat_init = rot.angle2quat(
                15*np.pi/180, 15*np.pi/180, 15*np.pi/180
            )
            omega_init = np.vstack((0., 0., 0.))
            tmp = np.vstack((pos_init, vel_init, quat_init, omega_init))
            initials.append(tmp)
            break
    return initials


class Quadrotor(BaseEnv):
    def __init__(self):
        super().__init__()
        self.pos = BaseSystem(shape=(3,1))
        self.vel = BaseSystem(shape=(3,1))
        self.quat = BaseSystem(shape=(4,1))
        self.omega = BaseSystem(shape=(3,1))

        self.e3 = np.vstack((0., 0., 1.))
        self.g = -9.81 * self.e3
        self.m = 4.34
        self.J = np.diag([0.0820, 0.0845, 0.1377])
        self.J_inv = np.linalg.inv(self.J)
        d = 0.315 # Distance from center of mass to center of each rotor
        ctf = 8.004e-4 # Torque coefficient. ``torquad_i = (-1)^i cft f_i``
        self.B = np.array([
            [1, 1, 1, 1],
            [0, -d, 0, d],
            [d, 0, -d, 0],
            [-ctf, ctf, -ctf, ctf]
        ])
        self.B_inv = np.linalg.inv(self.B)
        self.thrust_max = 15.

    def set_dot(self, thrust):
        fM = self.convert_thrust2FM(thrust)
        f, M1, M2, M3 = fM
        F = f * self.e3
        M = np.vstack((M1, M2, M3))
        _, vel, quat, omega = self.observe_list()
        dcm = rot.quat2dcm(quat)
        omega_hat = hat(omega)
        self.pos.dot = vel
        self.vel.dot = self.g + dcm.T.dot(F)
        p, q, r = omega.ravel()
        eps = 1 - (quat[0]**2 + quat[1]**2 + quat[2]**2 + quat[3]**2)
        k = 1
        self.quat.dot = 0.5 * np.array([[0., -p, -q, -r],
                                        [p, 0., r, -q],
                                        [q, -r, 0., p],
                                        [r, q, -p, 0.]]).dot(quat) + k*eps*quat
        self.omega.dot = self.J_inv.dot(M - omega_hat.dot(self.J.dot(omega)))

    def convert_thrust2FM(self, thrust):
        """Convert thust of each rotor to force and moment
        Parameters:
            thrust: (4,1) array
        """
        return (self.B.dot(thrust)).ravel()

    def convert_FM2thrust(self, fM):
        """Convert force and moment to thust of each rotor
        Parameters:
            fM: (4,1)
        """
        return self.B_inv.dot(fM)


class Env(BaseEnv, gym.Env):
    def __init__(self, env_config):
        super().__init__(**env_config)
        self.quad = Quadrotor()

        self.action_space = gym.spaces.Box(
            low=np.float32(0.),
            high=np.float32(self.quad.thrust_max),
            shape=(4,)
        )
        # state_bound = np.vstack([
        #         np.inf, np.inf, np.inf,
        #         np.inf, np.inf, np.inf,
        #         1, 1, 1, 1, 1, 1, 1, 1, 1,
        #         np.inf, np.inf, np.inf
        #     ])
        # state_bound = np.vstack([
        #         1, 1, 1, 1, 1, 1, 1, 1, 1,
        #         np.inf, np.inf, np.inf
        #     ])
        state_ubound = np.vstack([
                np.inf, np.inf, np.inf,
                np.inf, np.inf, np.inf,
                1., 1., 1., 1., 0., 0.,
                np.inf, np.inf, np.inf
            ])
        state_lbound = np.vstack([
                -np.inf, -np.inf, -np.inf,
                -np.inf, -np.inf, -np.inf,
                -1., -1., -1., -1., -2., -2.,
                -np.inf, -np.inf, -np.inf
            ])
        self.observation_space = gym.spaces.Box(
            low=np.float32(state_lbound),
            high=np.float32(state_ubound),
            shape=(15,1)
        )

    def observe(self):
        pos = self.quad.pos.state
        vel = self.quad.vel.state
        quat = self.quad.quat.state
        omega = self.quad.omega.state
        # C_reshaped = np.reshape(self.quad.C.state, (-1, 1))
        psi, theta, phi = rot.quat2angle(quat)
        des_pos = np.vstack((0, 0, 10))
        des_psi = 0
        e_pos = pos - des_pos
        e_attitude = np.vstack((
            np.cos(phi), np.sin(phi),
            np.cos(theta), np.sin(theta),
            np.cos(psi) - np.cos(des_psi), np.sin(psi) - np.sin(psi)
        ))
        x = np.vstack((e_pos, vel, e_attitude, omega))
        obs = np.float32(x)
        return obs
        
    def reset(self, initial="random"):
        if initial == "random":
            pos_init = 20 * (np.vstack([2, 2, 1])*random.rand(3, 1) \
                        - np.vstack([1, 1, 0]))
            vel_init = 20 * (2*random.rand(3, 1) - 1)
            psi, theta, phi = np.pi * (2*random.rand(3) - 1)
            quat_init = rot.angle2quat(psi, theta, phi)
            omega_init = 2*np.pi * (2*random.rand(3, 1) - 1)
            self.quad.initial_state = np.vstack((
                pos_init, vel_init, quat_init, omega_init
            ))
        else:
            self.quad.initial_state = initial
        super().reset()
        obs = self.observe()
        return obs

    def set_dot(self, t, u):
        self.quad.set_dot(u)
        obs = self.observe()
        V = self.lyapunov(obs)
        return dict(t=t, **self.observe_dict(), thrust=u, obs=obs, lyapunov=V)

    def step(self, action):
        pre_obs = self.observe()
        u = np.vstack(action)
        *_, done = self.update(u=u)
        next_obs = self.observe()
        reward = self.get_reward(pre_obs, next_obs, u)
        info = {}
        return next_obs, reward, done, info

    def get_reward(self, pre_obs, next_obs, u):
        V_pre, _ = self.lyapunov(pre_obs)
        V_next, e_next = self.lyapunov(next_obs)
        del_V = V_next - V_pre
        exp = np.float32(np.exp(
            (-e_next.T @ np.diag([1, 1, 1, 1, 1]) @ e_next \
             -u.T @ np.diag([0, 0, 0, 0]) @ u).item()
        ))
        if (del_V<=-1e-7 and V_next>1e-6) or (del_V<=0  and V_next<=1e-6):
            reward = -1 + exp
        else:
            reward = -10 + exp
        return reward

    def lyapunov(self, obs):
        e = np.vstack((obs[0:3], obs[10:12]))
        P = np.diag([1, 1, 1, 1, 1])
        V = e.T @ P @ e
        # V = e_att.T @ e_att
        return V, e

