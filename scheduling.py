from typing import Union

import numpy as np
import torch


class EDICTScheduler:
    def __init__(
        self,
        p: float = 0.93,
        beta_1: float = 0.00085,
        beta_T: float = 0.012,
        num_train_timesteps: int = 1000,  # T = 1000
        set_alpha_to_one: bool = False,
        ext_scale: float = 0.02,
    ):
        self.p = p
        self.ext_scale = ext_scale
        self.num_train_timesteps = num_train_timesteps

        # scaled linear
        betas = (
            torch.linspace(
                beta_1**0.5, beta_T**0.5, num_train_timesteps, dtype=torch.float32
            )
            ** 2
        )

        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.final_alpha_cumprod = (
            torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        )

        # For PEP 412's sake
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(
            np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64)
        )

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device]):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps

        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def denoise_mixing_layer(self, x: torch.Tensor, y: torch.Tensor):
        # p = 0.93
        # x = p * x + (1 - p) * y
        # y = p * y + (1 - p) * x
        x = self.p * x + (1 - self.p) * y
        y = self.p * y + (1 - self.p) * x

        return [x, y]

    def denoise_mixing_layer_with_ext(self, x: torch.Tensor, y: torch.Tensor, delta_x: torch.Tensor,
                                      delta_y: torch.Tensor):
        x = (self.p * x + (1 - self.p) * y) * (1 - self.ext_scale) + self.ext_scale * delta_x
        y = (self.p * y + (1 - self.p) * x) * (1 - self.ext_scale) + self.ext_scale * delta_y
        return [x, y]

    def noise_mixing_layer(self, x: torch.Tensor, y: torch.Tensor):
        # p = 0.93
        # y = (y - (1 - p) * x) / p
        # x = (x - (1 - p) * y) / p
        y = (y - (1 - self.p) * x) / self.p
        x = (x - (1 - self.p) * y) / self.p

        return [x, y]

    def noise_mixing_layer_with_ext(self, x: torch.Tensor, y: torch.Tensor, delta_x: torch.Tensor,
                                      delta_y: torch.Tensor):
        y = ((y - self.ext_scale * delta_y) / (1 - self.ext_scale) - (1 - self.p) * x) / self.p
        x = ((x - self.ext_scale * delta_x) / (1 - self.ext_scale) - (1 - self.p) * y) / self.p

        return [x, y]
    def get_alpha_and_beta(self, t: torch.Tensor):
        # as self.alphas_cumprod is always in cpu
        t = int(t)

        alpha_prod = self.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod

        return alpha_prod, 1 - alpha_prod

    def noise_step(
        self,
        base: torch.Tensor,
        model_input: torch.Tensor,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
    ):
        prev_timestep = timestep - self.num_train_timesteps / self.num_inference_steps

        alpha_prod_t, beta_prod_t = self.get_alpha_and_beta(timestep)
        alpha_prod_t_prev, beta_prod_t_prev = self.get_alpha_and_beta(prev_timestep)

        a_t = (alpha_prod_t_prev / alpha_prod_t) ** 0.5
        b_t = -a_t * (beta_prod_t**0.5) + beta_prod_t_prev**0.5

        next_model_input = (base - b_t * model_output) / a_t

        return model_input, next_model_input.to(base.dtype)

    def noise_step_with_ext(
            self,
            base: torch.Tensor,
            model_input: torch.Tensor,
            model_output: torch.Tensor,
            timestep: torch.Tensor,
            model_output_ext: torch.Tensor,
    ):
        prev_timestep = timestep - self.num_train_timesteps / self.num_inference_steps

        alpha_prod_t, beta_prod_t = self.get_alpha_and_beta(timestep)
        alpha_prod_t_prev, beta_prod_t_prev = self.get_alpha_and_beta(prev_timestep)

        a_t = (alpha_prod_t_prev / alpha_prod_t) ** 0.5
        c_t = -a_t * (beta_prod_t ** 0.5)
        b_t = c_t + beta_prod_t_prev ** 0.5

        model_output_delta = self.ext_scale * (model_output_ext - model_output)
        next_model_input = (base - b_t * model_output - c_t * model_output_delta) / a_t

        return model_input, next_model_input.to(base.dtype)

    def denoise_step(
            self,
            base: torch.Tensor,
            model_input: torch.Tensor,
            model_output: torch.Tensor,
            timestep: torch.Tensor,
    ):
        prev_timestep = timestep - self.num_train_timesteps / self.num_inference_steps

        alpha_prod_t, beta_prod_t = self.get_alpha_and_beta(timestep)
        alpha_prod_t_prev, beta_prod_t_prev = self.get_alpha_and_beta(prev_timestep)

        a_t = (alpha_prod_t_prev / alpha_prod_t) ** 0.5
        b_t = -a_t * (beta_prod_t ** 0.5) + beta_prod_t_prev ** 0.5

        next_model_input = a_t * base + b_t * model_output

        return model_input, next_model_input.to(base.dtype)

    def denoise_step_with_ext(
            self,
            base: torch.Tensor,
            model_input: torch.Tensor,
            model_output: torch.Tensor,
            timestep: torch.Tensor,
            model_output_ext: torch.Tensor,
    ):
        prev_timestep = timestep - self.num_train_timesteps / self.num_inference_steps

        alpha_prod_t, beta_prod_t = self.get_alpha_and_beta(timestep)
        alpha_prod_t_prev, beta_prod_t_prev = self.get_alpha_and_beta(prev_timestep)

        a_t = (alpha_prod_t_prev / alpha_prod_t) ** 0.5
        # b_t = -a_t * (beta_prod_t ** 0.5)
        # c_t = beta_prod_t_prev ** 0.5
        c_t = -a_t * (beta_prod_t ** 0.5)
        b_t = c_t + beta_prod_t_prev ** 0.5

        model_output_delta = self.ext_scale * (model_output_ext - model_output)
        next_model_input = a_t * base + b_t * model_output + c_t * model_output_delta

        # next_model_input = a_t * base + b_t * (model_output * (1-self.ext_scale) + self.ext_scale * model_output_ext) + c_t * model_output

        return model_input, next_model_input.to(base.dtype)