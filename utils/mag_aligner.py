# ===== utils/mag_aligner.py =====
import torch

class EMAMeter:
    def __init__(self, beta=0.98):
        self.beta = beta
        self.v = None

    def update(self, x: float):
        if self.v is None:
            self.v = float(x)
        else:
            self.v = self.beta * self.v + (1.0 - self.beta) * float(x)

    @property
    def value(self):
        return 0.0 if self.v is None else float(self.v)


class MagAligner:
    """
    按量级对齐：
      ddcl_w   = target_ratio_ddcl * E[loss_s] / E[ddcl]
      smooth_w = target_ratio_smooth * E[loss_s] / E[smooth]
    再做 warmup + clamp，确保平稳。
    """
    def __init__(
        self,
        beta=0.98,
        warmup_steps=500,
        target_ratio_ddcl=1.0,      # 希望 ddcl 贡献 ~ loss_s
        target_ratio_smooth=0.2,    # 希望 smooth 贡献 ~ 20% 的 loss_s
        clamp_ddcl=(0.005, 0.05),   # 经验范围，对应你之前的量级（0.02 在这个区间内）
        clamp_smooth=(0.5, 5.0),    # 经验范围，2.0 在这个区间内
        eps=1e-6
    ):
        self.s = EMAMeter(beta)
        self.ddcl = EMAMeter(beta)
        self.sm = EMAMeter(beta)
        self.warmup_steps = warmup_steps
        self.target_ratio_ddcl = target_ratio_ddcl
        self.target_ratio_smooth = target_ratio_smooth
        self.clamp_ddcl = clamp_ddcl
        self.clamp_smooth = clamp_smooth
        self.eps = eps
        self.step = 0

    def update_and_get_weights(self, loss_s, ddcl, smooth):
        # 更新 EMA
        self.s.update(loss_s)
        if ddcl is not None:
            self.ddcl.update(ddcl)
        if smooth is not None:
            self.sm.update(smooth)

        # 基于 EMA 做“量级对齐”
        w_ddcl = self.target_ratio_ddcl * (self.s.value / max(self.ddcl.value, self.eps))
        w_sm   = self.target_ratio_smooth * (self.s.value / max(self.sm.value,   self.eps))

        # clamp 避免抖动太大
        w_ddcl = float(max(self.clamp_ddcl[0], min(self.clamp_ddcl[1], w_ddcl)))
        w_sm   = float(max(self.clamp_smooth[0], min(self.clamp_smooth[1], w_sm)))

        # warmup：前 warmup_steps 步线性从 0->目标
        ramp = min(1.0, (self.step + 1) / float(self.warmup_steps))
        w_ddcl *= ramp
        # smooth 通常不需要从 0 爬升，但如需更稳也可乘 ramp：
        # w_sm *= ramp

        self.step += 1
        return w_ddcl, w_sm
