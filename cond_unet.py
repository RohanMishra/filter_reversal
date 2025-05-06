import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) / (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
class ConditionalUnet(nn.Module):
    def __init__(
        self,
        dim,
        cond_channels=3,         # <-- # of channels in the filter image
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,              # <-- # of channels in the noisy input
        resnet_block_groups=4,
    ):
        super().__init__()

        # save
        self.channels       = channels
        self.cond_channels  = cond_channels

        # total input channels = noisy image + filtered image
        input_channels = channels + cond_channels

        # initial conv (was 7×7 in some U-Nets, but you moved it to 1×1)
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, kernel_size=1)

        # compute channel dims at each resolution
        dims   = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time‐embedding MLP
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Down‐sampling path
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        self.downs = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind == len(in_out) - 1
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last 
                  else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        # Middle (bottleneck)
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn   = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Up‐sampling path
        self.ups = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last 
                  else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        # final conv head
        self.out_dim        = default(out_dim, channels)
        self.final_res_block = block_klass(dims[0] + init_dim, dim, time_emb_dim=time_dim)
        self.final_conv      = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x_noisy, t, cond_img):
        """
        x_noisy:   (B, channels, H, W)  — the current noisy sample x_t
        t:         (B,)                 — integer timesteps
        cond_img:  (B, cond_channels, H, W) — the filtered image to condition on
        """
        # 1) concat conditioning
        x = torch.cat([x_noisy, cond_img], dim=1)   # now (B, channels+cond_channels, H, W)
        x = self.init_conv(x)
        residual = x.clone()

        # 2) time‐embed
        t_emb = self.time_mlp(t)

        # 3) down path
        skips = []
        for block1, block2, attn, down in self.downs:
            x = block1(x, t_emb);   skips.append(x)
            x = block2(x, t_emb);   x = attn(x);   skips.append(x)
            x = down(x)

        # 4) bottleneck
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # 5) up path
        for block1, block2, attn, up in self.ups:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block1(x, t_emb)
            x = torch.cat([x, skips.pop()], dim=1)
            x = block2(x, t_emb);  x = attn(x)
            x = up(x)

        # 6) final residual merge & head
        x = torch.cat([x, residual], dim=1)
        x = self.final_res_block(x, t_emb)
        return self.final_conv(x)  # preds of shape (B, channels, H, W)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class Diffusion:
    def __init__(self, timesteps: int = 300):
        self.timesteps = timesteps
        self.betas = self.linear_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    def linear_beta_schedule(self):
        return torch.linspace(1e-4, 0.02, self.timesteps)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_acp = torch.sqrt(self.alphas_cumprod)
        sqrt_om_acp = torch.sqrt(1.0 - self.alphas_cumprod)
        return extract(sqrt_acp, t, x_start.shape) * x_start + extract(sqrt_om_acp, t, x_start.shape) * noise

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, cond):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_om_acp_t = extract(torch.sqrt(1.0 - self.alphas_cumprod), t, x.shape)
        sqrt_recip_alphas_t = extract(torch.sqrt(1.0 / self.alphas), t, x.shape)

        eps_pred = model(x, t, cond)
        model_mean = sqrt_recip_alphas_t * (x - (betas_t / sqrt_om_acp_t) * eps_pred)

        if t_index == 0:
            return model_mean
        else:
            alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1,0), value=1.0)
            posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            var_t = extract(posterior_variance, t, x.shape)
            return model_mean + torch.sqrt(var_t) * torch.randn_like(x)

    @torch.no_grad()
    def p_sample_loop(self, model, cond, shape):
        device = next(model.parameters()).device
        b, c, h, w = shape
        img = torch.randn(shape, device=device)
        imgs = []
        for i in tqdm(reversed(range(self.timesteps)), desc='sampling loop', total=self.timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, i, cond)
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self, model, cond):
        b, _, h, w = cond.shape
        return self.p_sample_loop(model, cond, shape=(b, model.channels, h, w))