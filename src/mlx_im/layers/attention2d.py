from typing import List, Optional, Union

import mlx.core as mx
from mlx import nn

from .config import use_fused_attn
from .create_conv2d import create_conv2d
from .helpers import to_2tuple
from .pool2d_same import create_pool2d


class MultiQueryAttentionV2(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        num_heads: int = 8,
        key_dim: int = 64,
        value_dim: int = 64,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        dim_out = dim_out or dim  # O
        self.num_heads = num_heads  # N
        self.key_dim = key_dim  # K
        self.value_dim = value_dim  # V
        self.scale = key_dim**-0.5  # NOTE: Seems wrong, not applied.

        self.query_proj = mx.random.normal((self.num_heads, self.key_dim, dim))
        self.key_proj = mx.random.normal((dim, self.key_dim))
        self.value_proj = mx.random.normal((dim, self.value_dim))
        self.out_proj = mx.random.normal((dim_out, self.num_heads, self.value_dim))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def __call__(self, x: mx.array, m: Optional[mx.array] = None) -> mx.array:
        b, h, w, c = x.shape  # (B, H, W, D)

        reshaped_x = x.reshape(b, -1, c)  # (B, H1*W1, D)

        # (B, H2*W2, D)
        if m is not None:
            reshaped_m = m.reshape(b, -1, m.shape[-1])
        else:
            reshaped_m = reshaped_x

        # (B, H1*W1, D) (N, K, D) -> (B, H1*W1, N, K)
        q = mx.einsum("bnd,hkd->bnhk", reshaped_x, self.query_proj)

        # (B, H2*W2, D) (D, K) -> (B, H2*W2, K)
        k = mx.einsum("bmd,dk->bmk", reshaped_m, self.key_proj)

        # NOTE: Timm implementation does not use the scale: doing the same here.
        # (B, H1*W1, N, K) (B, H2*W2, K) -> (B, H1*W1, N, H2*W2)
        attn = mx.einsum("bnhk,bmk->bnhm", q, k)

        # (B, H1*W1, N, H2*W2)
        attn = mx.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        # (B, H2*W2, D) (D, V) -> (B, H2*W2, V)
        v = mx.einsum("bmd,dv->bmv", reshaped_m, self.value_proj)

        # (B, H1*W1, N, H2*W2) (B, H2*W2, V) -> (B, H1*W1, N, V)
        o = mx.einsum("bnhm,bmv->bnhv", attn, v)

        # (B, H1*W1, N, V) (O, N, V) -> (B, H1*W1, O)
        result = mx.einsum("bnhv,dhv->bnd", o, self.out_proj)
        result = self.proj_drop(result)

        # (B, H1, W1, O)
        return result.reshape(b, h, w, -1)


class MultiQueryAttention2d(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        num_heads: int = 8,
        key_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        query_strides: int = 1,
        kv_stride: int = 1,
        dw_kernel_size: int = 3,
        dilation: int = 1,
        padding: Union[str, int, List[int]] = "",
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.BatchNorm,
        use_bias: bool = False,
    ):
        super().__init__()
        dim_out = dim_out or dim  # O
        self.num_heads = num_heads  # N
        self.key_dim = key_dim or dim // num_heads  # K
        self.value_dim = value_dim or dim // num_heads  # V
        self.query_strides = to_2tuple(query_strides)
        self.kv_stride = kv_stride
        self.has_query_strides = any([s > 1 for s in self.query_strides])
        self.scale = self.key_dim**-0.5
        self.fused_attn = use_fused_attn()
        self.drop = attn_drop

        query = []
        if self.has_query_strides:
            if padding == "same":
                query.append(
                    create_pool2d(
                        "avg",
                        kernel_size=self.query_strides,
                        padding="same",
                    ),
                )
            else:
                query.append(nn.AvgPool2d(kernel_size=query_strides))
            query.append(norm_layer(dim))
        query.append(
            create_conv2d(
                dim,
                self.num_heads * self.key_dim,
                kernel_size=1,
                bias=use_bias,
            ),
        )
        self.query = nn.Sequential(query)

        key = []
        if kv_stride > 1:
            key.append(
                create_conv2d(
                    dim,
                    dim,
                    kernel_size=dw_kernel_size,
                    stride=kv_stride,
                    dilation=dilation,
                    padding=padding,
                    depthwise=True,
                ),
            )
            key.append(norm_layer(dim))
        key.append(
            create_conv2d(
                dim,
                self.key_dim,
                kernel_size=1,
                padding=padding,
                bias=use_bias,
            ),
        )
        self.key = nn.Sequential(key)

        value = []
        if kv_stride > 1:
            value.append(
                create_conv2d(
                    dim,
                    dim,
                    kernel_size=dw_kernel_size,
                    stride=kv_stride,
                    dilation=dilation,
                    padding=padding,
                    depthwise=True,
                ),
            )
            value.append(norm_layer(dim))
        value.append(
            create_conv2d(
                dim,
                self.value_dim,
                kernel_size=1,
                bias=use_bias,
            ),
        )
        self.value = nn.Sequential(value)

        self.attn_drop = nn.Dropout(attn_drop)

        output = []
        if self.has_query_strides:
            output.append(
                nn.Upsample(
                    scale_factor=self.query_strides,
                    mode="bilinear",
                    align_corners=False,
                ),
            )
        output.append(
            create_conv2d(
                self.value_dim * self.num_heads,
                dim_out,
                kernel_size=1,
                bias=use_bias,
            ),
        )
        output.append(nn.Dropout(proj_drop))
        self.output = nn.Sequential(output)

        self.einsum = False

    def _reshape_input(self, t: mx.array) -> mx.array:
        s = t.shape
        t = t.reshape(s[0], -1, s[1])  # (B, H*W, C)
        if self.einsum:
            return t
        else:
            return t[:, None, ...]  # (B, 1, H*W, C)

    def _reshape_projected_query(self, t: mx.array, num_heads: int, key_dim: int):
        """Reshapes projected query: [b, n, n, h x k] -> [b, n x n, h, k]."""
        s = t.shape
        t = t.reshape(s[0], num_heads, key_dim, -1)
        if self.einsum:
            return t.permute(0, 3, 1, 2).contiguous()
        else:
            return t.transpose(-1, -2).contiguous()

    def _reshape_output(self, t: mx.array, num_heads: int, h_px: int, w_px: int):
        """Reshape output:[b, n x n x h, k] -> [b, n, n, hk]."""
        s = t.shape
        feat_dim = s[-1] * num_heads
        if not self.einsum:
            t = t.transpose(1, 2)
        return t.reshape(s[0], h_px, w_px, feat_dim).permute(0, 3, 1, 2).contiguous()

    def __call__(self, x: mx.array, attn_mask: Optional[mx.array] = None) -> mx.array:
        """Run layer computation."""
        _, _, H, W = x.shape

        q = self.query(x)
        # desired q shape: [b, h, k, n x n] - [b, l, h, k]
        q = self._reshape_projected_query(q, self.num_heads, self.key_dim)

        k = self.key(x)
        # output shape of k: [b, k, p], p = m x m
        k = self._reshape_input(k)

        v = self.value(x)
        # output shape of v: [ b, p, k], p = m x m
        v = self._reshape_input(v)

        # desired q shape: [b, n x n, h, k]
        # desired k shape: [b, m x m, k]
        # desired logits shape: [b, n x n, h, m x m]
        if self.einsum:
            attn = mx.einsum("blhk,bpk->blhp", q, k) * self.scale
            if attn_mask is not None:
                # NOTE: assumes mask is float and in correct shape
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            o = mx.einsum("blhp,bpk->blhk", attn, v)
        else:
            if self.fused_attn:
                o = scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
            else:
                q = q * self.scale
                attn = q @ k.transpose(-1, -2)
                if attn_mask is not None:
                    # NOTE: assumes mask is float and in correct shape
                    attn = attn + attn_mask
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                o = attn @ v

        # reshape o into [b, hk, n, n,]
        o = self._reshape_output(
            o, self.num_heads, H // self.query_strides[0], W // self.query_strides[1]
        )
        x = self.output(o)
        return x


class Attention2d(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        num_heads: int = 32,
        bias: bool = True,
        expand_first: bool = False,
        head_first: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        assert attn_drop == 0.0, "Attention dropout not supported"
        super().__init__()
        dim_out = dim_out or dim
        dim_attn = dim_out if expand_first else dim  # E
        self.num_heads = num_heads  # N
        self.dim_head = dim_attn // num_heads  # E / N
        self.head_first = head_first
        self.scale = self.dim_head**-0.5  # NOTE: The scale defined in timm is wrong.
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Conv2d(dim, dim_attn * 3, 1, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim_attn, dim_out, 1, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def __call__(self, x: mx.array, attn_mask: Optional[mx.array] = None) -> mx.array:
        B, H, W, _ = x.shape

        if self.head_first:
            q, k, v = (
                self.qkv(x)  # (B, H, W, 3*E)
                .reshape(B, -1, self.num_heads, self.dim_head * 3)  # (B, H*W, N, 3*E/N)
                .split(3, axis=-1)  # 3 * (B, H*W, N, E/N)
            )
        else:
            q, k, v = (
                self.qkv(x)  # (B, H, W, 3*E)
                .reshape(B, -1, 3, self.num_heads, self.dim_head)  # (B, H*W, N, 3*E/N)
                .split(3, axis=2)  # 3 * (B, H*W, N, E/N)
            )

            q, k, v = [o.squeeze(2) for o in [q, k, v]]

        # (B, N, H*W, E/N)
        x = mx.fast.scaled_dot_product_attention(
            q=q.transpose(0, 2, 1, 3),
            k=k.transpose(0, 2, 1, 3),
            v=v.transpose(0, 2, 1, 3),
            scale=self.scale,
            mask=attn_mask,
        )

        x = x.transpose(0, 2, 1, 3).reshape(B, H, W, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
