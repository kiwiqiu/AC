import torch
import torch.nn as nn
import torch.nn.functional as F

from macls.models.utils import TDNNBlock, Conv1d, length_to_mask


class TemporalAveragePooling(nn.Module):
    def __init__(self):
        """TAP
        Paper: Multi-Task Learning with High-Order Statistics for X-vector based Text-Independent Speaker Verification
        Link: https://arxiv.org/pdf/1903.12058.pdf
        """
        super(TemporalAveragePooling, self).__init__()

    def forward(self, x):
        """Computes Temporal Average Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, channels)
        """
        x = x.mean(dim=-1) # x:(1,5120,13)-->(1,5120)
        # To be compatable with 2D input
        x = x.flatten(start_dim=1) # x:(1,5120)
        return x


class TemporalStatisticsPooling(nn.Module):
    def __init__(self):
        """TSP
        Paper: X-vectors: Robust DNN Embeddings for Speaker Recognition
        Link： http://www.danielpovey.com/files/2018_icassp_xvectors.pdf
        """
        super(TemporalStatisticsPooling, self).__init__()

    def forward(self, x):
        """Computes Temporal Statistics Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, channels*2)
        """
        mean = torch.mean(x, dim=2) #mean:(1,5120)
        var = torch.var(x, dim=2) #var:(1,5120)
        x = torch.cat((mean, var), dim=1) # x:(1,10240)
        return x


class SelfAttentivePooling(nn.Module):
    """SAP
    https://danielpovey.com/files/2018_interspeech_xvector_attention.pdf
    """

    def __init__(self, in_dim, bottleneck_dim=128):
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        # attention dim = 128
        super(SelfAttentivePooling, self).__init__()
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)  # equals V and k in the paper

    def forward(self, x):
        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x)) #(1,128,13)
        alpha = torch.softmax(self.linear2(alpha), dim=2) # (1,5120,13)
        mean = torch.sum(alpha * x, dim=2) # (1,5120)
        return mean


class AttentiveStatisticsPooling(nn.Module):
    """ASP
    This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.
    """

    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()

        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = Conv1d(in_channels=attention_channels, out_channels=channels, kernel_size=1)

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).
        """
        L = x.shape[-1] # x：(1,5120,13)

        def _compute_statistics(x, m, dim=2, eps=self.eps):
            mean = (m * x).sum(dim)
            std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps))
            return mean, std

        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device)

        # Make binary mask of shape [N, 1, L]
        mask = length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        if self.global_context:
            # torch.std is unstable for backward computation
            # https://github.com/pytorch/pytorch/issues/4320
            total = mask.sum(dim=2, keepdim=True).float()
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1) # x:(1,5120*3,13)
        else:
            attn = x

        # Apply layers
        attn = self.conv(self.tanh(self.tdnn(attn)))

        # Filter out zero-paddings
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=2)
        mean, std = _compute_statistics(x, attn)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)

        return pooled_stats


class TemporalStatsPool(nn.Module):
    """TSTP
    Temporal statistics pooling, concatenate mean and std, which is used in
    x-vector
    Comment: simple concatenation can not make full use of both statistics
    """

    def __init__(self):
        super(TemporalStatsPool, self).__init__()

    def forward(self, x):
        # The last dimension is the temporal axis
        pooling_mean = x.mean(dim=-1)
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-8)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        pooling_std = pooling_std.flatten(start_dim=1)

        stats = torch.cat((pooling_mean, pooling_std), 1)
        return stats


class MQMHASP(torch.nn.Module):
    """
     Reference:
        Miao Zhao, Yufeng Ma, and Yiwei Ding et al. "Multi-query multi-head attention pooling and Inter-topK penalty for speaker verification".
        https://arxiv.org/pdf/2110.05042.pdf
    """

    def __init__(self, in_dim,
                 num_q=1,
                 num_head=4,
                 hidden_size=128,
                 stddev=True,
                 share=True,
                 affine_layers=2,
                 time_attention=False,
                 norm_type='batch_norm',
                 **kargs
                 ):
        super(MQMHASP, self).__init__()
        self.stddev = stddev
        # self.output_dim = in_dim*2 if self.stddev else in_dim
        self.num_head = max(1, num_head)
        self.num_q = max(1, num_q)
        self.time_attention = time_attention
        assert (in_dim % num_head) == 0
        att_idim = in_dim // num_head
        if time_attention:
            att_idim = (in_dim * 3) // num_head if stddev else (in_dim * 2) // num_head
        att_odim = 1 if share else in_dim // num_head
        self.attention = self.build_attention(att_idim * num_head, att_odim * num_head * num_q, num_q, num_head,
                                              affine_layers, hidden_size, norm_type=norm_type)
        self.out_dim = in_dim * num_q * 2 if stddev else in_dim * num_q

    def forward(self, x, mask: torch.Tensor = torch.ones((0, 0, 0))):
        """
            x: input feature [B, F, T]
            returns: pooling statiscs [B, F * qs, 1]
        """

        def compute_statistics(x, m, dim: int = 2, stddev: bool = True, eps: float = 1e-5):
            mean = (m * x).sum(dim)

            if stddev:
                # std = torch.sqrt(
                #     (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps)
                # )
                std = torch.sqrt(
                    (torch.sum(m * (x ** 2), dim=dim) - mean ** 2).clamp(eps)
                )
            else:
                std = torch.empty(0)
            return mean, std

        B, C, T = x.shape

        if mask.size(2) == 0:
            mask = torch.ones((B, 1, T)).to(x.device)

        if self.time_attention:

            total = mask.sum(dim=2, keepdim=True)  # [B, *, 1]
            mean, std = compute_statistics(x, mask / total, stddev=self.stddev)
            mean = (mean.repeat(1, 1, T)).view(B, self.num_head, -1, T)
            x_in = x.view(B, self.num_head, -1, T)
            if self.stddev:
                std = (std.repeat(1, 1, T)).view(B, self.num_head, -1, T)
                x_in = torch.cat([x_in, mean, std], dim=2)
            else:
                x_in = torch.cat([x_in, mean], dim=2)
            x_in = x_in.reshape(B, -1, T)
        else:
            x_in = x
        alpha = self.attention(x_in)  # [B, head * att_dim, T]

        alpha = alpha.masked_fill(mask == 0, float("-inf"))

        alpha = F.softmax(alpha, dim=2)
        alpha = alpha.reshape(B, self.num_head, self.num_q, -1, T)

        mean, std = compute_statistics(x.reshape(B, self.num_head, 1, -1, T), alpha,
                                       stddev=self.stddev)  # mean: [B, head, q, C/head, 1]
        mean = mean.reshape(B, -1, 1)
        if self.stddev:
            std = std.reshape(B, -1, 1)
            out = torch.cat([mean, std], dim=1)
        else:
            out = mean
        # return out
        return out.squeeze(dim=-1)

    def get_output_dim(self):
        return self.out_dim

    def build_attention(self,
                        idim,
                        odim,
                        num_q,
                        num_head,
                        affine_layers=1,
                        hidden_size=128,
                        norm_type='batch_norm',
                        ):
        assert affine_layers in [1, 2], "Expected 1 or 2 affine layers."
        assert (idim % num_head) == 0
        assert (odim % (num_head * num_q)) == 0

        if affine_layers == 2:
            if norm_type == 'batch_norm':
                norm = torch.nn.BatchNorm1d(hidden_size * num_head * num_q)
            elif norm_type == 'layer_norm':
                norm = torch.nn.GroupNorm(num_head * num_q, hidden_size * num_head * num_q)
            else:
                raise ValueError("Unsupport norm type:{}".format(norm_type))
            att = torch.nn.Sequential(
                torch.nn.Conv1d(idim, hidden_size * num_head * num_q, kernel_size=1, groups=num_head),
                torch.nn.ReLU(),
                norm,
                torch.nn.Tanh(),
                torch.nn.Conv1d(hidden_size * num_head * num_q, odim, kernel_size=1, groups=num_head * num_q)
            )
        elif affine_layers == 1:
            att = torch.nn.Sequential(
                torch.nn.Conv1d(idim, odim, kernel_size=1, groups=num_head),
            )
        else:
            raise ValueError("Expected 1 or 2 affine layers, but got {}.".format(affine_layers))
        return att

    def extra_repr(self):
        return '(stddev={stddev}, num_head={num_head}, num_q={num_q}, out_dim={out_dim}) '.format(**self.__dict__)


class LDEPooling(torch.nn.Module):
    """A novel learnable dictionary encoding layer.
    Reference: Weicheng Cai, etc., "A NOVEL LEARNABLE DICTIONARY ENCODING LAYER FOR END-TO-END
               LANGUAGE IDENTIFICATION", icassp, 2018
    """
    def __init__(self, input_dim, c_num=1, eps=1.0e-10):
        super(LDEPooling, self).__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim * c_num
        self.eps = eps

        self.mu = torch.nn.Parameter(torch.randn(input_dim, c_num))
        self.s = torch.nn.Parameter(torch.ones(c_num))

        self.softmax_for_w = torch.nn.Softmax(dim=3)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        r = inputs.transpose(1,2).unsqueeze(3) - self.mu
        # Make sure beta=self.s**2+self.eps > 0
        w = self.softmax_for_w(- (self.s**2 + self.eps) * torch.sum(r**2, dim=2, keepdim=True))
        e = torch.mean(w * r, dim=1)

        # return e.reshape(-1, self.output_dim, 1)
        return e.flatten(1)  # 展平为 [1, 5120*c_num]
    def get_output_dim(self):
        return self.output_dim

# Attention-based
class TdnnAffine(torch.nn.Module):
    """ An implemented tdnn affine component by conv1d
        y = splice(w * x, context) + b

    @input_dim: number of dims of frame <=> inputs channels of conv
    @output_dim: number of layer nodes <=> outputs channels of conv
    @context: a list of context
        e.g.  [-2,0,2]
    If context is [0], then the TdnnAffine is equal to linear layer.
    """

    def __init__(self, input_dim, output_dim, context=[0], bias=True, pad=True, stride=1, groups=1, norm_w=False,
                 norm_f=False):
        super(TdnnAffine, self).__init__()
        assert input_dim % groups == 0
        # Check to make sure the context sorted and has no duplicated values
        for index in range(0, len(context) - 1):
            if (context[index] >= context[index + 1]):
                raise ValueError("Context tuple {} is invalid, such as the order.".format(context))

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context = context
        self.bool_bias = bias
        self.pad = pad
        self.groups = groups

        self.norm_w = norm_w
        self.norm_f = norm_f

        # It is used to subsample frames with this factor
        self.stride = stride

        self.left_context = context[0] if context[0] < 0 else 0
        self.right_context = context[-1] if context[-1] > 0 else 0

        self.tot_context = self.right_context - self.left_context + 1

        # Do not support sphereConv now.
        if self.tot_context > 1 and self.norm_f:
            self.norm_f = False
            print("Warning: do not support sphereConv now and set norm_f=False.")

        kernel_size = (self.tot_context,)

        self.weight = torch.nn.Parameter(torch.randn(output_dim, input_dim // groups, *kernel_size))

        # For jit compiling.
        # 2021-07-08 Leo
        self.bias = torch.nn.Parameter(torch.randn(output_dim)) if self.bool_bias else None

        # if self.bool_bias:
        #     self.bias = torch.nn.Parameter(torch.randn(output_dim))
        # else:
        # self.register_parameter('bias', None)

        # init weight and bias. It is important
        self.init_weight()

        # Save GPU memory for no skiping case
        if len(context) != self.tot_context:
            # Used to skip some frames index according to context
            self.mask = torch.tensor([[[1 if index in context else 0 \
                                        for index in range(self.left_context, self.right_context + 1)]]])
        else:
            self.mask = None

        ## Deprecated: the broadcast method could be used to save GPU memory,
        # self.mask = torch.randn(output_dim, input_dim, 0)
        # for index in range(self.left_context, self.right_context + 1):
        #     if index in context:
        #         fixed_value = torch.ones(output_dim, input_dim, 1)
        #     else:
        #         fixed_value = torch.zeros(output_dim, input_dim, 1)

        #     self.mask=torch.cat((self.mask, fixed_value), dim = 2)

        # Save GPU memory of thi case.

        self.selected_device = False

    def init_weight(self):
        # Note, var should be small to avoid slow-shrinking
        torch.nn.init.normal_(self.weight, 0., 0.01)

        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3

        assert inputs.shape[1] == self.input_dim

        # Do not use conv1d.padding for self.left_context + self.right_context != 0 case.
        if self.pad:
            inputs = F.pad(inputs, (-self.left_context, self.right_context), mode="constant", value=0.0)

        assert inputs.shape[2] >= self.tot_context

        # if not self.selected_device and self.mask is not None:
        #     # To save the CPU -> GPU moving time
        #     # Another simple case, for a temporary tensor, jus specify the device when creating it.
        #     self.mask = to_device(self, self.mask)

        #
        #     self.selected_device = True
        #    filters = self.weight  * self.mask if self.mask is not None else self.weight

        # For jit compiling.
        # 2021-07-08 Leo
        if self.mask is not None:

            self.mask = self.mask.to(inputs.device)
            filters = self.weight * self.mask
        else:
            filters = self.weight

        if self.norm_w:
            filters = F.normalize(filters, dim=1)

        if self.norm_f:
            inputs = F.normalize(inputs, dim=1)

        outputs = F.conv1d(inputs, filters, self.bias, stride=self.stride, padding=0, dilation=1, groups=self.groups)

        return outputs

    def extra_repr(self):
        return '{input_dim}, {output_dim}, context={context}, bias={bool_bias}, stride={stride}, ' \
               'pad={pad}, groups={groups}, norm_w={norm_w}, norm_f={norm_f}'.format(**self.__dict__)

    @classmethod
    def thop_count(self, m, x, y):
        x = x[0]

        kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
        bias_ops = 1 if m.bias is not None else 0

        # N x Cout x H x W x  (Cin x Kw x Kh + bias)
        total_ops = y.nelement() * (m.input_dim * kernel_ops + bias_ops)

        m.total_ops += torch.DoubleTensor([int(total_ops)])
class AttentionAlphaComponent(torch.nn.Module):
    """Compute the alpha with attention module.
            alpha = softmax(v'·f(w·x + b) + k) or softmax(v'·x + k)
    where f is relu here and bias could be lost.
    Support:
            1. Single or Multi-head attention
            2. One affine or two affine
            3. Share weight (last affine = vector) or un-shared weight (last affine = matrix)
            4. Self-attention or time context attention (supported by context parameter of TdnnAffine)
            5. Different temperatures for different heads.
    """
    def __init__(self, input_dim, num_head=1, split_input=True, share=True, affine_layers=2,
                 hidden_size=64, context=[0], bias=True, temperature=False, fixed=True):
        super(AttentionAlphaComponent, self).__init__()
        assert num_head >= 1
        # Multi-head case.
        if num_head > 1:
            if split_input:
                # Make sure fatures/planes with input_dim dims could be splited to num_head parts.
                print("input_dim:",input_dim)
                assert input_dim % num_head == 0
            if temperature:
                if fixed:
                    t_list = []
                    for i in range(num_head):
                        t_list.append([[max(1, (i // 2) * 5)]])
                    # shape [1, num_head, 1, 1]
                    self.register_buffer('t', torch.tensor([t_list]))
                else:
                    # Different heads have different temperature.
                    # Use 1 + self.t**2 in forward to make sure temperature >= 1.
                    self.t = torch.nn.Parameter(torch.zeros(1, num_head, 1, 1))

        self.input_dim = input_dim
        self.num_head = num_head
        self.split_input = split_input
        self.share = share
        self.temperature = temperature
        self.fixed = fixed

        if share:
            # weight: [input_dim, 1] or [input_dim, hidden_size] -> [hidden_size, 1]
            final_dim = 1
        elif split_input:
            # weight: [input_dim, input_dim // num_head] or [input_dim, hidden_size] -> [hidden_size, input_dim // num_head]
            final_dim = input_dim // num_head
        else:
            # weight: [input_dim, input_dim] or [input_dim, hidden_size] -> [hidden_size, input_dim]
            final_dim = input_dim

        first_groups = 1
        last_groups = 1

        if affine_layers == 1:
            last_affine_input_dim = input_dim
            # (x, 1) for global case and (x, h) for split case.
            if num_head > 1 and split_input:
               last_groups = num_head
            self.relu_affine = False
        elif affine_layers == 2:
            last_affine_input_dim = hidden_size * num_head
            if num_head > 1:
                # (1, h) for global case and (h, h) for split case.
                last_groups = num_head
                if split_input:
                    first_groups = num_head
            # Add a relu-affine with affine_layers=2.
            self.relu_affine = True
            self.first_affine = TdnnAffine(input_dim, last_affine_input_dim, context=context, bias=bias, groups=first_groups)
            self.relu = torch.nn.ReLU(inplace=True)
        else:
            raise ValueError("Expected 1 or 2 affine layers, but got {}.",format(affine_layers))

        self.last_affine = TdnnAffine(last_affine_input_dim, final_dim * num_head, context=context, bias=bias, groups=last_groups)
        # Dim=2 means to apply softmax in different frames-index (batch is a 3-dim tensor in this case).
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        if self.temperature:
            batch_size = inputs.shape[0]
            chunk_size = inputs.shape[2]

        x = inputs
        if self.relu_affine:
            x = self.relu(self.first_affine(x))
        if self.num_head > 1 and self.temperature:
            if self.fixed:
                t = self.t
            else:
                t = 1 + self.t**2
            x = self.last_affine(x).reshape(batch_size, self.num_head, -1, chunk_size) / t
            return self.softmax(x.reshape(batch_size, -1, chunk_size))
        else:
            return self.softmax(self.last_affine(x))
class MultiHeadAttentionPooling(torch.nn.Module):
    """Implement multi-head attention pooling based on AttentionAlphaComponent.
    Reference: Safari, Pooyan, and Javier Hernando. 2019. “Self Multi-Head Attention for Speaker
               Recognition.” ArXiv Preprint ArXiv:1906.09890.
    Note, in this paper, affine_layers is default to 1, and final_dim is 1 which means the weights are shared.
    """
    def __init__(self, input_dim, stddev=True, stddev_attention=True, num_head=4, share=True, affine_layers=1, **options):
        super(MultiHeadAttentionPooling, self).__init__()

        self.input_dim = input_dim
        self.stddev = stddev
        self.stddev_attention = stddev_attention
        self.num_head = num_head

        if self.stddev :
            self.output_dim = 2 * input_dim
        else :
            self.output_dim = input_dim

        if "split_input" in options.keys():
            if not options["split_input"]:
                raise ValueError("split_input==False is not valid for this MultiHeadAttentionPooling.")
            options.pop("split_input")

        # In this pooling, the special point is that inputs will be splited.
        self.attention = AttentionAlphaComponent(input_dim, num_head=num_head, split_input=True, share=share,
                                                 affine_layers=affine_layers, bias=False, **options)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        batch_size = inputs.shape[0]
        chunk_size = inputs.shape[2] # a.k.a total frames

        # alpha: [batch, weight, frames]
        # When using the conv1d to implement the multi-multiple of multi-head, we can get
        # the weight distribution of multi-head: [h11, h12, h13, h21, h22, h23, ..., hn1, hn2, ...]
        # So, just reshape it to split different heads.
        alpha = self.attention(inputs)

        # In sharing weight case, the shape of alpha is [batch, head, 1, frames] and [batch, head, splited-features, frames]
        # for another case.
        # inputs: [batch, head, splited-features, frames]
        after_mul = alpha.reshape(batch_size, self.num_head, -1, chunk_size) * \
                    inputs.reshape(batch_size, self.num_head, -1, chunk_size)

        # After multi-multipling alpha and inputs for multi-head case, the mean could be got by reshaping back.
        mean = torch.sum(after_mul.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True)

        if self.stddev :
            if self.stddev_attention:
                after_mul_2 = alpha.reshape(batch_size, self.num_head, -1, chunk_size) * \
                        inputs.reshape(batch_size, self.num_head, -1, chunk_size)**2
                var = torch.sum(after_mul_2.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True) - mean**2
                std = torch.sqrt(var.clamp(min=1.0e-10))
            else:
                var = torch.mean((inputs - mean)**2, dim=2, keepdim=True)
                std = torch.sqrt(var.clamp(min=1.0e-10))
            # return torch.cat((mean, std), dim=1)
            return torch.cat((mean, std), dim=1).squeeze(-1)
        else :
            # return mean
            return mean.squeeze(-1)

    def get_output_dim(self):
        return self.output_dim


class GlobalMultiHeadAttentionPooling(torch.nn.Module):
    """Implement global multi-head attention pooling based on AttentionAlphaComponent.
    Reference: Zhiming Wang, Kaisheng Yao, Xiaolong Li, Shuo Fang. "MULTI-RESOLUTION MULTI-HEAD
               ATTENTION IN DEEP SPEAKER EMBEDDING." ICASSP, 2020.
    It is not equivalent to multi-head attention pooling even when
               input_dim of global multi-head = 1/num_head * input_dim of multi-head.
    """
    def __init__(self, input_dim, stddev=True, stddev_attention=True, num_head=4, share=True, affine_layers=2, **options):
        super(GlobalMultiHeadAttentionPooling, self).__init__()

        self.input_dim = input_dim
        self.num_head = num_head
        self.stddev = stddev
        self.stddev_attention = stddev_attention

        if self.stddev :
            self.output_dim = 2 * input_dim
        else :
            self.output_dim = input_dim

        if "split_input" in options.keys():
            if options["split_input"]:
                raise ValueError("split_input==True is not valid for GlobalMultiHeadAttentionPooling.")
            options.pop("split_input")
        if "temperature" in options.keys():
            if options["temperature"]:
                raise ValueError("temperature==True is not valid for GlobalMultiHeadAttentionPooling.")
            options.pop("temperature")

        # In this pooling, the special point is that all (global) features of inputs will be used.
        self.attention = AttentionAlphaComponent(input_dim, num_head=num_head, split_input=False, share=share,
                                                 temperature=False, affine_layers=affine_layers, bias=True, **options)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        batch_size = inputs.shape[0]
        chunk_size = inputs.shape[2] # a.k.a total frames

        # alpha: [batch, weight, frames]
        # When using the conv1d to implement the multi-multiple of multi-head, we can get
        # the weight distribution of multi-head: [h11, h12, h13, h21, h22, h23, ..., hn1, hn2, ...]
        # So, just reshape it to split different heads.
        alpha = self.attention(inputs)

        # In sharing weight case, the shape of alpha is [batch, head, 1, frames] and [batch, head, all-features, frames]
        # for another case.
        # inputs: [batch, 1, all-features, frames]
        after_mul = alpha.reshape(batch_size, self.num_head, -1, chunk_size) * \
                    inputs.reshape(batch_size, 1, -1, chunk_size)

        # After multi-multipling alpha and inputs for multi-head case, the mean could be got by reshaping back.
        mean = torch.sum(after_mul.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True)

        if self.stddev :
            if self.stddev_attention:
                after_mul_2 = alpha.reshape(batch_size, self.num_head, -1, chunk_size) * \
                        inputs.reshape(batch_size, 1, -1, chunk_size)**2
                var = torch.sum(after_mul_2.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True) - mean**2
                std = torch.sqrt(var.clamp(min=1.0e-10))
            else:
                var = torch.mean((inputs - mean)**2, dim=2, keepdim=True)
                std = torch.sqrt(var.clamp(min=1.0e-10))
            out = torch.cat((mean, std), dim=1)
        else :
            out = mean
        return out.squeeze(-1)

    def get_output_dim(self):
        return self.output_dim * self.num_head