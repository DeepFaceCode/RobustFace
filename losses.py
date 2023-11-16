import torch
import math
from torch.nn.functional import linear, normalize

class ArcFace(torch.nn.Module):
    def __init__(self,
                 num_class,
                 embedding_size,
                 s,
                 m,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.num_class = num_class
        self.embedding_size = embedding_size
        self.s = s
        self.m = m
        self.interclass_filtering_threshold = interclass_filtering_threshold

        # For ArcFace
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.theta = math.cos(math.pi - self.m)
        self.sinmm = math.sin(math.pi - self.m) * self.m
        self.easy_margin = False
        self.weight_activated = torch.nn.Parameter(torch.normal(0, 0.01, (self.num_class, self.embedding_size)))
        self.fp16 = True


    def forward(self, logits, labels):
        index_positive = torch.where(labels != -1)[0]

        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(logits)
            norm_weight_activated = normalize(self.weight_activated)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)


        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)

        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)


        logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
        logits = logits * self.s
        easy_num = 0.0
        noise_num = 0.0
        hard_num = 0
        month_phi = 0.0

        return logits, easy_num, noise_num, hard_num, month_phi

class RobustFace(torch.nn.Module):
    def __init__(self,
                 num_class,
                 embedding_size,
                 s,
                 m, # cos(theta+m)
                 m1, # cos(tehta-m1)
                 t,
                 errsum,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.num_class = num_class
        self.embedding_size = embedding_size
        self.s = s
        self.m = m
        self.m1 = m1
        self.interclass_filtering_threshold = interclass_filtering_threshold

        # For m
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.theta = math.cos(math.pi - self.m)
        self.sinmm = math.sin(math.pi - self.m) * self.m

        # For m1
        self.cos_m_ = math.cos(self.m1)
        self.sin_m_ = math.sin(self.m1)
        self.theta_ = math.cos(- self.m1)
        self.sinmm_ = math.sin( - self.m1) * self.m1

        self.easy_margin = False
        self.weight_activated = torch.nn.Parameter(torch.normal(0, 0.01, (self.num_class, self.embedding_size)))
        self.fp16 = True
        self.t = t
        self.phi_smooth = 0
        self.clear_rio = 1.0 - errsum # 1-mu
    def forward(self, logits, labels):
        index_positive = torch.where(labels != -1)[0]

        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(logits)
            norm_weight_activated = normalize(self.weight_activated)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        self.m1 = (1 - self.phi_smooth / self.clear_rio) * self.m1
        self.cos_m_ = math.cos(self.m1)
        self.sin_m_ = math.sin(self.m1)
        self.theta_ = math.cos(- self.m1)
        self.sinmm_ = math.sin(- self.m1) * self.m1

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+m)
        cos_theta_m_ = target_logit * self.cos_m_ - sin_theta * self.sin_m_  # cos(target-m1)

        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

            final_target_logit_ = torch.where(
                target_logit <= self.cos_m_, cos_theta_m_, target_logit + self.sinmm_)
        mask_hard = logits > final_target_logit.unsqueeze(1)
        mask_noise = logits > final_target_logit_.unsqueeze(1)

        hard_example = logits[mask_hard] # hard+noise
        noise_example = logits[mask_noise]

        num_all = self.num_class * logits.shape[0]
        easy_num = num_all - hard_example.size(0)
        noise_num = noise_example.size(0)
        hard_num = hard_example.size(0) - noise_num

        # φ  exponential moving average(EMA)
        self.phi_smooth = torch.div(easy_num, num_all) * 0.01 + (1 - 0.01) * self.phi_smooth

        # hard: (1+t) * cos(θ)
        logits[mask_hard] = hard_example * (self.t + 1)
        gamma = 2

        # noise: (1-φ/clear_rio)^2 * cos(θ)
        logits[mask_noise] = (noise_example * (1 - self.phi_smooth/self.clear_rio) ** gamma).clamp_min_(1e-30)

        # target(theta+m)
        logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
        logits = logits * self.s

        return logits, easy_num, noise_num, hard_num, self.phi_smooth

class CurricularFace(torch.nn.Module):
    def __init__(self,
                 num_class,
                 embedding_size,
                 s=64.0,
                 margin=0.5):

        super(CurricularFace, self).__init__()
        self.scale = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False
        self.num_class = num_class
        self.embedding_size = embedding_size
        self.weight_activated = torch.nn.Parameter(torch.normal(0, 0.01, (self.num_class, self.embedding_size)))
        self.fp16 = True
        self.register_buffer('t', torch.zeros(1))
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index_positive = torch.where(labels != -1)[0]

        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(logits)
            norm_weight_activated = normalize(self.weight_activated)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)
        target_logit = logits[index_positive, labels[index_positive].view(-1)]
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)

        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)
        mask = logits > final_target_logit.unsqueeze(1)
        hard_example = logits[mask]
        easy_num = 10974976 - hard_example.size(0)
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        logits[mask] = hard_example * (self.t + hard_example)
        logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
        logits = logits * self.scale
        noise_num = 0.0
        hard_num = hard_example.size(0)
        month_phi = 0.0
        return logits, easy_num, noise_num, hard_num, month_phi

class BoundaryMargin(torch.nn.Module):
    def __init__(self,
                 num_class,
                 embedding_size,
                 s=64.0,
                 marigin=0.5,
                 easy_margin=False,
                 epoch_start=7
                 ):
        super(BoundaryMargin, self).__init__()
        self.num_class = num_class
        self.embedding_size = embedding_size
        self.s = s
        self.marigin = marigin
        self.fp16 = True

        self.weight_activated = torch.nn.Parameter(torch.normal(0, 0.01, (self.num_class, self.embedding_size)))

        self.easy_margin = easy_margin
        self.cos_m = math.cos(marigin)
        self.sin_m = math.sin(marigin)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - marigin)
        self.mm = math.sin(math.pi - marigin) * marigin

        self.epoch_start = epoch_start


    def forward(self, logits: torch.Tensor, labels: torch.Tensor, epoch):

        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(logits)
            norm_weight_activated = normalize(self.weight_activated)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1,1)

        # cos(theta + m)
        sine_theta = torch.sqrt(1.0 - torch.pow(logits, 2))
        cos_theta_m = logits * self.cos_m - sine_theta * self.sin_m

        if self.easy_margin:
            phi = torch.where(logits > 0, cos_theta_m, logits)
        else:
            phi = torch.where((logits - self.th) > 0, cos_theta_m, logits - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        if epoch > self.epoch_start:
            right2 = one_hot * logits.detach()
            left2 = (1.0 - one_hot) * phi.detach()
            left_max2, argmax2 = torch.max(left2.detach(), dim=1)
            max_index2 = argmax2.detach()
            right_max2, _ = torch.max(right2.detach(), dim=1)
            sub2 = left_max2 - right_max2
            zero2 = torch.zeros_like(sub2)
            temp2 = torch.where(sub2 > 0, sub2, zero2)
            # non_zero_index2 = torch.nonzero(temp2.detach(), as_tuple=True)
            #non_zero_index2 = torch.nonzero(temp2.detach())
            #rectified_label = torch.topk(one_hot, 1)[1].squeeze(1).cuda()
            right = one_hot * phi
            left = (1.0 - one_hot) * logits
            logits = left + right
            logits = logits * self.s

            left_max, _ = torch.max(left, dim=1)
            right_max, _ = torch.max(right, dim=1)
            sub = left_max - right_max
            zero = torch.zeros_like(sub)
            temp = torch.where(sub > 0, sub, zero)
            final = torch.mean(temp) * math.pi

        if epoch <= self.epoch_start:
            rectified_label = labels
            final = 0.0
            logits = (one_hot * phi) + ((1.0 - one_hot) * logits)
            logits = logits * self.s

        easy_num = 0.0
        noise_num = 0.0
        hard_num = 0
        month_phi = 0.0

        return logits, easy_num, noise_num, hard_num, month_phi

class AdaFace(torch.nn.Module):
    def __init__(self,
                 num_class,
                 embedding_size=512,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 ):
        super(AdaFace, self).__init__()
        self.embedding_size = embedding_size
        self.num_class = num_class
        self.weight_activated = torch.nn.Parameter(torch.normal(0, 0.01, (self.num_class, self.embedding_size)))


        self.m = m
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

        print('\n\AdaFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)

    def forward(self, embbedings, label, norms):

        kernel_norm = normalize(self.weight_activated)
        norm_embeddings = normalize(embbedings)
        cosine = linear(norm_embeddings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h
        margin_scaler = torch.clip(margin_scaler, -1, 1)

        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        logits = cosine * self.s
        easy_num = 0.0
        noise_num = 0.0
        hard_num = 0
        month_phi = 0.0

        return logits, easy_num, noise_num, hard_num, month_phi

