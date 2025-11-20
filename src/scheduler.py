import math


def mt_scheduler_factory(scheduler_type):
    if scheduler_type == "constant":
        return mt_constant
    elif scheduler_type == "linear":
        return mt_linear
    elif scheduler_type == "step":
        return mt_step
    else:
        raise NotImplementedError


def mt_constant(step, max_step, mt_ratio):
    return mt_ratio


def mt_linear(step, max_step, mt_ratio):
    return mt_ratio + (step/max_step) * (1.0 - mt_ratio)


def mt_step(step, max_step, mt_ratio, n_stages=10):
    interval = max_step / n_stages
    ratio_step = (1.0 - mt_ratio) / n_stages
    stage = step // interval
    new_ratio = min(mt_ratio + stage * ratio_step, 1.0)
    return new_ratio