def strategy_factory(strategy_type):
    if strategy_type == "incremental":
        return incremental
    elif strategy_type == "decremental":
        return decremental
    elif strategy_type == "dense":
        return dense
    elif strategy_type == "dense2":
        return dense2
    elif strategy_type == "void":
        return void
    else:
        raise NotImplementedError


def void(step, max_steps):
    return False, None


def dense(step, max_steps):
    return True, 1


def dense2(step, max_steps, interval=5):
    if step % interval == 0:
        return True, interval
    else:
        return False, interval


def incremental(step, max_steps, min_interval=1, max_interval=50, n_increments=5, startup_ratio=.25):
    startup_step = int(max_steps * startup_ratio)
    step_size = (max_steps - startup_step) // n_increments
    increment_size = int((max_interval - min_interval + 1) // n_increments * (max(1, step - startup_step) // step_size + 1))
    
    if step // step_size == 0 or step < startup_step:
        return True, 1
    else:
        if step % increment_size == 0:
            return True, increment_size
        else:
            return False, increment_size


def decremental(step, max_steps, min_interval=1, max_interval=50, n_increments=5):
    # Dense stage in last 25% (symmetric to incremental's startup_ratio=0.25)
    dense_start = int(max_steps * 0.75)

    if step >= dense_start:
        return True, 1  # Dense sampling in final 25%

    # Sparse→dense logic for first 75% with n_increments phases
    step_size = dense_start // n_increments  # Divide only the first 75%
    curr_increment = (n_increments - (step // step_size) - 1)
    increment_size = int((max_interval - min_interval + 1) // n_increments * curr_increment)

    if curr_increment == 0:
        return True, 1
    else:
        if step % increment_size == 0:
            return True, increment_size
        else:
            return False, increment_size

