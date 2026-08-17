import time
from collections import deque


def apply_api_rate_limit(request_deque: deque, max_requests: int = 10, window_seconds: int = 30) -> None:
    """
    简单的滑动窗口限流（令牌桶简化版）
    :param request_deque: 记录请求时间戳的双端队列
    :param max_requests: 时间窗口内最大请求数
    :param window_seconds: 时间窗口大小（秒）
    """
    now = time.time()

    # 移除窗口外的旧时间戳
    while request_deque and now - request_deque[0] > window_seconds:
        request_deque.popleft()

    # 如果窗口内请求已满，则等待直到最早请求过期
    if len(request_deque) >= max_requests:
        wait_time = window_seconds - (now - request_deque[0])
        if wait_time > 0:
            time.sleep(wait_time)

    # 记录本次请求时间
    request_deque.append(time.time())
