from typing import Dict
import math

import torch


def _taylor_from_factors(factors, x, max_terms=None) -> torch.Tensor:
    """
    Evaluate the Taylor polynomial stored in `factors` at offset `x`.
    """
    if max_terms is None:
        max_terms = len(factors)

    output = 0
    for i in range(max_terms):
        output += (1 / math.factorial(i)) * factors[i] * (x ** i)
    return output


def _should_use_pade(cache_dic: Dict, current: Dict, factors) -> bool:
    """
    Conservative Padé gate.

    By default Padé is disabled unless explicitly enabled from model_kwargs.
    Even when enabled, we only allow it in the safest case: single-step
    extrapolation with enough derivative information for at least [1/1].
    """
    if not cache_dic.get("enable_pade", False):
        return False

    x = current["step"] - current["activated_steps"][-1]
    if cache_dic.get("pade_only_single_step", True) and abs(x) != 1:
        return False

    return len(factors) >= 3


def _get_approx_stats(cache_dic: Dict) -> Dict:
    return cache_dic.setdefault(
        "_approx_stats",
        {
            "step_modes": {},
            "call_counts": {"pade": 0, "taylor": 0},
            "printed": False,
        },
    )


def summarize_approx_stats(cache_dic: Dict) -> Dict:
    stats = _get_approx_stats(cache_dic)
    step_modes = stats["step_modes"]
    total_steps = len(step_modes)
    pade_steps = sum("pade" in modes for modes in step_modes.values())
    taylor_steps = sum("taylor" in modes for modes in step_modes.values())
    mixed_steps = sum(modes == {"pade", "taylor"} for modes in step_modes.values())
    pure_pade_steps = sum(modes == {"pade"} for modes in step_modes.values())
    pure_taylor_steps = sum(modes == {"taylor"} for modes in step_modes.values())
    pade_calls = stats["call_counts"].get("pade", 0)
    taylor_calls = stats["call_counts"].get("taylor", 0)
    total_calls = pade_calls + taylor_calls
    return {
        "total_steps": int(total_steps),
        "pade_steps": int(pade_steps),
        "taylor_steps": int(taylor_steps),
        "mixed_steps": int(mixed_steps),
        "pure_pade_steps": int(pure_pade_steps),
        "pure_taylor_steps": int(pure_taylor_steps),
        "pade_calls": int(pade_calls),
        "taylor_calls": int(taylor_calls),
        "pade_step_ratio": float(pade_steps / total_steps) if total_steps else 0.0,
        "pade_call_ratio": float(pade_calls / total_calls) if total_calls else 0.0,
    }


def _record_approx_mode(cache_dic: Dict, current: Dict, mode: str) -> None:
    stats = _get_approx_stats(cache_dic)
    step = int(current["step"])
    stats["step_modes"].setdefault(step, set()).add(mode)
    stats["call_counts"][mode] = stats["call_counts"].get(mode, 0) + 1


def _is_last_layer_mlp(cache_dic: Dict, current: Dict) -> bool:
    cache = cache_dic.get("cache", {})
    last_cache = cache.get(-1, {})
    if not last_cache:
        return False
    return (
        current.get("layer") == max(last_cache.keys())
        and current.get("module") == "mlp"
    )


def _maybe_print_approx_stats(cache_dic: Dict, current: Dict) -> None:
    stats = cache_dic.get("_approx_stats")
    if not stats or stats.get("printed", False):
        return

    if current.get("step") != 0 or not _is_last_layer_mlp(cache_dic, current):
        return

    summary = summarize_approx_stats(cache_dic)

    # print(
    #     "[TaylorSeer统计] "
    #     f"近似步数={summary['total_steps']}, "
    #     f"Padé步数={summary['pade_steps']}, "
    #     f"Taylor步数={summary['taylor_steps']}, "
    #     f"纯Padé步数={summary['pure_pade_steps']}, "
    #     f"纯Taylor步数={summary['pure_taylor_steps']}, "
    #     f"混合步数={summary['mixed_steps']}, "
    #     f"Padé调用次数={summary['pade_calls']}, "
    #     f"Taylor调用次数={summary['taylor_calls']}"
    # )
    stats["printed"] = True


def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation.
    :param cache_dic: Cache dictionary.
    :param current: Current step information.
    """
    difference_distance = current["activated_steps"][-1] - current["activated_steps"][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature
    for i in range(cache_dic["max_order"]):
        if (
            cache_dic["cache"][-1][current["layer"]][current["module"]].get(i, None)
            is not None
        ) and (current["step"] < (current["num_steps"] - cache_dic["first_enhance"] + 1)):
            updated_taylor_factors[i + 1] = (
                updated_taylor_factors[i]
                - cache_dic["cache"][-1][current["layer"]][current["module"]][i]
            ) / difference_distance
        else:
            break

    cache_dic["cache"][-1][current["layer"]][current["module"]] = updated_taylor_factors
    _maybe_print_approx_stats(cache_dic, current)


def pade_formula_11(cache_dic: Dict, current: Dict) -> torch.Tensor:
    x = current["step"] - current["activated_steps"][-1]
    factors = cache_dic["cache"][-1][current["layer"]][current["module"]]
    return _evaluate_pade_11_from_factors(factors, x)


def _evaluate_pade_11_from_factors(factors, x) -> torch.Tensor:
    """
    Stable [1/1] Padé approximation.

    For tensor features we only use Padé as a conservative correction over the
    2nd-order Taylor polynomial. If the local coefficients indicate a nearby
    pole or an ill-conditioned division, we fall back to Taylor directly.
    """
    if len(factors) < 3:
        return _taylor_from_factors(factors, x)

    taylor_2 = _taylor_from_factors(factors, x, max_terms=3)

    c0 = factors[0]
    c1 = factors[1]
    c2 = factors[2] / 2.0

    eps = 1e-6
    scale = c0.abs() + c1.abs() * abs(x) + c2.abs() * (abs(x) ** 2) + 1.0
    deriv_ok = c1.abs() > (eps * scale)
    safe_c1 = torch.where(deriv_ok, c1, torch.ones_like(c1))

    b1 = -c2 / safe_c1
    a0 = c0
    a1 = c1 + a0 * b1

    numerator = a0 + a1 * x
    denominator = 1.0 + b1 * x

    finite_mask = (
        torch.isfinite(numerator) & torch.isfinite(denominator) & torch.isfinite(b1)
    )
    denom_ok = denominator.abs() > 0.25
    safe_mask = deriv_ok & finite_mask & denom_ok

    safe_denominator = torch.where(safe_mask, denominator, torch.ones_like(denominator))
    pade_value = numerator / safe_denominator

    pole_risk = (b1 * x).abs()
    blend = torch.clamp(1.0 - pole_risk, min=0.0, max=1.0)
    blended = blend * pade_value + (1.0 - blend) * taylor_2

    return torch.where(safe_mask, blended, taylor_2)


def pade_coefficients_from_taylor(c, m, n):
    """
    Construct Padé (m/n) coefficients from Taylor coefficients c_k.
    """
    if m < 1 or n < 1:
        raise ValueError("Padé order requires m >= 1 and n >= 1")
    if len(c) < (m + n + 1):
        raise ValueError("Not enough Taylor coefficients for the requested Padé order")

    rows = []
    rhs_terms = []
    for row in range(n):
        k = m + 1 + row
        rhs_terms.append(-c[k])
        rows.append(torch.stack([c[k - col - 1] for col in range(n)], dim=-1))

    A = torch.stack(rows, dim=-2)
    rhs = torch.stack(rhs_terms, dim=-1)
    b = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)

    a = []
    for i in range(m + 1):
        s = c[i]
        for j in range(1, min(i, n) + 1):
            s += b[..., j - 1] * c[i - j]
        a.append(s)

    a = torch.stack(a, dim=-1)
    return a, b


def choose_pade_order_by_available_derivatives(
    factors, forced_order=None, alpha=0.7, max_order=3
):
    K = len(factors) - 1
    if K < 2:
        return None

    if forced_order is not None:
        m, n = forced_order
        if m < 1 or n < 1:
            return None
        if K < (m + n):
            return None
        return m, n

    return 1, 1


def _evaluate_generic_pade(factors, x, m, n, denom_threshold=1e-4) -> torch.Tensor:
    max_needed = m + n + 1
    if len(factors) < max_needed:
        return _taylor_from_factors(factors, x)

    c = [factors[k] / math.factorial(k) for k in range(max_needed)]
    taylor_output = _taylor_from_factors(factors, x)
    a, b = pade_coefficients_from_taylor(c, m, n)

    numerator = a[..., 0]
    for i in range(1, m + 1):
        numerator = numerator + a[..., i] * (x ** i)

    denominator = torch.ones_like(numerator)
    denom_scale = torch.ones_like(numerator)
    pole_risk = torch.zeros_like(numerator)
    for j in range(1, n + 1):
        term = b[..., j - 1] * (x ** j)
        denominator = denominator + term
        abs_term = term.abs()
        denom_scale = denom_scale + abs_term
        pole_risk = pole_risk + abs_term

    coeff_finite = torch.isfinite(a).all(dim=-1) & torch.isfinite(b).all(dim=-1)
    finite_mask = torch.isfinite(numerator) & torch.isfinite(denominator) & coeff_finite
    local_denom_threshold = denom_threshold * denom_scale
    denom_margin = denominator.abs() / local_denom_threshold.clamp_min(denom_threshold)
    denom_ok = denom_margin > 1.0
    safe_mask = finite_mask & denom_ok
    safe_denominator = torch.where(safe_mask, denominator, torch.ones_like(denominator))
    pade_value = numerator / safe_denominator
    correction = pade_value - taylor_output
    taylor_scale = taylor_output.abs()
    for k, coeff in enumerate(c):
        taylor_scale = taylor_scale + coeff.abs() * (abs(x) ** k)
    correction_ratio = correction.abs() / taylor_scale.clamp_min(1e-6)

    denom_blend = torch.clamp(denom_margin - 1.0, min=0.0, max=1.0)
    pole_blend = torch.clamp(1.0 - pole_risk, min=0.0, max=1.0)
    correction_blend = torch.clamp(1.0 - correction_ratio, min=0.0, max=1.0)
    safe_mask = safe_mask & (pole_risk < 0.5) & (correction_ratio < 1.0)

    blend = torch.minimum(torch.minimum(denom_blend, pole_blend), correction_blend)
    blended = blend * pade_value + (1.0 - blend) * taylor_output
    return torch.where(safe_mask, blended, taylor_output)


def pade_formula_mn(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Adaptive Padé / Taylor approximation.
    Drop-in replacement for taylor_formula.
    """
    x = current["step"] - current["activated_steps"][-1]
    factors = cache_dic["cache"][-1][current["layer"]][current["module"]]

    if not _should_use_pade(cache_dic, current, factors):
        _record_approx_mode(cache_dic, current, "taylor")
        output = _taylor_from_factors(factors, x)
        _maybe_print_approx_stats(cache_dic, current)
        return output

    forced_order = None
    if cache_dic.get("pade_m") is not None and cache_dic.get("pade_n") is not None:
        forced_order = (cache_dic["pade_m"], cache_dic["pade_n"])

    order = choose_pade_order_by_available_derivatives(factors, forced_order=forced_order)
    if order is None:
        _record_approx_mode(cache_dic, current, "taylor")
        output = _taylor_from_factors(factors, x)
        _maybe_print_approx_stats(cache_dic, current)
        return output

    m, n = order
    if (m, n) == (1, 1):
        _record_approx_mode(cache_dic, current, "pade")
        output = pade_formula_11(cache_dic, current)
        _maybe_print_approx_stats(cache_dic, current)
        return output

    try:
        output = _evaluate_generic_pade(
            factors,
            x,
            m,
            n,
            denom_threshold=cache_dic.get("pade_denom_threshold", 1e-4),
        )
    except RuntimeError:
        _record_approx_mode(cache_dic, current, "pade")
        output = _evaluate_pade_11_from_factors(factors, x)
        _maybe_print_approx_stats(cache_dic, current)
        return output

    _record_approx_mode(cache_dic, current, "pade")
    _maybe_print_approx_stats(cache_dic, current)
    return output





def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Compute Taylor expansion error.
    :param cache_dic: Cache dictionary.
    :param current: Current step information.
    """
    x = current["step"] - current["activated_steps"][-1]
    factors = cache_dic["cache"][-1][current["layer"]][current["module"]]
    _record_approx_mode(cache_dic, current, "taylor")
    output = _taylor_from_factors(factors, x)
    _maybe_print_approx_stats(cache_dic, current)
    return output


def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and expand storage for different-order derivatives.
    :param cache_dic: Cache dictionary.
    :param current: Current step information.
    """
    if current["step"] == (current["num_steps"] - 1):
        cache_dic["cache"][-1][current["layer"]][current["module"]] = {}



# def pade_coefficients_from_taylor(c, m, n):
#     """
#     Construct Padé (m/n) coefficients from Taylor coefficients c_k.
#     This is equivalent to the extended Euclidean algorithm.

#     c: list of Taylor coefficients c_k = f^(k) / k!
#     m: numerator degree
#     n: denominator degree

#     return:
#         a: tensor of shape (m+1,)  numerator coefficients
#         b: tensor of shape (n,)    denominator coefficients (b1...bn)
#     """
#     device = c[0].device
#     dtype = c[0].dtype

#     # Solve for denominator coefficients b_j
#     # ∑_{j=1}^n c_{k-j} b_j = -c_k,  k = m+1 ... m+n
#     A = torch.zeros((n, n), device=device, dtype=dtype)
#     rhs = torch.zeros((n,), device=device, dtype=dtype)

#     for row in range(n):
#         k = m + 1 + row
#         rhs[row] = -c[k]
#         for col in range(n):
#             A[row, col] = c[k - col - 1]

#     b = torch.linalg.solve(A, rhs)  # (b1...bn)

#     # Solve for numerator coefficients a_i
#     a = []
#     for i in range(m + 1):
#         s = c[i]
#         for j in range(1, min(i, n) + 1):
#             s += b[j - 1] * c[i - j]
#         a.append(s)

#     a = torch.stack(a)
#     return a, b


# def choose_pade_order_by_available_derivatives(
#     factors,
#     alpha=0.7,
#     max_order=3
# ):
#     K = len(factors) - 1
#     if K < 2:
#         return None

#     max_mn = min(max_order, int(alpha * K // 2))
#     if max_mn < 1:
#         return None

#     return max_mn, max_mn




# def pade_formula_mn(cache_dic: Dict, current: Dict) -> torch.Tensor:
#     """
#     Adaptive Padé / Taylor approximation.
#     Drop-in replacement for taylor_formula.
#     """

#     x = current['step'] - current['activated_steps'][-1]
#     factors = cache_dic['cache'][-1][current['layer']][current['module']]

#     # ===============================
#     # Step 1: 自适应选择 Padé 阶数
#     # ===============================
#     order = choose_pade_order_by_available_derivatives(factors)

#     # ---------- 回退 Taylor ----------
#     if order is None:
#         output = 0
#         for i in range(len(factors)):
#             output += (1 / math.factorial(i)) * factors[i] * (x ** i)
#         return output

#     m, n = order
#     max_needed = m + n + 1

#     # ===============================
#     # Step 2: 再做一次安全检查
#     # ===============================
#     if len(factors) < max_needed:
#         output = 0
#         for i in range(len(factors)):
#             output += (1 / math.factorial(i)) * factors[i] * (x ** i)
#         return output

#     # ===============================
#     # Step 3: 构造 Taylor 系数
#     # ===============================
#     c = [factors[k] / math.factorial(k) for k in range(max_needed)]

#     # ===============================
#     # Step 4: 欧几里得法构造 Padé
#     # ===============================
#     try:
#         a, b = pade_coefficients_from_taylor(c, m, n)
#     except RuntimeError:
#         # 数值不稳定，安全回退 Taylor
#         output = 0
#         for i in range(max_needed):
#             output += c[i] * (x ** i)
#         return output

#     # ===============================
#     # Step 5: 计算有理函数值
#     # ===============================
#     numerator = a[0]
#     for i in range(1, m + 1):
#         numerator = numerator + a[i] * (x ** i)

#     denominator = 1.0
#     for j in range(1, n + 1):
#         denominator = denominator + b[j - 1] * (x ** j)

#     return numerator / denominator







# #原版
# def pade_coefficients_from_taylor(c, m, n):
#     """
#     Construct Padé (m/n) coefficients from Taylor coefficients c_k.
#     This is equivalent to the extended Euclidean algorithm.

#     c: list of Taylor coefficients c_k = f^(k) / k!
#     m: numerator degree
#     n: denominator degree

#     return:
#         a: tensor of shape (m+1,)  numerator coefficients
#         b: tensor of shape (n,)    denominator coefficients (b1...bn)
#     """
#     device = c[0].device
#     dtype = c[0].dtype

#     # Solve for denominator coefficients b_j
#     # ∑_{j=1}^n c_{k-j} b_j = -c_k,  k = m+1 ... m+n
#     A = torch.zeros((n, n), device=device, dtype=dtype)
#     rhs = torch.zeros((n,), device=device, dtype=dtype)

#     for row in range(n):
#         k = m + 1 + row
#         rhs[row] = -c[k]
#         for col in range(n):
#             A[row, col] = c[k - col - 1]

#     b = torch.linalg.solve(A, rhs)  # (b1...bn)

#     # Solve for numerator coefficients a_i
#     a = []
#     for i in range(m + 1):
#         s = c[i]
#         for j in range(1, min(i, n) + 1):
#             s += b[j - 1] * c[i - j]
#         a.append(s)

#     a = torch.stack(a)
#     return a, b


# def choose_pade_order_by_available_derivatives(
#     factors,
#     alpha=0.7,
#     max_order=3
# ):
#     K = len(factors) - 1
#     if K < 2:
#         return None

#     max_mn = min(max_order, int(alpha * K // 2))
#     if max_mn < 1:
#         return None

#     return max_mn, max_mn


# def pade_formula_mn(cache_dic: Dict, current: Dict) -> torch.Tensor:
#     """
#     Adaptive Padé / Taylor approximation (新版替换版).
#     Drop-in replacement for taylor_formula.

#     保留统计打印功能：
#     - 调用 _record_approx_mode 记录当前步使用 Padé/Taylor
#     - 调用 _maybe_print_approx_stats 打印统计
#     """
#     x = current["step"] - current["activated_steps"][-1]
#     factors = cache_dic["cache"][-1][current["layer"]][current["module"]]

#     # ===============================
#     # Step 1: 自适应选择 Padé 阶数
#     # ===============================
#     order = choose_pade_order_by_available_derivatives(factors)

#     # ---------- 回退 Taylor ----------
#     if order is None:
#         _record_approx_mode(cache_dic, current, "taylor")
#         output = _taylor_from_factors(factors, x)
#         _maybe_print_approx_stats(cache_dic, current)
#         return output

#     m, n = order
#     max_needed = m + n + 1

#     # ===============================
#     # Step 2: 再做一次安全检查
#     # ===============================
#     if len(factors) < max_needed:
#         _record_approx_mode(cache_dic, current, "taylor")
#         output = _taylor_from_factors(factors, x)
#         _maybe_print_approx_stats(cache_dic, current)
#         return output

#     # ===============================
#     # Step 3: 构造 Taylor 系数
#     # ===============================
#     c = [factors[k] / math.factorial(k) for k in range(max_needed)]

#     # ===============================
#     # Step 4: 欧几里得法构造 Padé
#     # ===============================
#     try:
#         a, b = pade_coefficients_from_taylor(c, m, n)
#     except RuntimeError:
#         # 数值不稳定，安全回退 Taylor
#         _record_approx_mode(cache_dic, current, "taylor")
#         output = _taylor_from_factors(factors, x)
#         _maybe_print_approx_stats(cache_dic, current)
#         return output

#     # ===============================
#     # Step 5: 计算有理函数值
#     # ===============================
#     numerator = a[0]
#     for i in range(1, m + 1):
#         numerator = numerator + a[i] * (x ** i)

#     denominator = 1.0
#     for j in range(1, n + 1):
#         denominator = denominator + b[j - 1] * (x ** j)

#     pade_value = numerator / denominator

#     _record_approx_mode(cache_dic, current, "pade")
#     _maybe_print_approx_stats(cache_dic, current)
#     return pade_value
