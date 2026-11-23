"""Batched evaluation of existing periodic pooling statistics, without label tuning."""
from modern_extension import *
from numba import njit,prange,literally
from rocket_official_experiment import response_parts,KERNELS
from rocket_mpeg7_confirm import interval

def prepare_orders(counts, biases):
    """Sort each kernel/dilation threshold group once; reuse that order for every contour."""
    threshold_orders = np.empty(len(biases), dtype=np.int32)
    group_start = 0
    for group_size in counts:
        for kernel in range(84):
            threshold_orders[group_start:group_start + group_size] = np.argsort(biases[group_start:group_start + group_size], kind='stable')
            group_start += group_size
    assert group_start == len(biases)
    return threshold_orders

@njit(cache=True)
def add_compensated(running_sum, correction, term):
    """Accumulate a term and retain rounding loss in a separate correction."""
    updated_sum = running_sum + term
    if abs(running_sum) >= abs(term):
        correction += running_sum - updated_sum + term
    else:
        correction += term - updated_sum + running_sum
    return (updated_sum, correction)

@njit(cache=True)
def direct_queries(response, thresholds):
    """Average the exact above-threshold length over the closed response edges."""
    pooled = np.empty(len(thresholds), np.float32)
    length = len(response)
    for threshold_index in range(len(thresholds)):
        threshold = np.float64(thresholds[threshold_index])
        positive_length = 0.0
        for edge_index in range(length):
            positive_length += interval(np.float64(response[edge_index]) - threshold, np.float64(response[(edge_index + 1) % length]) - threshold, 0, False)
        pooled[threshold_index] = positive_length / length
    return pooled

@njit(cache=True)
def cached_endpoint_queries(response, thresholds, K):
    """Reuse edge endpoints for exact EI (K=0) or fixed-K positive-sample counts."""
    K = literally(K)
    length = len(response)
    edge_lower = np.empty(length, np.float64)
    edge_upper = np.empty(length, np.float64)
    edge_rising = np.empty(length, np.bool_)
    response_min = np.float64(response[0])
    response_max = np.float64(response[0])
    for edge_index in range(length):
        left_value = np.float64(response[edge_index])
        right_value = np.float64(response[(edge_index + 1) % length])
        edge_lower[edge_index] = min(left_value, right_value)
        edge_upper[edge_index] = max(left_value, right_value)
        edge_rising[edge_index] = left_value < right_value
        response_min = min(response_min, left_value)
        response_max = max(response_max, left_value)
    pooled = np.empty(len(thresholds), np.float32)
    for threshold_index in range(len(thresholds)):
        threshold = np.float64(thresholds[threshold_index])
        positive_length = 0.0
        positive_samples = 0
        if threshold < response_min:
            pooled[threshold_index] = 1.0
            continue
        if threshold >= response_max:
            pooled[threshold_index] = 0.0
            continue
        for edge_index in range(length):
            if threshold < edge_lower[edge_index]:
                if K == 0:
                    positive_length += 1.0
                else:
                    positive_samples += K
            elif threshold >= edge_upper[edge_index]:
                continue
            elif K == 0:
                positive_length += (edge_upper[edge_index] - threshold) / (edge_upper[edge_index] - edge_lower[edge_index])
            elif edge_rising[edge_index]:
                positive_samples += K - 1 - int(np.floor(K * (threshold - edge_lower[edge_index]) / (edge_upper[edge_index] - edge_lower[edge_index])))
            else:
                positive_samples += int(np.ceil(K * (edge_upper[edge_index] - threshold) / (edge_upper[edge_index] - edge_lower[edge_index])))
        pooled[threshold_index] = positive_length / length if K == 0 else positive_samples / (length * K)
    return pooled

@njit(cache=True)
def sweep_queries(response, thresholds, threshold_order):
    """Sweep the response-value axis, integrating edge densities and flat-edge point masses."""
    length = len(response)
    pooled = np.empty(len(thresholds), np.float32)
    density_events = np.zeros(length, np.float64)
    point_masses = np.zeros(length, np.int32)
    vertex_order = np.argsort(response)
    response_min = np.float64(response[vertex_order[0]])
    response_max = np.float64(response[vertex_order[-1]])
    smallest_edge = np.inf
    for edge_index in range(length):
        next_edge_index = (edge_index + 1) % length
        left_value = np.float64(response[edge_index])
        right_value = np.float64(response[next_edge_index])
        edge_height = abs(left_value - right_value)
        if edge_height == 0:
            point_masses[edge_index] += 1
        else:
            smallest_edge = min(smallest_edge, edge_height)
            edge_density = 1.0 / edge_height
            if left_value < right_value:
                density_events[edge_index] += edge_density
                density_events[next_edge_index] -= edge_density
            else:
                density_events[next_edge_index] += edge_density
                density_events[edge_index] -= edge_density
    # Very short edges create large reciprocal weights; use the direct integral
    # when cancellation would make the sweep poorly conditioned.
    if response_max > response_min and (response_max - response_min) / smallest_edge > 10000000000.0:
        return direct_queries(response, thresholds)
    active_density = 0.0
    density_correction = 0.0
    remaining_mass = float(length)
    mass_correction = 0.0
    scan_position = response_min
    vertex_cursor = 0
    for k in range(len(thresholds)):
        threshold_index = threshold_order[k]
        threshold = np.float64(thresholds[threshold_index])
        if threshold < response_min:
            pooled[threshold_index] = 1.0
            continue
        if threshold >= response_max:
            pooled[threshold_index] = 0.0
            continue
        while vertex_cursor < length and np.float64(response[vertex_order[vertex_cursor]]) <= threshold:
            vertex_value = np.float64(response[vertex_order[vertex_cursor]])
            remaining_mass, mass_correction = add_compensated(remaining_mass, mass_correction, -(active_density + density_correction) * (vertex_value - scan_position))
            scan_position = vertex_value
            while vertex_cursor < length and np.float64(response[vertex_order[vertex_cursor]]) == vertex_value:
                vertex_index = vertex_order[vertex_cursor]
                active_density, density_correction = add_compensated(active_density, density_correction, density_events[vertex_index])
                remaining_mass, mass_correction = add_compensated(remaining_mass, mass_correction, -float(point_masses[vertex_index]))
                vertex_cursor += 1
        pooled_value = (remaining_mass + mass_correction - (active_density + density_correction) * (threshold - scan_position)) / length
        if pooled_value < -1e-08 or pooled_value > 1 + 1e-08 or (not np.isfinite(pooled_value)):
            return direct_queries(response, thresholds)
        pooled[threshold_index] = min(1.0, max(0.0, pooled_value))
    return pooled

@njit(cache=True)
def sorted_sample_queries(response, thresholds, threshold_order, K):
    """Sort interpolated phase samples; count strict threshold exceedances."""
    length = len(response)
    phase_samples = np.empty(length * K, np.float64)
    for edge_index in range(length):
        left_value = np.float64(response[edge_index])
        right_value = np.float64(response[(edge_index + 1) % length])
        for k in range(K):
            phase_samples[edge_index * K + k] = ((K - k) * left_value + k * right_value) / K
    phase_samples.sort()
    count_le_threshold = 0
    pooled = np.empty(len(thresholds), np.float32)
    tie_tolerance = 8 * np.finfo(np.float64).eps * max(abs(float(response.min())), abs(float(response.max())), np.finfo(np.float64).tiny)
    for k in range(len(thresholds)):
        threshold_index = threshold_order[k]
        right_value = np.float64(thresholds[threshold_index])
        while count_le_threshold < len(phase_samples) and phase_samples[count_le_threshold] <= right_value:
            count_le_threshold += 1
        at_rounding_tie = False
        if count_le_threshold < len(phase_samples) and abs(phase_samples[count_le_threshold] - right_value) <= tie_tolerance:
            at_rounding_tie = True
        if count_le_threshold > 0 and abs(phase_samples[count_le_threshold - 1] - right_value) <= tie_tolerance:
            at_rounding_tie = True
        # Near a threshold tie, use the same centered arithmetic as the
        # canonical finite-K counter so strict > semantics stay consistent.
        if at_rounding_tie:
            positive_length = 0.0
            for edge_index in range(length):
                positive_length += interval(np.float64(response[edge_index]) - right_value, np.float64(response[(edge_index + 1) % length]) - right_value, K, True)
            pooled[threshold_index] = positive_length / length
        else:
            pooled[threshold_index] = (len(phase_samples) - count_le_threshold) / len(phase_samples)
    return pooled

@njit(parallel=True, cache=True)
def batched_features(signals, dilations, counts, biases, orders, mode, sweep_min_thresholds=0):
    """Extract periodic features with a selected pooling backend.

mode=0 selects EI. A positive sweep_min_thresholds enables the hybrid
dispatch: direct for B<8, cached endpoints for 8<=B<L, sweep for B>=L.
Negative modes select fixed cached backends; positive modes give finite K."""
    feature_matrix = np.empty((len(signals), len(biases)), np.float32)
    for example in prange(len(signals)):
        group_start = 0
        for dilation_index in range(len(dilations)):
            alpha, gamma = response_parts(signals[example], dilations[dilation_index], True)
            threshold_count = counts[dilation_index]
            for i, j, k in KERNELS:
                response = alpha + gamma[i] + gamma[j] + gamma[k]
                thresholds = biases[group_start:group_start + threshold_count]
                threshold_order = orders[group_start:group_start + threshold_count]
                # A short threshold group does not amortize sweep setup.
                # Dense groups share one sweep; intermediate groups reuse endpoints.
                if mode == 0:
                    if sweep_min_thresholds > 0 and threshold_count < 8:
                        pooled_response = direct_queries(response, thresholds)
                    elif sweep_min_thresholds > 0 and threshold_count < sweep_min_thresholds:
                        pooled_response = cached_endpoint_queries(response, thresholds, 0)
                    else:
                        pooled_response = sweep_queries(response, thresholds, threshold_order)
                elif mode == -2:
                    pooled_response = cached_endpoint_queries(response, thresholds, 0)
                elif mode == -1:
                    pooled_response = sorted_sample_queries(response, thresholds, threshold_order, 1)
                elif mode == -4:
                    pooled_response = cached_endpoint_queries(response, thresholds, 4)
                elif mode == -16:
                    pooled_response = cached_endpoint_queries(response, thresholds, 16)
                elif mode == -32:
                    pooled_response = cached_endpoint_queries(response, thresholds, 32)
                elif mode == -64:
                    pooled_response = cached_endpoint_queries(response, thresholds, 64)
                elif mode == -128:
                    pooled_response = cached_endpoint_queries(response, thresholds, 128)
                elif mode == -1024:
                    pooled_response = cached_endpoint_queries(response, thresholds, 1024)
                else:
                    pooled_response = sorted_sample_queries(response, thresholds, threshold_order, mode)
                feature_matrix[example, group_start:group_start + threshold_count] = pooled_response
                group_start += threshold_count
    return feature_matrix

def probe():
    import csv
    dest=HERE.parent/'outputs/shared_threshold_study';dest.mkdir(exist_ok=True)
    rows=[];rng=np.random.default_rng(193)
    # Use training data exclusively for this initial implementation/efficiency probe.
    with threadpool_limits(limits=1):
        for n in (32,64):
            d,_=input_data(f'fish_{n}');x=np.ascontiguousarray(d['train'][:32])
            for budget in (756,9996,49728):
                p=params(d['train'],'eippv',23,budget);orders=prepare_orders(p[1],p[2]);ref=features(x,*p,0,False)
                results={}
                for name,fn in [('direct_EI',lambda:features(x,*p,0,False)),('cached_EI',lambda:batched_features(x,*p,orders,-2)),('sweep_EI',lambda:batched_features(x,*p,orders,0)),('hybrid_EI',lambda:batched_features(x,*p,orders,0,n)),('K32_count',lambda:features(x,*p,32,True)),('K32_cached',lambda:batched_features(x,*p,orders,-32)),('K32_sort',lambda:batched_features(x,*p,orders,32)),('hard_sort',lambda:batched_features(x,*p,orders,-1))]:
                    val=fn();assert np.isfinite(val).all()
                    if name in ('sweep_EI','hybrid_EI','cached_EI'):np.testing.assert_allclose(val,ref,rtol=0,atol=2e-7)
                    if name in ('K32_cached','K32_sort'):np.testing.assert_array_equal(val,features(x,*p,32,True))
                    tt=[]
                    for _ in range(5):
                        start=time.perf_counter();fn();tt.append(1000*(time.perf_counter()-start))
                    results[name]=float(np.median(tt))
                row=dict(length=n,budget=budget,actual_features=ref.shape[1],**results)
                rows.append(row);print(json.dumps(row),flush=True)
    with (dest/'training_probe.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':set_num_threads(2);probe()
