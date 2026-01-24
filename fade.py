import math
import shutil
import sys
import time
import argparse
import torch
import torch.nn.functional as F
import os
import numpy as np
import multiprocessing as mp
import ctypes
from numba import jit
import threading
import struct
import arithmeticcoding_fast
from thop import profile

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MODEL_REGISTRY = {
    'fade': 'fade_model.FADE',
}

def get_model(model_name, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
    
    module_path, class_name = MODEL_REGISTRY[model_name].rsplit('.', 1)
    try:
        module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(module, class_name)
        return model_class(**kwargs)
    except ImportError as e:
        print(f'Error importing {model_name}: {e}')
        sys.exit(1)

def calculate_model_stats(model, input_shape, vocab_size, device='cuda'):
    model.eval()
    dummy_input = torch.randint(0, vocab_size, input_shape).to(device).long()
    
    try:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops_str = '{:.3f}'.format(flops / 1e9)
        params_str = '{:.3f}'.format(params / 1e6)
    except Exception as e:
        print(f'Warning: Failed to calculate FLOPs/Params: {e}')
        flops_str = 'N/A'
        params_str = 'N/A'
    
    # 计算延迟warm up + 多次测试取平均
    try:
        with torch.no_grad():
            # Warm up
            for _ in range(10):
                _ = model(dummy_input)
            
            torch.cuda.synchronize()
            start_time = time.time()
            num_runs = 100
            for _ in range(num_runs):
                _ = model(dummy_input)
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_latency = (end_time - start_time) / num_runs * 1000
            latency_str = '{:.2f}'.format(avg_latency)
    except Exception as e:
        print(f'Warning: Failed to calculate latency: {e}')
        latency_str = 'N/A'
    
    model.train()
    
    return {
        'flops': flops_str,
        'params': params_str,
        'latency': latency_str
    }

def print_compression_results(data_name, original_size, compressed_size, compression_time, model_stats, gpu_memory):
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    throughput = original_size / compression_time if compression_time > 0 else 0  # bytes/s
    throughput_kb = throughput / 1024  # KB/s
    
    width = 77
    
    print(' COMPRESSION RESULTS '.center(width, '='))
    
    print(f'{"Dataset":^12} {"Orig.(B)":^12} {"Cmp.(B)":^12} {"CR":^12} {"Time(s)":^12} {"TP(KB/s)":^12}')
    print(f'{data_name:^12} {original_size:^12} {compressed_size:^12} {compression_ratio:^12.5f} {compression_time:^12.3f} {throughput_kb:^12.3f}')

    print('-' * width)
    print(f'{"GFLOPs":^12} {"Params(M)":^12} {"Latency(ms)":^12} {"GPU Mem(MB)":^12}')
    print(f'{model_stats["flops"]:^12} {model_stats["params"]:^12} {model_stats["latency"]:^12} {gpu_memory / 1024:^12.3f}')
    print('=' * width + '\n')

def print_decompression_results(data_name, compressed_size, decompressed_size, decompression_time, model_stats, gpu_memory):
    throughput = decompressed_size / decompression_time if decompression_time > 0 else 0  # bytes/s
    throughput_kb = throughput / 1024  # KB/s
    
    width = 64
    
    print(' DECOMPRESSION RESULTS '.center(width, '='))
    
    print(f'{"Dataset":^12} {"Cmp.(B)":^12} {"Decmp.(B)":^12} {"Time(s)":^12} {"TP(KB/s)":^12}')
    print(f'{data_name:^12} {compressed_size:^12} {decompressed_size:^12} {decompression_time:^12.3f} {throughput_kb:^12.3f}')

    print('-' * width)

    print(f'{"GFLOPs":^12} {"Params(M)":^12} {"Latency(ms)":^12} {"GPU Mem(MB)":^12}')
    print(f'{model_stats["flops"]:^12} {model_stats["params"]:^12} {model_stats["latency"]:^12} {gpu_memory / 1024:^12.3f}')
    print('=' * width + '\n')


# === aux functions ===
def strided_app(a, L, S):
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))

def var_int_encode(byte_str_len, f):
    while True:
        this_byte = byte_str_len & 127
        byte_str_len >>= 7
        if byte_str_len == 0:
            f.write(struct.pack('B', this_byte))
            break
        f.write(struct.pack('B', this_byte | 128))
        byte_str_len -= 1

def var_int_decode(f):
    byte_str_len = 0
    shift = 1
    while True:
        this_byte = struct.unpack('B', f.read(1))[0]
        byte_str_len += (this_byte & 127) * shift
        if this_byte & 128 == 0:
            break
        shift <<= 7
        byte_str_len += shift
    return byte_str_len


@jit(nopython=True)
def cumulative_sum_inplace(prob, cumul, batch_size, vocab_size):
    scale = 10000000
    for i in range(batch_size):
        cumul[i, 0] = 0
        s = 0
        for j in range(vocab_size):
            val = int(prob[i, j] * scale + 1)
            s += val
            cumul[i, j + 1] = s

def compress_worker_double_buffer(rank, num_workers, bs, vocab_size, ts, temp_file, shared_cumul_A, shared_cumul_B, shared_y_A, shared_y_B, shared_context, events_gpu_ready, events_cpu_done, stop_flag, total_iters, worker_sync_barrier):
    try:
        chunk_size = bs // num_workers
        start_idx = rank * chunk_size

        # 从共享内存创建 numpy 视图
        cumul_buffers = [
            np.frombuffer(shared_cumul_A.get_obj(), dtype=np.uint64).reshape(bs, vocab_size + 1),
            np.frombuffer(shared_cumul_B.get_obj(), dtype=np.uint64).reshape(bs, vocab_size + 1)
        ]
        y_buffers = [
            np.frombuffer(shared_y_A.get_obj(), dtype=np.int32),
            np.frombuffer(shared_y_B.get_obj(), dtype=np.int32)
        ]
        context_np = np.frombuffer(shared_context.get_obj(), dtype=np.int32).reshape(bs, ts)

        f = [open(temp_file + '.' + str(i), 'wb') for i in range(start_idx, start_idx + chunk_size)]
        bitout = [arithmeticcoding_fast.BitOutputStream(f[i]) for i in range(len(f))]
        enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i]) for i in range(len(f))]

        prob_uniform = np.ones(vocab_size) / vocab_size
        cumul_uniform = np.zeros(vocab_size + 1, dtype=np.uint64)
        cumul_uniform[1:] = np.cumsum(prob_uniform * 10000000 + 1)

        for i in range(chunk_size):
            global_idx = start_idx + i
            for j in range(ts):
                enc[i].write(cumul_uniform, context_np[global_idx, j])

        for step in range(total_iters):
            buf_idx = step % 2
            
            events_gpu_ready[buf_idx].wait()
            
            if stop_flag.value:
                break

            cumul_np = cumul_buffers[buf_idx]
            y_np = y_buffers[buf_idx]
            for i in range(chunk_size):
                global_idx = start_idx + i
                enc[i].write(cumul_np[global_idx], y_np[global_idx])

            # 等待所有 Worker 完成编码, 防止 Rank 0 提前通知 GPU 覆写数据
            worker_sync_barrier.wait()

            # 通知 GPU 可以重用这个 buffer, 由 rank 0 负责
            if rank == 0:
                events_gpu_ready[buf_idx].clear()
                events_cpu_done[buf_idx].set()

        # 清理
        for i in range(chunk_size):
            enc[i].finish()
            bitout[i].close()
            f[i].close()

    except Exception as e:
        print(f'Compress worker {rank} failed: {e}')
        import traceback
        traceback.print_exc()

def decompress_worker_sync(rank, num_workers, bs, vocab_size, ts, temp_file, shared_cumul, shared_series, barrier_gpu_ready, barrier_cpu_done, barrier_init_done, stop_flag, total_iters, iter_num):
    try:
        chunk_size = bs // num_workers
        start_idx = rank * chunk_size

        cumul_np = np.frombuffer(shared_cumul.get_obj(), dtype=np.uint64).reshape(bs, vocab_size + 1)
        series_np = np.frombuffer(shared_series.get_obj(), dtype=np.int32).reshape(bs, iter_num)

        f = [open(temp_file + '.' + str(i), 'rb') for i in range(start_idx, start_idx + chunk_size)]
        bitin = [arithmeticcoding_fast.BitInputStream(f[i]) for i in range(len(f))]
        dec = [arithmeticcoding_fast.ArithmeticDecoder(32, bitin[i]) for i in range(len(f))]

        prob_uniform = np.ones(vocab_size) / vocab_size
        cumul_uniform = np.zeros(vocab_size + 1, dtype=np.uint64)
        cumul_uniform[1:] = np.cumsum(prob_uniform * 10000000 + 1)

        for i in range(chunk_size):
            global_idx = start_idx + i
            for j in range(min(ts, iter_num)):
                series_np[global_idx, j] = dec[i].read(cumul_uniform, vocab_size)

        barrier_init_done.wait()

        current_pos = ts
        for step in range(total_iters):
            # 等待 GPU 写好概率
            barrier_gpu_ready.wait()
            
            if stop_flag.value:
                barrier_cpu_done.wait()
                break

            # 解码
            for i in range(chunk_size):
                global_idx = start_idx + i
                symbol = dec[i].read(cumul_np[global_idx], vocab_size)
                series_np[global_idx, current_pos] = symbol

            current_pos += 1
            # 通知 GPU 解码完成
            barrier_cpu_done.wait()

        for i in range(chunk_size):
            bitin[i].close()
            f[i].close()

    except Exception as e:
        print(f'Decompress worker {rank} failed: {e}')
        import traceback
        traceback.print_exc()



def compress_chunk(args, temp_file, series, train_data, final):
    bs, ts = args.batch_size, args.timesteps
    vocab_size = args.vocab_size

    num_workers = min(args.num_workers, mp.cpu_count(), bs)
    while bs % num_workers != 0 and num_workers > 1:
        num_workers -= 1
    
    print(f'Compression using {num_workers} worker processes...')

    shared_cumul_A = mp.Array(ctypes.c_uint64, bs * (vocab_size + 1))
    shared_cumul_B = mp.Array(ctypes.c_uint64, bs * (vocab_size + 1))
    shared_y_A = mp.Array(ctypes.c_int32, bs)
    shared_y_B = mp.Array(ctypes.c_int32, bs)
    shared_context = mp.Array(ctypes.c_int32, bs * ts)

    events_gpu_ready = [mp.Event(), mp.Event()]
    events_cpu_done = [mp.Event(), mp.Event()]
    stop_flag = mp.Value(ctypes.c_bool, False)

    worker_sync_barrier = mp.Barrier(num_workers)

    events_cpu_done[0].set()
    events_cpu_done[1].set()

    iter_num = len(train_data) // bs
    ind = np.array(range(bs)) * iter_num
    total_iters = iter_num - ts

    context_np = np.frombuffer(shared_context.get_obj(), dtype=np.int32).reshape(bs, ts)
    for i in range(bs):
        context_np[i] = series[ind[i]:ind[i] + ts]

    processes = []
    for rank in range(num_workers):
        p = mp.Process(target=compress_worker_double_buffer, args=(
            rank, num_workers, bs, vocab_size, ts, temp_file,
            shared_cumul_A, shared_cumul_B,
            shared_y_A, shared_y_B,
            shared_context,
            events_gpu_ready, events_cpu_done,
            stop_flag, total_iters, worker_sync_barrier
        ))
        p.start()
        processes.append(p)

    cumul_buffers = [
        np.frombuffer(shared_cumul_A.get_obj(), dtype=np.uint64).reshape(bs, vocab_size + 1),
        np.frombuffer(shared_cumul_B.get_obj(), dtype=np.uint64).reshape(bs, vocab_size + 1)
    ]
    y_buffers = [
        np.frombuffer(shared_y_A.get_obj(), dtype=np.int32),
        np.frombuffer(shared_y_B.get_obj(), dtype=np.int32)
    ]

    model = get_model(args.model, batch_size=bs, timesteps=ts, vocab_dim=args.vocab_dim, vocab_size=vocab_size).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    flag = 0

    try:
        for train_index in range(total_iters):
            buf_idx = train_index % 2
            
            events_cpu_done[buf_idx].wait()
            events_cpu_done[buf_idx].clear()

            model.train()
            train_batch = train_data[ind, :]
            y = train_batch[:, -1]

            train_batch_gpu = torch.from_numpy(train_batch).cuda().long()
            logits = model.forward(train_batch_gpu[:, :-1])
            loss = F.cross_entropy(logits[:, -1, :], train_batch_gpu[:, -1])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            prob = F.softmax(logits[:, -1, :], dim=1).detach().cpu().numpy()
            cumulative_sum_inplace(prob, cumul_buffers[buf_idx], bs, vocab_size)
            np.copyto(y_buffers[buf_idx], y)

            events_gpu_ready[buf_idx].set()
            ind += 1
            if train_index >= (total_iters * 0.1 * flag):
                print(f'[Progress:{flag * 10:>3.0f}%] Current Bit-Rate: {loss.item() / np.log(2):.10f} bps')
                flag += 1

    finally:
        events_cpu_done[0].wait()
        events_cpu_done[1].wait()
        stop_flag.value = True
        events_gpu_ready[0].set()
        events_gpu_ready[1].set()
        for p in processes:
            p.join()

    if final is not None:
        f = open(temp_file + '.last', 'wb')
        bitout = arithmeticcoding_fast.BitOutputStream(f)
        enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
        prob = np.ones(vocab_size) / vocab_size
        cumul = np.zeros(vocab_size + 1, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob * 10000000 + 1)
        for j in range(len(final)):
            enc.write(cumul, int(final[j]))
        enc.finish()
        bitout.close()
        f.close()

    print(f'[Progress:{100}%] Compression finished.')
    
    return model # 返回模型用于计算性能

def decompress_chunk(args, temp_file, info, last):
    bs, ts = args.batch_size, args.timesteps
    len_series = info['len_series']
    vocab_size = args.vocab_size

    iter_num = (len_series - ts) // bs
    total_iters = iter_num - ts

    num_workers = min(args.num_workers, mp.cpu_count(), bs)
    while bs % num_workers != 0 and num_workers > 1:
        num_workers -= 1

    print(f'Decompression using {num_workers} worker processes...')

    # 单缓冲，因为解压必须同步
    shared_cumul = mp.Array(ctypes.c_uint64, bs * (vocab_size + 1))
    shared_series = mp.Array(ctypes.c_int32, bs * iter_num)

    # 使用 Barrier 保证严格同步
    barrier_gpu_ready = mp.Barrier(num_workers + 1)
    barrier_cpu_done = mp.Barrier(num_workers + 1)
    barrier_init_done = mp.Barrier(num_workers + 1)
    stop_flag = mp.Value(ctypes.c_bool, False)

    processes = []
    for rank in range(num_workers):
        p = mp.Process(target=decompress_worker_sync, args=(
            rank, num_workers, bs, vocab_size, ts, temp_file,
            shared_cumul, shared_series,
            barrier_gpu_ready, barrier_cpu_done,
            barrier_init_done, stop_flag, total_iters, iter_num
        ))
        p.start()
        processes.append(p)

    barrier_init_done.wait()

    cumul_np = np.frombuffer(shared_cumul.get_obj(), dtype=np.uint64).reshape(bs, vocab_size + 1)
    series_np = np.frombuffer(shared_series.get_obj(), dtype=np.int32).reshape(bs, iter_num)

    model = get_model(args.model, batch_size=bs, timesteps=ts, vocab_dim=args.vocab_dim, vocab_size=vocab_size).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    flag = 0

    try:
        for train_index in range(total_iters):
            model.train()
            train_batch = torch.LongTensor(series_np[:, train_index:train_index + ts]).cuda()
            logits = model.forward(train_batch)

            prob = F.softmax(logits[:, -1, :], dim=1).detach().cpu().numpy()
            cumulative_sum_inplace(prob, cumul_np, bs, vocab_size)

            barrier_gpu_ready.wait()

            barrier_cpu_done.wait()

            logits = logits.transpose(1, 2)
            label = torch.from_numpy(series_np[:, train_index + 1:train_index + ts + 1].copy()).cuda()
            train_loss = F.cross_entropy(logits[:, :, -1], label[:, -1].long())
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (train_index + 1) >= (total_iters * 0.1 * flag):
                print(f'[Progress:{flag * 10:>3.0f}%] Current Bit-Rate: {train_loss.item() / np.log(2):.10f} bps')
                flag += 1

    finally:
        stop_flag.value = True
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    print(f'[Progress:{100}%] Decompression finished.')

    with open(args.output, 'wb') as fout:
        fout.write(bytearray(series_np.reshape(-1).tolist()))

    if last:
        series = np.zeros(last, dtype=np.uint8).astype('int')
        f = open(temp_file + '.last', 'rb')
        bitin = arithmeticcoding_fast.BitInputStream(f)
        dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
        prob = np.ones(vocab_size) / vocab_size
        cumul = np.zeros(vocab_size + 1, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob * 10000000 + 1)

        for j in range(last):
            series[j] = dec.read(cumul, vocab_size)
        with open(args.output, 'ab') as fout:
            fout.write(bytearray(series.tolist()))
        bitin.close()
        f.close()
    
    return model

def _init_environment(args):
    if not args.prefix:
        filename = os.path.basename(args.input)
        args.prefix = filename.split('.')[0]
    if not args.tempdir:
        args.tempdir = '{}_{}_bs{}_ts{}_vd{}'.format(
                args.model, args.prefix, args.batch_size, args.timesteps, 
                args.vocab_dim)

    if os.path.exists(args.tempdir):
        shutil.rmtree(args.tempdir)
    os.mkdir(args.tempdir)
    return args.tempdir + '/temp_file'

def main_compress(args):
    t1 = time.time()
    torch.cuda.reset_peak_memory_stats()
    temp_file = _init_environment(args)
    with open(args.input, 'rb') as f:
        series = np.frombuffer(f.read(), dtype=np.uint8)
    train_data = strided_app(series, args.timesteps + 1, 1)
    total_num = len(train_data)
    model = None
    
    if total_num % args.batch_size == 0:
        model = compress_chunk(args, temp_file, series, train_data, None)
    else:
        ini_num = total_num // args.batch_size * args.batch_size
        model = compress_chunk(args, temp_file, series[:ini_num + args.timesteps], train_data[:ini_num], series[ini_num:])

    f = open(args.output, 'wb')
    f.write(struct.pack('<Q', len(series)))
    for i in range(args.batch_size):
        f_in = open(temp_file + '.' + str(i), 'rb')
        byte_str = f_in.read()
        byte_str_len = len(byte_str)
        var_int_encode(byte_str_len, f)
        f.write(byte_str)
        f_in.close()

    if total_num % args.batch_size != 0:
        f_in = open(temp_file + '.last', 'rb')
        byte_str = f_in.read()
        byte_str_len = len(byte_str)
        var_int_encode(byte_str_len, f)
        f.write(byte_str)
        f_in.close()
    f.close()
    shutil.rmtree(args.tempdir)

    t2 = time.time()
    
    original_size = os.stat(args.input).st_size
    compressed_size = os.stat(args.output).st_size
    compression_time = t2 - t1
    gpu_memory = torch.cuda.max_memory_allocated() // 1024  # KB
    
    # 计算模型性能
    model_stats = {'flops': 'N/A', 'params': 'N/A', 'latency': 'N/A'}
    if model is not None:
        model_stats = calculate_model_stats(model, (args.batch_size, args.timesteps), args.vocab_size, device='cuda')
    
    data_name = os.path.basename(args.input)
    
    print_compression_results(data_name, original_size, compressed_size, compression_time, model_stats, gpu_memory)


def main_decompress(args):
    t1 = time.time()
    torch.cuda.reset_peak_memory_stats()
    
    temp_file = _init_environment(args)

    f = open(args.input, 'rb')
    len_bytes = f.read(8)
    len_series = struct.unpack('<Q', len_bytes)[0]
    info_dict = {'len_series': len_series}
    for i in range(args.batch_size):
        f_out = open(temp_file + '.' + str(i), 'wb')
        byte_str_len = var_int_decode(f)
        byte_str = f.read(byte_str_len)
        f_out.write(byte_str)
        f_out.close()

    f_out = open(temp_file + '.last', 'wb')
    byte_str_len = var_int_decode(f)
    byte_str = f.read(byte_str_len)
    f_out.write(byte_str)
    f_out.close()
    f.close()

    model = None
    if (info_dict['len_series'] - args.timesteps) % args.batch_size == 0:
        model = decompress_chunk(args, temp_file, info_dict, 0)
    else:
        last_length = (info_dict['len_series'] - args.timesteps) % args.batch_size + args.timesteps
        model = decompress_chunk(args, temp_file, info_dict, last_length)
    shutil.rmtree(args.tempdir)
    t2 = time.time()

    compressed_size = os.stat(args.input).st_size
    decompressed_size = os.stat(args.output).st_size
    decompression_time = t2 - t1
    gpu_memory = torch.cuda.max_memory_allocated() // 1024  # KB
    
    model_stats = {'flops': 'N/A', 'params': 'N/A', 'latency': 'N/A'}
    if model is not None:
        model_stats = calculate_model_stats(model, (args.batch_size, args.timesteps), args.vocab_size, device='cuda')
    
    data_name = os.path.basename(args.input)
    
    print_decompression_results(data_name, compressed_size, decompressed_size, decompression_time, model_stats, gpu_memory)

def add_shared_args(parser):
    parser.add_argument('input', type=str, help='Input file.')
    parser.add_argument('output', type=str, help='Output file.')
    parser.add_argument('--batch_size', '-b', type=int, default=512, help='Sample size in one batch')
    parser.add_argument('--timesteps', '-t', type=int, default=16, help='The number of history symbols')
    parser.add_argument('--vocab_dim', '-d', type=int, default=32, help='The dimension of vocab.')
    parser.add_argument('--vocab_size', type=int, default=256, help='The size of vocab.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use.')
    parser.add_argument('--prefix', '-p', type=str, help='Prefixes of files')
    parser.add_argument('--tempdir', '-T', type=str, help='Temporary folder name.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--wd', type=float, default=1e-7, help='Weight decay.')
    parser.add_argument('--seed', type=int, default=0, help='Random seeds.')
    parser.add_argument('--model', '-m', type=str, default='fade', choices=list(MODEL_REGISTRY.keys()), help='Model architecture')
    parser.add_argument('--num_workers', '-w', type=int, default=8, help='Number of worker processes for parallel encoding/decoding')

def main():
    parser = argparse.ArgumentParser(description='Compression and Decomporession Process of FADE.')
    subparsers = parser.add_subparsers(dest='command', required=True, help='Sub-commands')

    parser_c = subparsers.add_parser('c', help='Compress a file')
    add_shared_args(parser_c)
    parser_d = subparsers.add_parser('d', help='Decompress a file')
    add_shared_args(parser_d)

    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    print('Running %s' % ' '.join(sys.argv))

    starttime = time.time()
    if args.command == 'c':
        main_compress(args)
    elif args.command == 'd':
        main_decompress(args)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
