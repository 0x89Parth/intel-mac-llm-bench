#!/usr/bin/env python3
"""A/B test prompt lookup on the same Qwen code edit and explanation used on M2."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from statistics import median
import subprocess
import tempfile
import time
import urllib.request

from compare_avx2 import DEFAULT_AVX2_ENGINE, ENVIRONMENT, binary_sha256, thermal
from compare_inference import DEFAULT_MODEL, PROJECT, free_port, request

RESULTS = PROJECT / 'results'
MODEL_SHA256 = "1d9614638d18024d0fbb36575a15f1302a3adf044df10345688ec4f6e1c4ff32"
SOURCE = r'''def summarize_orders(orders):
    totals = {}
    for order in orders:
        customer = order["customer"]
        amount = order["amount"]
        if customer not in totals:
            totals[customer] = 0
        totals[customer] += amount
    return totals


def top_customers(totals, limit=5):
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return ranked[:limit]


def format_report(ranked):
    lines = []
    for customer, amount in ranked:
        lines.append(f"{customer}: {amount:.2f}")
    return "\n".join(lines)'''
RENAME = (
    'Rename the first function from summarize_orders to total_by_customer. '
    'Its new signature must be exactly "def total_by_customer(orders):". '
    'Return the complete revised Python file. Leave all three function bodies and all other names unchanged.'
)
EXPLANATION = (
    'Explain to a junior programmer why sorting a list in place can surprise the caller, '
    'and give a small concrete Python example. Use around 100 words.'
)
REQUEST = dict(n_predict=256, temperature=0, seed=42, return_tokens=True,
               cache_prompt=False, stream=False)
# Leave room for the verification token in the laptop's 32-token prompt batch.
LOOKUP = ['--spec-type', 'ngram-simple', '--spec-draft-n-max', '16',
          '--spec-ngram-simple-size-m', '16']
COMMON = ['-dev', 'MTL0', '-ngl', 'all', '--fit', 'off', '--load-mode', 'mmap',
          '-t', '3', '-tb', '3', '-b', '32', '-ub', '32', '-c', '2048',
          '-np', '1', '--cache-ram', '0', '--no-warmup', '--no-mmproj',
          '--reasoning', 'off', '--offline', '--no-webui', '--cors-origins', 'localhost',
          '--host', '127.0.0.1', '-lv', '4']


def rename_passes(content, expected):
    source = re.sub(r'^\s*```(?:python)?\s*\n|\n```\s*$', '', content).strip()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    if ast.dump(tree) != ast.dump(ast.parse(expected)):
        return False
    # Evaluate the trusted reference only after verifying the generated AST is identical.
    namespace = {}
    exec(compile(expected, '<verified-rename>', 'exec'), namespace)
    rows = [{'customer': 'A', 'amount': 2}, {'customer': 'B', 'amount': 4},
            {'customer': 'A', 'amount': 1}]
    totals = namespace['total_by_customer'](rows)
    return (totals == {'A': 3, 'B': 4} and namespace['top_customers'](totals, 1) == [('B', 4)]
            and namespace['format_report']([('B', 4), ('A', 3)]) == 'B: 4.00\nA: 3.00')


def run_block(engine, model, baseline, order, repetitions, tasks, expected, config):
    port = free_port()
    url = f'http://127.0.0.1:{port}'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    args = [str(engine), '-m', str(model), *COMMON, '--port', str(port)]
    if not baseline:
        args.extend(LOOKUP)
    rows = []
    before = thermal()
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen(args, cwd=engine.parent, env=ENVIRONMENT, stdout=log,
                                stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline and proc.poll() is None:
                try:
                    if request(opener, url + '/health').get('status') == 'ok':
                        break
                except OSError:
                    pass
                time.sleep(.25)
            else:
                raise RuntimeError('Benchmark server failed to become ready')
            for rep in range(repetitions + 1):
                for name in order:
                    payload = dict(config, prompt=tasks[name])
                    start = time.monotonic()
                    answer = request(opener, url + '/completion', payload, timeout=150)
                    elapsed = time.monotonic() - start
                    if not rep:
                        continue
                    timings, tokens = answer['timings'], answer['tokens']
                    if not tokens or timings['predicted_n'] < 1 or timings['prompt_n'] < 2:
                        raise ValueError(f'Incomplete generation or prompt reuse in {name}')
                    rows.append({'task': name, 'rep': rep, 'timings': timings,
                                 'wall_seconds': elapsed, 'token_count': len(tokens),
                                 'tokens_sha256': hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest(),
                                 'content': answer['content'], 'stop_type': answer.get('stop_type'),
                                 'rename_passed': rename_passes(answer['content'], expected) if name == 'rename' else None})
        except Exception:
            log.seek(0)
            print(log.read().decode(errors='replace')[-2500:], flush=True)
            raise
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        log.seek(0)
        hardware = [line for line in log.read().decode(errors='replace').splitlines()
                    if 'offloaded' in line and 'layers to GPU' in line]
    if not any('25/25 layers to GPU' in line for line in hardware):
        raise ValueError(f'Full Radeon offload was not confirmed: {hardware}')
    return {'mode': 'baseline' if baseline else 'lookup', 'offload': hardware,
            'thermal_before': before, 'thermal_after': thermal(), 'trials': rows}


def summarize(blocks, tasks):
    summary = {}
    for task in tasks:
        task_rows = [(block['mode'], row) for block in blocks for row in block['trials'] if row['task'] == task]
        metrics = summary[task] = {}
        for mode in ('baseline', 'lookup'):
            rows = [row for arm, row in task_rows if arm == mode]
            speeds = [row['timings']['predicted_per_second'] for row in rows]
            metrics[mode] = {'median_output_tps': median(speeds), 'range_output_tps': [min(speeds), max(speeds)],
                             'median_wall_s': median(row['wall_seconds'] for row in rows),
                             'draft_tokens': sum(row['timings'].get('draft_n', 0) for row in rows),
                             'accepted_draft_tokens': sum(row['timings'].get('draft_n_accepted', 0) for row in rows)}
        metrics['ratio'] = metrics['lookup']['median_output_tps'] / metrics['baseline']['median_output_tps']
        metrics['all_tokens_equal'] = len({row['tokens_sha256'] for _, row in task_rows}) == 1
        if task == 'rename':
            metrics['all_renames_passed'] = all(row['rename_passed'] for _, row in task_rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, default=DEFAULT_AVX2_ENGINE)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--output', type=Path, default=RESULTS / 'intel-qwen-prompt-lookup.json')
    parser.add_argument('--repetitions', type=int, default=2, help='timed requests per workload in each block')
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('at least one repetition required')
    engine, model = args.engine.resolve(strict=True), args.model.resolve(strict=True)
    template = ("<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n"
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")
    tasks = {'rename': template.format(f"Here is the current Python file:\n\n{SOURCE}\n\n{RENAME}"),
             'explain': template.format(EXPLANATION)}
    expected = SOURCE.replace('def summarize_orders(', 'def total_by_customer(')
    config = REQUEST
    report = {'start_utc': datetime.now(timezone.utc).isoformat(), 'mode_args': {'baseline': [], 'lookup': LOOKUP},
              'engine_sha256': binary_sha256(engine), 'model': model.name, 'model_sha256': binary_sha256(model),
              'cpu': subprocess.check_output(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
              'server_args': COMMON, 'request': config, 'prompts': tasks, 'expected_rename': expected,
              'warmups_per_task_per_block': 1, 'repetitions_per_task_per_block': args.repetitions, 'blocks': []}
    if report['model_sha256'] != MODEL_SHA256:
        parser.error('expected the same Qwen Q4_K_M weights as the original workload')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        partial = args.output.with_suffix('.json.tmp')
        partial.write_text(json.dumps(report, indent=2) + '\n')
        partial.replace(args.output)
    for index, baseline in enumerate((True, False, False, True)):
        order = list(tasks) if index in (0, 2) else list(reversed(tasks))
        print(f"Starting block {index + 1} ({'baseline' if baseline else 'lookup'})", flush=True)
        block = run_block(engine, model, baseline, order, args.repetitions, tasks, expected, config)
        report['blocks'].append(block)
        save()
        print([(row['task'], round(row['timings']['predicted_per_second'], 2), row['rename_passed'])
               for row in block['trials']], flush=True)
    report['summary'] = summarize(report['blocks'], tasks)
    save()
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
