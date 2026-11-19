"""Run real, paid evaluations explicitly. No Confident AI login/upload required."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path

# Set before importing DeepEval. Metric.measure runs locally with the judge API;
# no cloud reporting, tracing, or private transcript upload is configured.
os.environ['DEEPEVAL_TELEMETRY_OPT_OUT'] = 'YES'
os.environ['DEEPEVAL_UPDATE_WARNING_OPT_OUT'] = 'YES'

from rag import MIN_SIMILARITY, TOP_K
from service import ChatService, corpus_fingerprint

ROOT = Path(__file__).resolve().parent


def retrieval_scores(expected, sources):
    headings = [s['heading'].split(' > ')[-1] for s in sources]
    return dict(top1=bool(expected and any(h in expected for h in headings[:1])),
                top4=bool(expected and any(h in expected for h in headings[:4])),
                fallback=not sources if not expected else None)


def grade(question, answer, expected_output, model):
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric
    from deepeval.test_case import LLMTestCase
    context = [s['text'] for s in answer.sources]
    test = LLMTestCase(input=question, actual_output=answer.text,
                       expected_output=expected_output, retrieval_context=context)
    classes = [AnswerRelevancyMetric]
    if context:
        classes += [FaithfulnessMetric, ContextualRelevancyMetric]
    results = []
    for cls in classes:
        metric = cls(model=model, threshold=0.7, async_mode=False, include_reason=True)
        try:
            metric.measure(test)
            results.append(dict(name=metric.__name__, score=metric.score,
                                threshold=metric.threshold, passed=metric.is_successful(),
                                reason=metric.reason))
        except Exception as exc:
            results.append(dict(name=metric.__name__, error=type(exc).__name__))
    return results


def summary(cases):
    positives = [c for c in cases if c['expected_headings']]
    negatives = [c for c in cases if not c['expected_headings']]
    def metric(rows, key):
        return dict(hits=sum(bool(c.get('retrieval', {}).get(key)) for c in rows), total=len(rows))
    names = sorted({m['name'] for c in cases for m in c.get('metrics', []) if 'score' in m})
    means = {}
    for name in names:
        values = [m['score'] for c in cases for m in c.get('metrics', [])
                  if m['name'] == name and m.get('score') is not None]
        if values:
            means[name] = dict(mean=sum(values) / len(values), count=len(values))
    return dict(top1=metric(positives, 'top1'), top4=metric(positives, 'top4'),
                fallback=metric(negatives, 'fallback'), deepeval=means,
                errors=sum(bool(c.get('error')) for c in cases),
                metric_errors=sum('error' in m for c in cases for m in c.get('metrics', [])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deepeval', action='store_true', help='Run paid model-judge checks too')
    parser.add_argument('--limit', type=int, default=0, help='0 runs the whole dataset')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be non-negative')
    from openai import OpenAI
    service = ChatService(OpenAI(timeout=30, max_retries=1))
    path = ROOT / 'eval_cases.json'
    dataset = json.loads(path.read_text(encoding='utf-8'))['cases']
    judge = os.getenv('EVAL_MODEL', 'gpt-4o-mini') if args.deepeval else None
    results = []
    for case in (dataset[:args.limit] if args.limit else dataset):
        print(f"Evaluating {case['id']}…", flush=True)
        result = dict(case)
        try:
            answer = service.ask(case['question'])
            result['answer'] = asdict(answer)
            result['retrieval'] = retrieval_scores(case['expected_headings'], answer.sources)
            result['metrics'] = grade(case['question'], answer, case['expected_output'], judge) if judge else []
        except Exception as exc:
            result['error'] = type(exc).__name__  # never write raw API errors/credentials
        results.append(result)
        report = dict(schema_version=1, timestamp=datetime.now(timezone.utc).isoformat(),
                      corpus_sha256=corpus_fingerprint(), dataset_sha256=sha256(path.read_bytes()).hexdigest(),
                      answer_model=service.model, judge_model=judge, embedding_model=service.index.model,
                      threshold=MIN_SIMILARITY, top_k=TOP_K, deepeval_enabled=args.deepeval,
                      deepeval_version=version('deepeval') if args.deepeval else None,
                      evaluated_cases=len(results), dataset_cases=len(dataset),
                      summary=summary(results), cases=results)
        output = ROOT / 'reports' / 'latest.json'
        output.parent.mkdir(exist_ok=True)
        temp = output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        temp.replace(output)
    print(json.dumps(report['summary'], indent=2))
    return int(bool(report['summary']['errors'] or report['summary']['metric_errors']))


if __name__ == '__main__':
    raise SystemExit(main())
