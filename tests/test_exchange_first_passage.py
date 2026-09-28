"""Tests for the exchange-first-passage endpoint, incl. a brute-force check."""

from __future__ import annotations

import random
from fractions import Fraction

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def parse_fraction(text: str) -> Fraction:
    n, d = text.split("/")
    return Fraction(int(n), int(d))


def fmt(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def brute_force_first_passage(payload: dict):
    """Exact first-passage inference by explicit exchange-sequence enumeration.

    Every step picks one of the n*m (molecule site, bath site) pairs, so each
    of the (n*m)**steps pick sequences is equally likely; sequences are
    enumerated recursively with the actual heavy-site sets tracked. The first
    hit step of a prefix is the earliest t with X_t >= threshold; once set it
    never changes even if the count later falls below. Every sequence passing
    all observation filters adds one to the joint count and to the bin of its
    first hit step (or to the "never" count when no prefix reaches the
    threshold). Returns the joint probability of the observations and the
    conditional first-hit-step distribution plus the never-hit probability.
    """
    n, m, h, b, steps = (payload[k] for k in ("n", "m", "h", "b", "steps"))
    threshold = payload["threshold"]
    filters = {
        obs["step"]: (obs["lower"], obs["upper"])
        for obs in payload.get("observations", [])
    }
    z = 0
    hits: dict = {}
    never_count = 0

    def walk(t, mol_heavy, bath_heavy, first_hit):
        nonlocal z, never_count
        x = len(mol_heavy)
        if first_hit is None and x >= threshold:
            first_hit = t
        if t in filters:
            lower, upper = filters[t]
            if not lower <= x <= upper:
                return
        if t == steps:
            z += 1
            if first_hit is None:
                never_count += 1
            else:
                hits[first_hit] = hits.get(first_hit, 0) + 1
            return
        for i in range(n):
            for j in range(m):
                i_heavy = i in mol_heavy
                j_heavy = j < bath_heavy  # bath sites 0..bath_heavy-1 heavy
                if i_heavy == j_heavy:
                    walk(t + 1, mol_heavy, bath_heavy, first_hit)
                elif i_heavy:  # heavy isotope leaves the molecule
                    walk(t + 1, mol_heavy - {i}, bath_heavy + 1, first_hit)
                else:  # heavy isotope enters the molecule
                    walk(t + 1, mol_heavy | {i}, bath_heavy - 1, first_hit)

    walk(0, frozenset(range(h)), b, 0 if h >= threshold else None)
    probability = Fraction(z, (n * m) ** steps)
    if z == 0:
        return probability, None, None
    distribution = {
        step: Fraction(count, z) for step, count in sorted(hits.items())
    }
    return probability, distribution, Fraction(never_count, z)


def basic_payload():
    return {
        "n": 2,
        "m": 2,
        "h": 1,
        "b": 1,
        "steps": 2,
        "threshold": 2,
        "observations": [{"step": 1, "lower": 0, "upper": 2}],
    }


def test_starting_at_or_above_threshold_hits_at_step_zero():
    # n = m = 1 with the heavy isotope on the molecule: h = 1 >= threshold,
    # so the first hit time is 0 for every trajectory even though the count
    # oscillates 1, 0, 1, 0 afterwards (falls below do not move it).
    resp = client.post(
        "/v1/exchange-first-passage",
        json={
            "n": 1,
            "m": 1,
            "h": 1,
            "b": 0,
            "steps": 3,
            "threshold": 1,
            "observations": [],
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "first_hit_distribution": [{"step": 0, "probability": "1/1"}],
        "never_probability": "0/1",
    }


def test_threshold_zero_always_hits_at_step_zero():
    payload = basic_payload()
    payload.update(threshold=0, steps=3)
    resp = client.post("/v1/exchange-first-passage", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "first_hit_distribution": [{"step": 0, "probability": "1/1"}],
        "never_probability": "0/1",
    }


def test_unreachable_threshold_is_never():
    # n = m = 2, h = b = 0: no heavy isotope exists at all, so the count
    # stays at 0 forever and threshold 1 is never reached.
    resp = client.post(
        "/v1/exchange-first-passage",
        json={
            "n": 2,
            "m": 2,
            "h": 0,
            "b": 0,
            "steps": 3,
            "threshold": 1,
            "observations": [],
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "first_hit_distribution": [],
        "never_probability": "1/1",
    }


def test_unconditional_two_step_law_hand_computed():
    # From x = 1 with n = m = 2 and total 2: up/down numerators are both 1
    # out of n*m = 4 pairs, stay is 2. Reaching 2 first happens at step 1 on
    # the up transition (1/4); reaching it first at step 2 requires staying
    # at 1 then going up (2/4 * 1/4 = 1/8); the remaining 5/8 never reach 2
    # within two steps.
    resp = client.post(
        "/v1/exchange-first-passage", json=basic_payload()
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "first_hit_distribution": [
            {"step": 1, "probability": "1/4"},
            {"step": 2, "probability": "1/8"},
        ],
        "never_probability": "5/8",
    }


def test_absorbing_hit_on_uniform_bath():
    # n = 1, m = 2, h = 0, b = 1: at x = 0 one of the two bath sites is heavy,
    # so the up numerator is 1 out of 2 pairs and stay is 1. Once at x = 1
    # the single molecule site is heavy and no heavy bath site remains, so
    # the count stays at 1 forever. First hit: step 1 with 1/2, step 2 with
    # 1/4 (stay then up), never with 1/4.
    resp = client.post(
        "/v1/exchange-first-passage",
        json={
            "n": 1,
            "m": 2,
            "h": 0,
            "b": 1,
            "steps": 2,
            "threshold": 1,
            "observations": [],
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "first_hit_distribution": [
            {"step": 1, "probability": "1/2"},
            {"step": 2, "probability": "1/4"},
        ],
        "never_probability": "1/4",
    }


def test_fall_below_after_hit_keeps_first_time_under_observation():
    # Observing X_2 = 1 keeps three equally weighted paths (total 12/16):
    # 1 -> 1 -> 1 (stay,stay, weight 4, never hits 2), 1 -> 2 -> 1 (up then
    # down, weight 4, first hits at step 1 even though X_2 fell back to 1)
    # and 1 -> 0 -> 1 (down then up, weight 4, never reaches 2). Hence the
    # joint probability is 3/4, first hit at step 1 with 1/3 and never 2/3.
    payload = basic_payload()
    payload["observations"] = [{"step": 2, "lower": 1, "upper": 1}]
    resp = client.post("/v1/exchange-first-passage", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "3/4",
        "first_hit_distribution": [{"step": 1, "probability": "1/3"}],
        "never_probability": "2/3",
    }


def test_observation_before_impossible_still_conditions():
    # threshold = 1 is already reached at step 0; observing X_1 = 2 has
    # probability 1/4 (only the up transition), and given it the first hit
    # remains step 0.
    payload = basic_payload()
    payload.update(steps=1, threshold=1)
    payload["observations"] = [{"step": 1, "lower": 2, "upper": 2}]
    resp = client.post("/v1/exchange-first-passage", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/4",
        "first_hit_distribution": [{"step": 0, "probability": "1/1"}],
        "never_probability": "0/1",
    }


def test_zero_probability_omits_distribution():
    # From x = 2 with total 2 the chain must move down, so X_1 = 2 is
    # impossible; no first-hit distribution is returned at all.
    payload = basic_payload()
    payload.update(h=2, b=0, steps=1, threshold=1)
    payload["observations"] = [{"step": 1, "lower": 2, "upper": 2}]
    resp = client.post("/v1/exchange-first-passage", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"probability": "0/1"}


def test_step_zero_observation_excluding_h_is_impossible():
    payload = basic_payload()
    payload["observations"] = [{"step": 0, "lower": 2, "upper": 2}]
    resp = client.post("/v1/exchange-first-passage", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"probability": "0/1"}


def test_steps_zero():
    # steps = 0 with h < threshold: the initial state alone is observed and
    # has not reached the threshold, so never carries all mass.
    resp = client.post(
        "/v1/exchange-first-passage",
        json={
            "n": 3,
            "m": 2,
            "h": 2,
            "b": 1,
            "steps": 0,
            "threshold": 3,
            "observations": [],
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "first_hit_distribution": [],
        "never_probability": "1/1",
    }

    # h >= threshold with steps = 0: first hit at step 0.
    resp = client.post(
        "/v1/exchange-first-passage",
        json={
            "n": 3,
            "m": 2,
            "h": 2,
            "b": 1,
            "steps": 0,
            "threshold": 2,
            "observations": [],
        },
    )
    assert resp.json() == {
        "probability": "1/1",
        "first_hit_distribution": [{"step": 0, "probability": "1/1"}],
        "never_probability": "0/1",
    }


def test_later_observations_apply_after_absorption():
    # Observing X_2 = 2 keeps the single path 1 -> 1 -> 2 (stay then up,
    # weight 2*1 = 2): the chain cannot stay at the top boundary x = 2, so
    # 1 -> 2 -> 2 is impossible. The joint probability is 2/16 = 1/8 and
    # the sole surviving path first hits at step 2.
    payload = basic_payload()
    payload["observations"] = [{"step": 2, "lower": 2, "upper": 2}]
    resp = client.post("/v1/exchange-first-passage", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["probability"] == "1/8"
    assert body["first_hit_distribution"] == [
        {"step": 2, "probability": "1/1"}
    ]
    assert body["never_probability"] == "0/1"


def test_randomized_matches_brute_force():
    rng = random.Random(20260928)
    for _ in range(40):
        n = rng.randint(1, 3)
        m = rng.randint(1, 3)
        steps = rng.randint(0, 4)
        obs_count = rng.randint(0, 3)
        observations = []
        for step in rng.sample(range(steps + 1), min(obs_count, steps + 1)):
            lower = rng.randint(0, n)
            upper = rng.randint(lower, n)
            observations.append(
                {"step": step, "lower": lower, "upper": upper}
            )
        payload = {
            "n": n,
            "m": m,
            "h": rng.randint(0, n),
            "b": rng.randint(0, m),
            "steps": steps,
            "threshold": rng.randint(0, n),
            "observations": observations,
        }
        resp = client.post("/v1/exchange-first-passage", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        probability, distribution, never_probability = brute_force_first_passage(
            payload
        )
        assert parse_fraction(body["probability"]) == probability
        if probability == 0:
            assert "first_hit_distribution" not in body
            assert "never_probability" not in body
            continue
        got = {
            peak["step"]: parse_fraction(peak["probability"])
            for peak in body["first_hit_distribution"]
        }
        assert got == distribution
        assert parse_fraction(body["never_probability"]) == never_probability
        # Ascending by step, reduced fractions, none zero, normalized.
        steps_list = [peak["step"] for peak in body["first_hit_distribution"]]
        assert steps_list == sorted(steps_list)
        for peak in body["first_hit_distribution"]:
            value = parse_fraction(peak["probability"])
            assert peak["probability"] == fmt(value)
            assert value > 0
        assert never_probability >= 0
        total = sum(got.values(), Fraction(0)) + never_probability
        assert total == 1


def test_large_request_stays_exact_and_fast():
    payload = {
        "n": 40,
        "m": 40,
        "h": 20,
        "b": 20,
        "steps": 80,
        "threshold": 30,
        "observations": [
            {"step": 20, "lower": 15, "upper": 25},
            {"step": 40, "lower": 10, "upper": 30},
            {"step": 60, "lower": 0, "upper": 40},
            {"step": 80, "lower": 18, "upper": 35},
        ],
    }
    resp = client.post("/v1/exchange-first-passage", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    probability = parse_fraction(body["probability"])
    assert 0 < probability < 1
    assert body["probability"] == fmt(probability)
    hit_steps = [peak["step"] for peak in body["first_hit_distribution"]]
    assert hit_steps == sorted(hit_steps)
    assert all(0 <= s <= 80 for s in hit_steps)
    total = Fraction(0)
    for peak in body["first_hit_distribution"]:
        value = parse_fraction(peak["probability"])
        assert peak["probability"] == fmt(value)
        assert value > 0
        total += value
    never_probability = parse_fraction(body["never_probability"])
    assert body["never_probability"] == fmt(never_probability)
    assert never_probability >= 0
    assert total + never_probability == 1


@pytest.mark.parametrize(
    "mutation",
    [
        # n out of range / wrong type / missing
        lambda p: p.update(n=0),
        lambda p: p.update(n=41),
        lambda p: p.update(n=2.0),
        lambda p: p.update(n="2"),
        lambda p: p.update(n=True),
        lambda p: p.pop("n"),
        # m out of range / wrong type / missing
        lambda p: p.update(m=0),
        lambda p: p.update(m=41),
        lambda p: p.update(m="2"),
        lambda p: p.pop("m"),
        # h out of range / wrong type / missing
        lambda p: p.update(h=-1),
        lambda p: p.update(h=3),  # h > n
        lambda p: p.update(h=1.5),
        lambda p: p.update(h="1"),
        lambda p: p.pop("h"),
        # b out of range / wrong type / missing
        lambda p: p.update(b=-1),
        lambda p: p.update(b=3),  # b > m
        lambda p: p.update(b=False),
        lambda p: p.pop("b"),
        # steps out of range / wrong type / missing
        lambda p: p.update(steps=-1),
        lambda p: p.update(steps=81),
        lambda p: p.update(steps=2.5),
        lambda p: p.update(steps="2"),
        lambda p: p.pop("steps"),
        # observations missing / too many / wrong type
        lambda p: p.pop("observations"),
        lambda p: p.update(
            observations=[{"step": 0, "lower": 0, "upper": 0}] * 9
        ),
        lambda p: p.update(observations={"step": 0}),
        lambda p: p.update(observations=[None]),
        # duplicate observation steps
        lambda p: p.update(
            observations=[
                {"step": 1, "lower": 0, "upper": 1},
                {"step": 1, "lower": 1, "upper": 2},
            ]
        ),
        # observation step / bounds out of range or wrong type
        lambda p: p["observations"][0].update(step=3),  # step > steps
        lambda p: p["observations"][0].update(step=-1),
        lambda p: p["observations"][0].update(step=1.0),
        lambda p: p["observations"][0].update(step="2"),
        lambda p: p["observations"][0].update(upper=3),  # upper > n
        lambda p: p["observations"][0].update(lower=-1),
        lambda p: p["observations"][0].update(lower=2, upper=1),  # lower > upper
        lambda p: p["observations"][0].update(upper="2"),
        lambda p: p["observations"][0].update(lower=None),
        # missing / unknown observation fields
        lambda p: p["observations"][0].pop("lower"),
        lambda p: p["observations"][0].pop("step"),
        lambda p: p["observations"][0].update(extra=1),
        # threshold out of range / wrong type / missing
        lambda p: p.update(threshold=-1),
        lambda p: p.update(threshold=3),  # threshold > n
        lambda p: p.update(threshold=1.5),
        lambda p: p.update(threshold="2"),
        lambda p: p.update(threshold=True),
        lambda p: p.pop("threshold"),
        # unknown request field
        lambda p: p.update(extra=1),
    ],
)
def test_invalid_payloads_return_422(mutation):
    payload = basic_payload()
    mutation(payload)
    resp = client.post("/v1/exchange-first-passage", json=payload)
    assert resp.status_code == 422


def test_malformed_json_returns_422():
    resp = client.post(
        "/v1/exchange-first-passage",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
