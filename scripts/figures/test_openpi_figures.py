"""옮겨 적은 마스크·시점 규칙과 결과 수치가 원 레포와 맞는지, SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import openpi_figures as o


def _idx(seq, kind):
    return [i for i, (k, _) in enumerate(seq) if k == kind]


@pytest.mark.parametrize("pi05", [False, True])
def test_mask_information_flow(pi05):
    seq = o.tokens(pi05)
    m = o.attn_mask(seq)
    img, act = _idx(seq, "img"), _idx(seq, "act")
    # 이미지·언어는 행동을 보지 않는다
    assert not any(m[i][j] for i in img for j in act)
    # 행동은 모든 토큰을 보고, 행동 블록 안은 양방향이다
    assert all(m[i][j] for i in act for j in range(len(seq)))
    # 이미지끼리는 양방향
    assert all(m[i][j] for i in img for j in img)


def test_state_block_only_in_pi0():
    s0, s5 = o.tokens(False), o.tokens(True)
    st = _idx(s0, "state")
    assert len(st) == 1 and not _idx(s5, "state")
    m = o.attn_mask(s0)
    # 상태 토큰은 행동을 보지 않는다
    assert not any(m[st[0]][j] for j in _idx(s0, "act"))


def test_flow_times():
    ts = o.euler_times()
    assert len(ts) == 10 and ts[0] == 1.0 and abs(ts[-1] - 0.1) < 1e-6
    # Beta(1.5, 1) 는 잡음 쪽(t→1)에 더 몰린다
    assert o.beta_pdf(0.9) > o.beta_pdf(0.1)


def test_libero_average():
    vals = [v for _, v in o.LIBERO]
    assert abs(sum(vals) / len(vals) - o.LIBERO_AVG) < 1e-9


@pytest.mark.parametrize("name,fn", o.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
