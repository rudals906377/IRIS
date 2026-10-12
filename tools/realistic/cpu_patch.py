"""CPU 검증용(여기 클라우드에는 GPU 가 없다): 코드에 박힌 cuda 호출을 CPU 로 돌린다. 학교 GPU 서버에서는 쓰지 않는다."""
import torch
from torch import nn

if not torch.cuda.is_available():
    _load = torch.load

    def _patched_load(*a, **k):
        ml = k.get("map_location")
        if ml is None or "cuda" in str(ml):
            k["map_location"] = "cpu"
        return _load(*a, **k)

    torch.load = _patched_load
    torch.Tensor.cuda = lambda self, *a, **k: self
    nn.Module.cuda = lambda self, *a, **k: self

    def _fix_dev(d):
        return "cpu" if d is not None and "cuda" in str(d) else d

    _t_to = torch.Tensor.to
    _m_to = nn.Module.to

    def _tensor_to(self, *a, **k):
        a = tuple(_fix_dev(x) if isinstance(x, (str, torch.device)) else x for x in a)
        if "device" in k:
            k["device"] = _fix_dev(k["device"])
        return _t_to(self, *a, **k)

    def _module_to(self, *a, **k):
        a = tuple(_fix_dev(x) if isinstance(x, (str, torch.device)) else x for x in a)
        if "device" in k:
            k["device"] = _fix_dev(k["device"])
        return _m_to(self, *a, **k)

    torch.Tensor.to = _tensor_to
    nn.Module.to = _module_to
    for name in ("randn", "zeros", "ones", "empty", "tensor", "rand", "full", "arange", "linspace"):
        _f = getattr(torch, name)

        def _wrap(_f=_f):
            def g(*a, **k):
                if "device" in k:
                    k["device"] = _fix_dev(k["device"])
                return _f(*a, **k)

            return g

        setattr(torch, name, _wrap())
    try:
        import clip

        _cl = clip.load
        clip.load = lambda *a, **k: _cl(*a, **{**k, "device": "cpu"})
    except Exception:  # noqa: BLE001
        pass

if not torch.cuda.is_available():
    # 검증 전용: 일부 코드가 assert torch.cuda.is_available() 로 막는다. 위 패치들이 cuda 호출을 모두 CPU 로 돌리므로 참으로 답한다
    torch.cuda.is_available = lambda: True
    torch.cuda.current_device = lambda: 0
    torch.cuda.device_count = lambda: 1
    torch.cuda.set_device = lambda *a, **k: None
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.empty_cache = lambda *a, **k: None
    torch.cuda.manual_seed = lambda *a, **k: None
    torch.cuda.manual_seed_all = lambda *a, **k: None
    torch.cuda.FloatTensor = torch.FloatTensor
    torch.cuda.LongTensor = torch.LongTensor

if not hasattr(torch, "_iris_cpu_type_patched"):
    torch._iris_cpu_type_patched = True
    _t_type = torch.Tensor.type

    def _tensor_type(self, *a, **k):
        if a and isinstance(a[0], str) and "cuda" in a[0]:
            a = (a[0].replace("torch.cuda.", "torch."),) + a[1:]
        return _t_type(self, *a, **k)

    torch.Tensor.type = _tensor_type
