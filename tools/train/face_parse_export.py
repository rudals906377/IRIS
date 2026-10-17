"""사진 모드 정밀 얼굴 분할 모델 만들기: BiSeNet 얼굴 파싱(zllrunning/face-parsing.PyTorch, MIT) → web/public/models/face-parse-512.onnx

  python face_parse_export.py <face_parsing 코드 폴더의 부모> <79999_iter.pth> <출력.onnx>

- 코드(model.py·resnet.py)와 가중치(79999_iter.pth)는 공식 저장소(HairFastGAN 설치본에도 같은 파일: models/face_parsing,
  pretrained_models/BiSeNet/face_parsing_79999_iter.pth). model.py 의 .cuda() 는 지우고 쓴다.
- 입력 RGB 0~1(1×3×512×512, 정규화는 모델 안), 출력 [머리카락, 얼굴(피부·눈썹·눈·귀·코·입), 목] 확률(1×3×512×512)
- 가중치는 fp16 으로 저장하고 불러올 때 fp32 로 되돌리는 Cast 를 끼운다(계산 fp32, 파일 53MB → 26MB, 출력 차이 0.0005 이하)
"""
import sys

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.utils.model_zoo as mz
from onnx import TensorProto, helper, numpy_helper

mz.load_url = lambda *a, **k: {}  # 백본 사전학습 가중치는 어차피 전체 가중치로 덮는다
sys.path.insert(0, sys.argv[1])
from face_parsing.model import BiSeNet  # noqa: E402


class Wrap(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        p = torch.softmax(self.net((x - self.mean) / self.std)[0], dim=1)
        # 부위 번호: 1 피부 2·3 눈썹 4·5 눈 6 안경 7·8 귀 9 귀걸이 10 코 11 입 12·13 입술 14 목 15 목걸이 16 옷 17 머리카락 18 모자
        hair = p[:, 17:18]
        face = p[:, 1:6].sum(1, keepdim=True) + p[:, 7:9].sum(1, keepdim=True) + p[:, 10:14].sum(1, keepdim=True)
        neck = p[:, 14:15]
        return torch.cat([hair, face, neck], 1)


def main() -> None:
    net = BiSeNet(n_classes=19)
    net.load_state_dict(torch.load(sys.argv[2], map_location="cpu"))
    m = Wrap(net).eval()
    tmp = sys.argv[3] + ".fp32"
    torch.onnx.export(m, torch.rand(1, 3, 512, 512), tmp, input_names=["image"], output_names=["parts"], opset_version=17)
    g_model = onnx.load(tmp)
    g = g_model.graph
    inits, casts = [], []
    for init in list(g.initializer):
        if init.data_type == TensorProto.FLOAT and np.prod(init.dims) > 16:
            h = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "_h")
            inits.append(h)
            casts.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT))
            g.initializer.remove(init)
    g.initializer.extend(inits)
    for c in reversed(casts):
        g.node.insert(0, c)
    onnx.save(g_model, sys.argv[3])
    print("저장:", sys.argv[3])


if __name__ == "__main__":
    main()
