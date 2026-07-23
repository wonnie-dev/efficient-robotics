# Efficient Robotics 프로젝트 전체 연구·구현 인수인계 문서

## 문서의 목적과 사용 방법

이 문서는 `efficient-robotics` 프로젝트를 데스크탑의 Codex CLI, 노트북, VS Code Remote SSH, 연구실 서버, NVIDIA Isaac Sim 환경에서 이어서 수행하기 위한 전체 인수인계 문서다. 단순한 개요나 한두 페이지짜리 요약본이 아니다. 연구가 처음 RT-2/VLA와 MPC를 결합하려던 단계에서 출발하여, VLM·Scene Graph·MPC 기반 modular framework로 이동하고, 다시 uncertainty-aware relational Scene Graph와 information-seeking MPC를 중심으로 좁혀진 과정 전체를 기록한다.

이 문서에는 다음 내용이 포함된다.

- 1차부터 4차까지의 연구 미팅에서 논의된 방향, 질문, 후보 아이디어, 결정 사항
- 기존에 검토한 RT-2, VLA, VLMPC, Scene Graph, MPC, uncertainty 관련 방향
- 현재 논문이 해결하려는 문제와 연구 질문
- 현재 추천되는 논문 제목, 주제, novelty, contribution
- Covered Basket / Container Active Re-observation 시나리오를 선택한 배경
- 시뮬레이션을 Isaac Sim에서 구현하는 구체적인 순서
- 최종 실물 실험이 확정된 박신규 교수님 연구실의 robot manipulator와 simulation platform을 일치시켜야 한다는 제약. 정확한 robot model은 확인 후 결정 log에 기록한다.
- baseline, ablation, metric, 실험 반복 수, logging 및 reproducibility 요구사항
- 데스크탑, 노트북, 서버 사이에서 Git과 Codex를 사용해 작업을 이어가는 방법
- Codex CLI가 코드를 수정하기 전에 반드시 알아야 하는 금지 사항과 미확정 사항

프로젝트의 목표 venue는 ICRA 2027이며, 논문 제출을 위한 실질적인 작업 마감은 2026년 9월 15일 전후로 잡혀 있다. 과거 계획 파일 중 일부가 “ICRA 2026”으로 작성되어 있지만, 내용상 2026년에 연구와 제출을 진행하여 ICRA 2027에 제출하는 일정이다.

이 문서는 대화 기록, 발표 자료, 연구 계획 문서, 2026년 7월 6일 미팅 녹취, 시나리오 문서, 이메일 캡처를 종합해서 재구성하였다. 4차 미팅은 실제 녹취록을 근거로 상세하게 정리했다. 1차부터 3차 미팅은 발표 자료와 당시 논의 기록을 기반으로 상세하게 재구성한 것이며, 모든 문장이 교수님의 발언을 그대로 옮긴 직접 인용은 아니다. 교수님의 정확한 문구가 필요한 경우에는 반드시 원본 녹취 또는 이메일 캡처를 다시 확인해야 한다.

## Codex CLI가 이 문서를 사용할 때 지켜야 할 원칙

1. 프로젝트 구조나 연구 architecture를 변경하기 전에 이 문서를 처음부터 끝까지 읽는다.
2. “현재 결정 사항”과 “절대 제약”으로 표시된 항목은 사용자가 명시적으로 변경하기 전까지 고정된 요구사항으로 취급한다.
3. 박신규 교수님 연구실의 실제 실험 robot model이 확인되기 전에는 Isaac Sim에서 편하다는 이유로 특정 robot을 임의로 가정하지 않는다.
4. 연구를 단순한 `VLM + MPC 데모`로 축소하지 않는다. 논문의 핵심은 target 및 spatial relation uncertainty가 Scene Graph에 표현되고, 그 uncertainty와 task-failure risk가 MPC action 선택에 실제로 영향을 주는 것이다.
5. 확인된 구현 사실과 아직 검증되지 않은 연구 가설을 구분한다.
6. 코드를 수정하기 전에 repository, Isaac Sim 버전, Python environment, robot asset, USD/URDF/MJCF, GPU, dependency를 먼저 조사한다.
7. 큰 변경 전에 Git checkpoint를 만들고, 작동이 확인된 뒤 다시 commit한다.
8. 실험 command, config, random seed, metric, log, video, failure reason을 저장하여 데스크탑·노트북·서버에서 결과를 재현할 수 있게 한다.
9. Codex 대화 기록이 여러 기기에 자동으로 동일하게 동기화된다고 가정하지 않는다. Git repository와 repository 내부 문서를 프로젝트의 영구 기록으로 사용한다.
10. 주요 설계 결정이 바뀌면 `docs/DECISIONS.md`, `docs/STATUS.md`, 또는 본 문서를 업데이트한다.

---

# 1. 프로젝트 기본 정보, 연구자, 장소, 장비

## 1.1 프로젝트 이름과 성격

프로젝트의 working name은 `efficient-robotics`다.

초기의 목표는 RT-2 스타일의 vision-language-action 모델과 MPC를 결합하여 자연어 명령으로부터 robot action을 생성하고, MPC가 그 action을 수정하거나 안전하게 제한하는 것이었다. 그러나 실제 구현 가능성, 공개 checkpoint의 부재, SO-ARM 101과의 불일치, 대규모 robot dataset 및 training cost 문제 때문에 방향을 수정했다.

현재는 pretrained VLM을 semantic perception과 reasoning에 사용하고, Scene Graph를 dynamic world state로 사용하며, MPC가 uncertainty를 줄이는 active re-observation 및 manipulation action을 선택하는 modular framework를 목표로 한다.

## 1.2 목표 학회와 일정

- 목표 학회: ICRA 2027
- working submission deadline: 2026년 9월 15일 전후
- 주요 실험 결과와 figure는 8월 말 이전에 대부분 고정하는 것이 바람직함
- 9월은 새로운 method를 추가하는 기간이 아니라 paper writing, result explanation, failure analysis, formatting에 사용하는 것이 원칙임

사용자는 ICRA accept를 매우 중요하게 생각한다. 따라서 많은 기능을 얕게 넣기보다 다음을 우선한다.

- 명확한 problem statement
- 기존 연구와 구분되는 한 가지 강한 technical contribution
- 실제 robot 또는 최소한 매우 설득력 있는 closed-loop 실험
- 적절한 baseline과 ablation
- wrong action 감소 및 task success 개선을 직접 보여주는 결과
- 재현 가능한 implementation과 experiment log

## 1.3 사람과 역할

- 고원희: 주 연구자이자 1저자 역할을 맡을 가능성이 높다. 연구 방향 구체화, 논문 조사, Isaac Sim 구현, SO-ARM 101 simulation, 실험 관리, 결과 분석, 논문 작성 담당.
- 이응주 교수님: University of Arizona의 지도교수. 정기적으로 scope를 줄이고, 연구 진행 속도와 논문 일정을 관리하며, 박신규 교수님과의 collaboration을 조율한다.
- 박신규 교수님: KAUST의 연구 협력 교수. robotics, MPC, Scene Graph, uncertainty, 실험 시나리오, 실제 robot validation 및 논문 framing에 관한 피드백을 제공한다.
- 고한솔: 연구 논의 및 VLM 관련 자료 조사에 참여하는 선배 연구자. 구체적 역할은 별도 지시가 있을 때만 확정한다.
- 박신규 교수님 연구실의 학생 또는 collaborator가 simulation 또는 real-robot 실험을 도울 가능성이 있지만, 명시적인 역할 분담 전에는 책임 범위를 임의로 가정하지 않는다.

## 1.4 작업 장소와 컴퓨팅 환경

### 데스크탑

- Windows 11
- Isaac Sim GUI 실행, scene 작성, robot asset 확인, camera 배치, interactive debugging에 사용
- 로컬 GPU 정보가 이전 기록에서 RTX 4070 Super와 RTX 5070으로 서로 다르게 나타난 적이 있으므로 반드시 `nvidia-smi`로 현재 GPU를 확인해야 함

### 노트북

- 코드 편집, 문서 작성, Git 작업, 작은 test, 원격 접속에 사용
- 전체 Isaac Sim GUI 작업을 노트북에서 무리하게 수행할 필요는 없음

### 연구실 서버

- Ubuntu/Linux
- VPN과 VS Code Remote SSH로 접속
- `engr-lee01s.engr.arizona.edu` 서버를 사용한 기록이 있음
- NVIDIA RTX A6000 GPU 6개가 인식된 기록이 있으며 각 GPU는 약 49 GB memory를 보유함
- headless Isaac Sim, batch experiment, VLM inference, randomized episode, 장시간 실험에 적합함

### 실제 robot 환경

- 박신규 교수님 연구실에서 real-robot validation을 수행할 계획
- 실제 사용 가능한 camera, gripper, SO-ARM 101 구성, container 및 cover의 형태는 실험 전 확인 필요
- 새로운 대규모 public dataset을 만드는 것이 연구의 중심은 아님

## 1.5 Robot 및 simulator 절대 조건

- simulator는 NVIDIA Isaac Sim을 사용한다.
- 최종 robot은 박신규 교수님 연구실에서 실제 실험에 사용하는 manipulator와 동일해야 한다.
- 정확한 model, gripper, camera 구성이 확인될 때까지 robot-specific 구현을 확정하지 않고 replaceable interface를 사용한다. SO-ARM 101은 더 이상 필수 최종 platform이 아니다.
- simulation에서 먼저 closed-loop를 검증한 뒤 real robot으로 이동한다.
- 초기 scene은 tabletop manipulation, camera observation, open container, simple object, active viewpoint change를 지원해야 한다.

---

# 2. 연구 방향이 변화한 전체 과정

## 2.1 초기 RT-2 스타일 VLA + MPC 방향

가장 초기에는 RT-2와 같은 VLA 모델이 image와 instruction을 입력받아 action token을 생성하고, MPC가 action을 보정하는 구조를 만들려고 했다.

예상 pipeline은 다음과 같았다.

1. camera image와 자연어 instruction 입력
2. RT-2 스타일 모델이 joint action 또는 tokenized action sequence 생성
3. MPC가 joint limit, rate limit, collision, step size를 고려해 action을 수정
4. robot이 action 수행
5. 다음 frame에서 다시 추론

사용자는 unofficial RT-2 repository, fixed-length action token, small demonstration training loop, offline inference, joint clamp, rate limit, step size 기반 MPC correction을 실험했다.

이 단계에서 확인된 문제는 다음과 같다.

- Google RT-2의 실제 전체 시스템과 checkpoint를 쉽게 사용할 수 없었음
- 공개된 third-party 구현이 SO-ARM 101과 직접 연결되지 않았음
- 진짜 RT-2 수준의 성능을 내려면 대규모 robot dataset과 학습 자원이 필요함
- “VLA가 준 action을 MPC가 조금 제한한다”는 구조만으로는 ICRA contribution이 약할 수 있음
- Windows에서 TorchCodec, ffmpeg, OpenMP, GPU library 문제 등이 반복됨
- WSL/Linux로 이동하면 일부 environment 문제는 해결되지만 novelty 문제는 해결되지 않음

따라서 RT-2 reproduction 자체를 논문의 중심으로 두는 것은 시간과 기술 위험이 너무 컸다.

## 2.2 Full VLA에서 modular VLM + MPC로 이동

지도교수님과 논의한 뒤 full VLA training 대신 pretrained VLM과 MPC를 역할별로 결합하는 방향으로 이동했다.

### VLM 역할

- scene에 어떤 object가 있는지 파악
- instruction에서 target object 추출
- object relation과 affordance 해석
- manipulation에 필요한 semantic goal 또는 candidate target 제안

### Scene Graph 역할

- object identity 유지
- object state 저장
- inside, outside, near, behind, on, occluded_by 같은 relation 저장
- task 진행 상태 기록
- observation과 robot action 이후 world state 업데이트

### MPC 역할

- short-horizon trajectory 및 control 최적화
- collision 및 joint constraint 처리
- current state와 semantic target을 바탕으로 안전한 action 선택
- 새로운 observation이 필요할 때 camera 또는 arm motion 수행

초기 modular pipeline은 다음과 같았다.

User Command → VLM Perception → Scene Graph Update → MPC Input Generation → MPC Control → Robot Execution → Feedback → Scene Graph Correction

## 2.3 Scene Graph를 단순 memory 이상으로 사용해야 한다는 논의

Scene Graph가 단순히 지난 상태를 저장하는 memory만 된다면 논문의 technical contribution이 약할 수 있다는 문제가 제기되었다.

검토한 역할은 다음과 같다.

### Memory-only Scene Graph

- object location, state, relation을 저장
- 이전 action과 task progress 기록
- 구현은 쉬우나 MPC decision에 직접 영향을 주지 않으면 novelty가 약함

### State-aware VLM-to-MPC interface

- Scene Graph의 node와 edge를 MPC가 사용할 target, cost, constraint로 변환
- object state와 relation이 control decision에 직접 관여
- generic module connection보다 강한 연구 방향

### Differentiable Scene Graph state

- Scene Graph를 neural MPC의 differentiable state로 encode
- planning loss가 graph representation에 영향을 주도록 설계
- 기술적으로 강하지만 현재 일정에서 구현 위험이 높음

### Action-conditioned Scene Graph forward model

- 현재 graph와 candidate action을 입력받아 action 후 graph state를 예측
- 미래 graph state를 MPC internal world model로 사용
- 매우 유망하지만 학습 data와 안정적인 prediction model이 필요함

### Uncertainty-propagating Scene Graph

- target object grounding uncertainty를 node에 저장
- spatial relation uncertainty를 edge에 저장
- MPC가 uncertainty와 task risk를 줄이는 action을 선택
- 시나리오와 wrong-action prevention을 연결하기 쉬워 가장 현실적인 핵심 방향으로 발전함

## 2.4 Efficiency 중심에서 uncertainty-aware control 중심으로 이동

초기에는 full VLA를 학습하지 않는 “efficient modular framework”를 novelty로 생각했다. 그러나 off-the-shelf VLM을 사용하여 학습 비용을 줄인다는 점만으로는 ICRA contribution이 충분하지 않을 가능성이 높다는 피드백이 있었다.

그 결과 연구 질문은 다음 문제로 좁혀졌다.

- VLM 또는 grounding model이 어느 object가 target인지 확실하지 않을 수 있음
- object가 basket 안인지 밖인지, 뒤인지 옆인지, 다른 object에 가려졌는지 불확실할 수 있음
- robot이 가장 가능성이 높은 action을 바로 실행하면 wrong object를 잡거나, 잘못된 container를 열거나, 불필요한 manipulation을 할 수 있음
- robot은 uncertainty를 감지하고, final manipulation 전에 uncertainty를 줄이는 action을 선택해야 함

현재의 핵심 연결은 다음과 같다.

Target/Relation Uncertainty → Uncertainty-Aware Scene Graph → Task-Risk-Aware MPC → New Observation → Graph Update → Final Retrieval

---

# 3. 1차 연구 미팅 상세 재구성

## 3.1 1차 미팅의 목적

1차 미팅에서는 VLM, Scene Graph, MPC를 결합한 modular architecture가 ICRA 논문 방향으로 설득력이 있는지 논의했다. Full VLA를 구현하지 않고 pretrained model과 classical/optimization-based controller를 연결하는 현실적인 방향을 제시했다.

## 3.2 발표에서 비교한 네 가지 연구 흐름

### VLA 계열

RT-2, RT-X 등은 vision-language input을 robot action으로 직접 연결한다. 장점은 generality와 end-to-end action generation이지만, 이 프로젝트에서는 다음 위험이 있었다.

- 대규모 robot data 필요
- 높은 training cost
- black-box behavior
- SO-ARM 101에 직접 적용하기 어려움
- 짧은 연구 기간 내 재현 위험

### VLM + MPC 계열

VLMPC 및 유사 연구는 VLM으로 candidate action, cost, trajectory 또는 future video를 평가하고 MPC를 통해 action을 고른다. 이 연구들은 현재 방향과 가까웠지만, explicit한 dynamic Scene Graph를 사용해 long-horizon object state와 relation을 지속적으로 유지하는 것이 핵심으로 보이지 않았다.

### VLM + Optimization 계열

VoxPoser와 ReKep은 language와 vision reasoning을 value map, affordance, keypoint, constraint, numerical cost로 변환한다. 이 흐름은 free-form VLM output을 그대로 robot controller에 보내지 않고 control-friendly representation으로 바꾸는 방법을 보여주었다.

### Scene Graph + Foundation Model 계열

박신규 교수님의 long-horizon manipulation framework를 참고했다. LLM, VLM, Scene Graph, motion planner가 역할을 분리하고, Scene Graph가 object location, state, relation, task progress를 저장한다. 이 구조를 그대로 복제하지 않고 execution layer를 MPC 기반 corrective control로 확장하는 방향을 검토했다.

## 3.3 1차 미팅에서 제안한 pipeline

1. 사용자가 자연어 명령을 입력한다.
2. VLM이 target object, object relation, affordance, goal point를 해석한다.
3. Scene Graph가 object와 relation을 저장하고 현재 world state를 업데이트한다.
4. Scene Graph와 VLM output을 MPC의 target, cost, constraint로 변환한다.
5. MPC가 short-horizon control을 최적화한다.
6. robot이 action을 실행한다.
7. camera와 state feedback으로 Scene Graph를 업데이트한다.
8. task가 끝날 때까지 반복한다.

## 3.4 당시 예상 novelty

- Full VLA training 없이 VLM과 MPC를 결합한 efficient modular framework
- Scene Graph state를 반영하는 state-aware VLM-to-MPC interface
- 실행 후 visual/state feedback을 graph와 controller에 다시 반영하는 closed loop

## 3.5 당시 후보 실험

- 기본 pick-and-place
- cluttered scene manipulation
- target 또는 object state가 변하는 feedback/correction task
- long-horizon multi-step manipulation

당시 생각한 baseline은 VLM-only, MPC-only, VLM+MPC, VLM+Scene Graph+MPC였다.

당시 metric은 task success rate, collision rate, replanning count, Scene Graph update accuracy였다.

## 3.6 1차 미팅에서 제기된 핵심 질문

- VLM + Scene Graph + MPC라는 architecture 자체만으로 ICRA novelty가 충분한가?
- 어떤 task가 Scene Graph의 장점과 MPC의 필요성을 가장 분명하게 보여주는가?
- Scene Graph를 memory로만 둘 것인가, 아니면 MPC target/cost/constraint 생성에 직접 사용할 것인가?
- VLM이 준 point만 따라가는 MPC가 아니라 relation과 task progress를 고려하는 state-aware MPC가 필요한가?
- semantic relation을 numerical control term으로 어떻게 변환할 것인가?

## 3.7 1차 미팅 이후 결론

전체 방향은 가능성이 있지만 generic framework에 머물러 있었다. “세 개의 module을 연결했다”가 아니라 구체적인 failure mode, mathematical interface, control objective가 필요하다는 결론으로 이동했다.

---

# 4. 2차 연구 미팅 상세 재구성

## 4.1 2차 미팅의 초점

2차 미팅과 그 전후 study period에서는 learning-based perception과 MPC를 실제로 어떻게 연결할지 집중했다. VLMPC, neural MPC, OpenVLA, VLA pipeline, residual learning plus MPC, learned dynamics와 analytic dynamics, policy warm-start 등을 검토했다.

## 4.2 MPC라고 부르기 위해 필요한 구성

단순히 VLM에게 action을 물어보고 그 결과를 실행하는 것은 MPC가 아니다. MPC는 적어도 다음 요소를 가져야 한다.

- state representation
- dynamics 또는 state transition model
- finite prediction horizon
- objective/cost function
- control 및 state constraint
- candidate control optimization 또는 sampling
- 첫 action만 실행하고 다음 observation에서 replanning하는 receding horizon 구조

## 4.3 VLM 및 Scene Graph 정보를 MPC에 연결하는 방식

검토한 mapping 예시는 다음과 같다.

- target object position → terminal cost 또는 tracking cost
- obstacle/interference object → collision constraint
- inside/behind/near 같은 relation → geometric relation cost
- task progress → active sub-goal 선택
- open/closed state → action feasibility constraint
- VLM uncertainty → risk cost, stop condition, re-observation trigger 또는 information-gain objective

## 4.4 Scene Graph schema에 대한 논의

Object node 후보 정보:

- object ID
- semantic label
- target probability
- 3D position 또는 pose
- visibility
- open/closed state
- graspability 또는 affordance
- observation timestamp
- confidence 및 uncertainty

Relation edge 후보:

- inside
- outside
- near
- on
- behind
- in_front_of
- occluded_by
- attached_to
- relation probability 및 uncertainty

Graph는 중요한 observation 또는 physical action 이후 반드시 업데이트되어야 한다.

## 4.5 기술적으로 검토한 contribution 후보

- Task-conditioned Scene Graph generation
- Differentiable Scene Graph to MPC state mapping
- Action-conditioned Scene Graph forward model
- Adaptive sparse graph attention
- Temporal graph keyframe compression
- Uncertainty-propagating Scene Graph
- Counterfactual graph reasoning
- Cross-domain transfer to surgical robotics

Differentiable 및 action-conditioned graph는 강하지만 구현 난도가 높았다. Uncertainty-propagating graph는 concrete scenario와 wrong-action prevention을 연결하기 쉬워 점차 중심 후보가 되었다.

## 4.6 2차 미팅에서 확인된 연구 제약

- generic framework alone은 약함
- controller 또는 perception-control interface에서 기술적 contribution이 보여야 함
- efficiency만 novelty로 내세우면 부족함
- 단순 pick-and-place는 uncertainty와 recovery mechanism을 드러내지 못하면 toy demo로 보일 수 있음
- real-robot validation이 중요함
- submission deadline 전에 완성할 수 있도록 scope를 제한해야 함

## 4.7 2차 미팅 이후 남은 질문

- 어떤 종류의 uncertainty를 다룰 것인가?
- uncertainty를 어떻게 측정하고 calibration할 것인가?
- robot motion이 uncertainty를 줄여야 하는 이유가 분명한 scenario는 무엇인가?
- novelty를 새로운 uncertainty algorithm에 둘 것인가, 기존 uncertainty method를 새로운 manipulation problem에 적용할 것인가?

---

# 5. 3차 연구 미팅 상세 재구성

## 5.1 3차 미팅의 초점

3차 미팅 단계에서는 uncertainty-aware manipulation을 중심으로 논문과 개념을 검토했다. KnowNo, VLMPC, Traj-VLMPC, conformal prediction, model confidence, re-observation, stop/ask policy, active sensing 등이 논의되었다.

## 5.2 구분한 uncertainty 종류

### Target object grounding uncertainty

사용자의 instruction이 어느 visible object를 의미하는지 불확실한 상황이다. 예를 들어 비슷한 두 cup이 같은 bowl 근처에 있을 때 어느 cup이 target인지 확실하지 않다.

### Spatial-relation grounding uncertainty

Target이 container 안인지 밖인지, bowl 뒤인지 옆인지, 다른 object에 가려졌는지 불확실한 상황이다.

### Localization uncertainty

Target identity는 알지만 3D position 또는 pose가 불확실하다.

### Grasping uncertainty

Object와 pose는 알지만 grasp 성공 여부가 불확실하다.

### Language ambiguity

명령이 여러 해석을 허용하여 사용자에게 질문해야 하는 상황이다.

최종적으로 target object grounding과 spatial-relation grounding uncertainty를 중심으로 선택했다. Grasping과 localization은 기존 연구가 많고, language ambiguity는 사람에게 질문하기만 해도 줄일 수 있어 robot movement와 MPC의 필요성이 약할 수 있기 때문이다.

## 5.3 검토한 uncertainty mitigation action

- 가장 confidence가 높은 hypothesis를 바로 실행
- confidence가 낮으면 stop
- 사용자에게 질문
- camera 또는 arm viewpoint 변경
- container 열기
- occluding object 이동
- 새로운 evidence로 Scene Graph 업데이트
- manipulation replanning

핵심 요구사항은 uncertainty 감소가 robot action을 필요로 해야 한다는 것이었다. 그렇지 않으면 연구가 LLM/VLM dialogue policy로 끝날 수 있었다.

## 5.4 초기 metric 후보

초기에는 perception quality, uncertainty calibration, control efficiency, task success, wrong action, re-observation count, execution time, graph consistency 등 많은 metric이 검토되었다. 이후 MPC 논문에서 가장 중요한 metric으로 줄여야 한다는 방향으로 변경되었다.

## 5.5 3차 미팅 이후 좁혀진 범위

- target object uncertainty
- spatial relation uncertainty
- robot action으로 uncertainty reduction
- Scene Graph update
- final action 전 MPC decision
- wrong-action prevention

남은 문제는 이 요소가 모두 필요한 concrete scenario를 정하는 것이었다.

---

# 6. 4차 연구 미팅 상세 기록

## 6.1 Scope, dataset, evaluation 논의

4차 미팅에서는 발표 자료에 task, model, dataset, metric이 너무 많아 연구 범위를 줄여야 한다는 논의가 시작되었다.

이응주 교수님은 computer vision conference에서 SOTA 비교를 위해 2~6개의 dataset을 쓰기도 하지만 ICRA에서는 어느 정도가 필요한지 박신규 교수님께 질문했다.

박신규 교수님의 답변 취지는 다음과 같다.

- 최근 robotics에도 ML/AI식 SOTA comparison이 늘고 있음
- 그러나 robotics reviewer가 오래전부터 가장 중요하게 보는 것은 실제 실험임
- simulation에서 성능이 좋아도 real experiment가 없으면 “실험 검증이 부족하다”는 review가 자주 나옴
- 이 프로젝트에서는 관련성이 높은 dataset 또는 evaluation setting 1~2개 정도면 충분할 수 있음
- 5개 dataset은 과도할 가능성이 큼
- 6개의 metric을 전부 사용할 필요가 없음
- VLM이 main contribution이 아니라면 perception metric을 간략화하고 MPC 및 robot behavior 중심 metric을 선택하는 것이 적절함

따라서 프로젝트는 많은 dataset을 모으는 방향보다 randomized simulation episode와 real-robot trial에 집중하는 것으로 정리되었다.

## 6.2 Uncertainty method가 아직 정해지지 않았다는 문제

이응주 교수님은 발표 자료가 어떤 uncertainty를 다룰지는 보여주지만 uncertainty를 방법론적으로 어떻게 측정할지는 아직 정하지 않았다고 지적했다.

결정해야 할 사항은 다음과 같았다.

- uncertainty method 자체에서 novelty를 만들 것인가?
- 기존 method를 manipulation application에 적용할 것인가?
- calibration을 어떻게 할 것인가?
- uncertainty가 threshold를 넘을 때 어떤 action을 선택할 것인가?

KnowNo가 가까운 reference로 언급되었다. KnowNo는 uncertainty가 높으면 사람에게 질문하여 ambiguity를 줄이는 방식이지만, 사람 개입이 필요하고 MPC가 robot을 움직여 uncertainty를 줄이는 구조는 아니므로 그대로 적용하기 어렵다는 의견이 나왔다.

## 6.3 “MPC가 어떻게 uncertainty를 줄이는가?”라는 핵심 질문

박신규 교수님은 MPC를 이용하여 uncertainty를 줄이는 방법을 생각해 보았는지 직접 질문했다.

미팅에서 형성된 구분은 다음과 같다.

- 사용자에게 질문하는 action은 robot이 움직일 필요가 없음
- re-observation은 camera나 sensor를 다른 각도로 이동해야 함
- robot 또는 sensor가 움직여 새로운 data를 얻어야 uncertainty가 감소하는 scenario라면 MPC와 직접 연결됨
- MPC는 더 좋은 observation을 얻기 위한 robot motion을 계획할 수 있음

핵심 chain은 다음과 같다.

Robot Motion → New Sensor Observation → Reduced Target/Relation Uncertainty → Updated Scene Graph → New MPC Decision

## 6.4 간단하고 구체적인 scenario 요청

박신규 교수님은 전문 용어로만 설명하지 말고 실제로 이해할 수 있는 simple and specific scenario를 생각해 보았는지 질문했다.

교수님은 과거 mobile robot object retrieval 연구에서 먼저 scenario를 정하니 학생들이 문제를 더 쉽게 formulation할 수 있었다고 설명했다.

### Mobile robot object retrieval example

사용자가 멀리 있는 물체를 가져오라고 요청한다. LLM이 navigation plan을 정하고 여러 가능한 해석이 있으면 사용자에게 질문한다. Scenario를 먼저 정해 연구 problem을 쉽게 formalize한 사례다.

### Factory pick-and-place example

Workbench 위에 여러 object가 있고 manipulator, gripper, vision system이 있다. Object A를 집어 tray에 놓는 작업에서 grasping uncertainty와 localization uncertainty를 formulation할 수 있다. 그러나 이 분야는 이미 많은 연구가 있어 차별화에 주의해야 한다.

### Basket/container spatial relation example

Robot이 object를 찾으려 하지만 Scene Graph에서 object가 basket 안인지 밖인지 불확실한 상황이다.

구체적인 예:

1. Robot은 target이 lid가 있는 basket 안에 있다고 가정한다.
2. MPC로 lid를 연다.
3. 안에 target이 없다.
4. Scene Graph를 업데이트하거나 다른 observation을 얻는다.
5. 새로운 MPC operation으로 search 또는 manipulation을 이어간다.
6. Physical interaction과 re-observation을 통해 uncertainty를 감소시킨다.

박신규 교수님은 간단한 예제에서 시작해 concrete idea를 만든 뒤 advanced example과 experiment로 확장하는 것이 좋다고 말했다.

## 6.5 Public dataset보다 실험을 우선

교수님의 실무적 우선순위는 다음과 같았다.

1. 실제 또는 simulation experiment를 수행한다.
2. Existing method와 비교하여 성능이 얼마나 개선되는지 보여준다.
3. 필요한 경우 관련 dataset을 보조적으로 사용한다.

Computer vision처럼 5개 또는 10개의 dataset과 수많은 SOTA model을 비교하는 것이 필수는 아니라는 의견이었다.

## 6.6 Overleaf 협업과 일정

- 미팅 때까지 기다리지 말고 Overleaf에 idea를 적어 continuous discussion을 진행
- 이응주 교수님과 원희가 일주일에 1~2회 추가로 discussion하여 scope를 좁힘
- 프로젝트 기간이 약 2~2.5개월로 짧음
- 8월 말까지 substantial result를 준비해야 writing 시간이 확보됨
- 필요하면 3주 간격 미팅을 2주로 앞당길 수 있음

## 6.7 Model selection 논의

박신규 교수님은 model을 선택할 때 당시 성능표나 relevant benchmark를 근거로 설명할 필요가 있다고 말했다.

- Grounding DINO는 grounding에 적합한 후보로 언급됨
- Qwen 계열 VLM이 후보로 논의됨
- 정확한 model/version은 implementation 시점의 최신 evidence와 hardware requirement를 확인해야 함
- 단순히 “가장 유명해서”가 아니라 task relevance, latency, output reliability, reproducibility를 근거로 선택해야 함

## 6.8 4차 미팅의 실제 결과

4차 미팅에서 완성된 algorithm이 결정된 것은 아니다. 다음 방향과 task가 정리되었다.

- 연구 scope를 줄인다.
- 관련성이 높은 evaluation setting 1~2개를 사용한다.
- metric 수를 줄이고 robot/MPC 결과를 중심으로 한다.
- real experiment를 중요하게 고려한다.
- uncertainty measure를 구체적으로 정의한다.
- robot movement가 uncertainty를 줄이는 scenario를 만든다.
- target 및 spatial relation grounding uncertainty를 중심으로 한다.
- basket/container example에서 시작한다.
- Overleaf를 이용해 빠르게 idea를 발전시킨다.

---

# 7. 4차 미팅 이후 시나리오 비교와 선택

## 7.1 고려한 시나리오 후보

### Ambiguous Tabletop Relation

Setup:

- table 위에 cup 2개, bowl 1개, block 1개를 배치
- instruction: “Pick up the cup next to the bowl.”
- 두 cup이 모두 bowl 근처에 있어 target이 ambiguous

장점:

- 구현이 가장 쉬움
- lid, drawer, 복잡한 articulation이 필요 없음
- target grounding, relation 이해, stop/re-observe, wrong pick rate를 테스트 가능

단점:

- robot motion과 MPC가 반드시 필요해 보이지 않을 수 있음
- perception benchmark처럼 보일 위험

### Open Container Relation

Setup:

- 뚜껑 없는 basket, tray, box를 table 위에 배치
- object를 boundary 근처에 두어 inside/outside/near가 single view에서 ambiguous하게 보이도록 함
- instruction: “Pick up the object inside the basket.” 또는 “Pick up the block near the container.”

장점:

- lid opening보다 쉬움
- inside, outside, near relation 테스트 가능
- covered-container scenario의 bridge 역할

단점:

- scene이 너무 명확하면 uncertainty가 약함
- camera angle과 object placement를 신중히 설계해야 함

### Occluded Target Active Viewpoint

Setup:

- target을 bowl, box 또는 다른 object 뒤에 부분적으로 숨김
- initial view에서 behind/next-to relation이 불명확
- wrist camera 또는 arm viewpoint를 변경
- instruction: “Pick up the red block behind the bowl.”

장점:

- re-observation 필요성이 명확함
- MPC가 어느 viewpoint로 이동할지 직접 결정 가능
- multi-view consistency를 uncertainty signal로 사용 가능
- articulated lid보다 구현이 쉬움

단점:

- relation uncertainty를 강조하지 않으면 기존 active perception과 겹침

### Covered Basket / Container Active Re-observation

Setup:

- target이 container 안, 밖, 뒤, near 또는 partially occluded 상태일 수 있음
- initial observation만으로 target location과 relation을 확정하기 어려움
- robot이 viewpoint를 바꾸거나 lightweight cover를 제거하거나 이후 단계에서 lid를 열 수 있음
- instruction: “Find and pick up the target object inside or near the basket.”

장점:

- target grounding uncertainty와 spatial relation uncertainty를 동시에 포함
- Scene Graph update가 의미 있음
- MPC가 final grasp trajectory뿐 아니라 information-gathering action을 담당
- open container에서 시작해 occlusion, cover, lid로 단계적 확장 가능

단점:

- hinge lid는 구현 시간이 많이 듦
- scenario 자체는 novelty가 아니며 algorithm이 paper를 이끌어야 함

### Drawer / Shelf Hidden Object Retrieval

Setup:

- target이 drawer 내부, shelf 위 또는 다른 object 뒤에 있음
- robot이 drawer를 열거나 viewpoint를 변경

장점:

- household long-horizon manipulation story가 강함
- inside, on, behind relation이 명확함

단점:

- Isaac Sim 및 real robot 구현 난도가 높음
- 현재 일정에서 risk가 큼

### Relation-Constrained Placement

Setup:

- robot이 object를 inside, next to, behind relation을 만족하도록 배치

장점:

- spatial relation을 MPC cost/constraint로 바꾸기 쉬움
- MPC contribution이 시각적으로 명확할 수 있음

단점:

- 연구 중심이 target uncertainty에서 goal relation uncertainty로 이동
- object retrieval story와 달라짐

## 7.2 선택한 시작 시나리오

선택한 starting point는 `Covered Basket / Container Active Re-observation`이다.

단, 구현은 가장 쉬운 단계에서 시작해야 한다.

1. Open container
2. inside/outside/near ambiguity
3. partial occlusion
4. active viewpoint change
5. lightweight removable cover
6. core loop가 안정된 뒤에만 hinge lid 검토

## 7.3 선택 이유

- target object uncertainty와 relation uncertainty를 직접 표현함
- physical re-observation이 필요하여 MPC의 역할이 분명함
- observation 이후 Scene Graph update가 필요함
- ambiguous tabletop보다 robotics contribution이 강함
- drawer/shelf보다 구현 위험이 낮음
- 단계적으로 확장 가능함

---

# 8. 박신규 교수님의 시나리오 문서 이메일 피드백

## 8.1 보낸 문서

사용자는 Covered Basket / Container Active Re-observation을 선택한 이유, 초기 simulation plan, alternative scenario를 정리한 PDF를 박신규 교수님께 보냈다. 이응주 교수님과 한솔이형을 참조에 넣었다.

박신규 교수님은 여행 중이어서 확인이 늦었으며 검토 후 연락하겠다고 먼저 답했다.

## 8.2 박신규 교수님의 상세 제안

교수님의 피드백 핵심은 다음과 같다.

### Uncertainty를 반영한 Scene Graph

LLM/VLM으로 Scene Graph를 만들 때 uncertainty를 어떻게 체계적으로 적용할지 고민해야 한다. Graph가 `inside`와 같은 하나의 확정 label만 가지는 것이 아니라 target identity 및 relation에 대한 불확실한 hypothesis를 표현해야 한다.

### Uncertain Scene Graph를 기반으로 MPC action 정의

Uncertainty를 포함한 Scene Graph를 바탕으로 robot의 MPC action을 정하는 방향으로 진행할 수 있다. 즉 uncertainty representation이 단순 visualization이 아니라 control에 실제로 사용되어야 한다.

### Two phases 또는 integrated phase

Active Re-observation과 Covered Container를 두 phase로 나눌 수 있지만, MPC action으로 uncertainty를 줄이는 관점에서 하나의 단계로 통합하는 것도 가능하다.

더 강한 interpretation은 다음과 같다.

- uncertainty를 해결한 후에 별도 MPC가 task를 실행하는 것이 아님
- MPC action 자체가 uncertainty 감소를 목적으로 선택됨

### Autonomous uncertainty reduction

사용자가 특정 object를 찾아 전달하라고 할 때 robot이 framework를 통해 스스로 uncertainty를 줄여가며 정확한 object를 찾을 때까지 진행하는 구조가 적절하다는 의견이었다.

## 8.3 이응주 교수님의 response

이응주 교수님은 phase를 통합하면 framework가 더 깔끔해질 수 있고, 박신규 교수님이 제안한 방향이 모두 좋다고 답했다. ICRA deadline까지 시간이 많지 않으므로 연구팀도 속도를 내겠다고 했다.

## 8.4 사용자의 response

사용자는 다음을 중심으로 연구 방향을 구체화하겠다고 답했다.

- uncertainty를 반영한 Scene Graph
- uncertainty를 줄이기 위한 MPC action
- 박신규 교수님이 Overleaf에 추가할 내용 확인 및 준비

---

# 9. 현재 최종 연구 방향

## 9.1 Research problem

부분 관측 환경에서 language-guided manipulation robot이 특정 object를 찾아야 한다. 그러나 robot은 다음을 확실히 판단하지 못할 수 있다.

- 어느 visible object가 instruction의 target인지
- target이 container 안, 밖, 뒤, near 또는 occluded 상태인지
- initial observation이 충분한지
- 어느 information-gathering action이 가장 안전하고 유용한지

Direct VLM-to-action 방식은 wrong object를 집거나 불필요한 container를 열 수 있다. Deterministic Scene Graph는 uncertainty를 하나의 hard label로 숨길 수 있다. Conventional MPC는 collision-free trajectory는 만들 수 있지만 semantic goal이 불확실하다는 사실을 알지 못할 수 있다.

따라서 semantic uncertainty를 robot action과 연결해야 한다.

## 9.2 Research question

Target object와 spatial relation에 대한 calibrated uncertainty를 dynamic Scene Graph에 표현하고, MPC가 final retrieval 전에 expected task failure를 줄이는 observation 또는 manipulation action을 선택하도록 만들 수 있는가?

## 9.3 Hypothesis

Direct execution, deterministic Scene Graph, fixed re-observation, greedy viewpoint, uncertainty term이 없는 MPC와 비교할 때 proposed method는 다음을 개선할 것이다.

- correct target retrieval success 증가
- wrong object pick 및 premature action 감소
- 불필요한 re-observation/manipulation 감소
- 안전성과 execution efficiency 유지

## 9.4 현재 추천 제목

### 추천 제목

`Uncertainty-Aware Relational Scene Graph MPC for Language-Guided Object Retrieval under Partial Observability`

### 대안 제목

`BeliefGraph-MPC: Information-Seeking Model Predictive Control over Uncertainty-Aware Scene Graphs for Language-Guided Object Retrieval`

최종 제목은 실제 구현된 method와 experiment를 기준으로 정해야 한다. Future belief prediction을 구현하지 않았다면 title에서 과도하게 예측 기능을 암시하면 안 된다.

## 9.5 한 문장 정의

Robot은 target object와 spatial relation의 calibrated uncertainty를 Scene Graph node와 edge에 저장하고, MPC를 이용하여 wrong-action 또는 task-failure risk를 줄이는 re-observation 및 manipulation action을 선택한 뒤 object를 회수한다.

---

# 10. 추천 novelty와 경계

## 10.1 Novelty 1: Task-conditioned node 및 relation-edge uncertainty

Scene Graph는 object node뿐 아니라 relation edge에도 uncertainty를 저장한다.

Object node 예:

- object A가 requested target일 probability
- object B가 distractor일 probability
- existence confidence
- position/pose covariance
- visibility 또는 occlusion probability

Relation edge 예:

- P(target inside basket)
- P(target outside basket)
- P(target behind basket)
- P(target near container)
- P(target occluded_by object B)

Graph는 room 전체의 모든 relation을 완벽하게 만들 필요가 없다. Instruction과 next manipulation decision에 필요한 task-relevant relation을 우선한다.

## 10.2 Novelty 2: Raw VLM confidence가 아닌 calibrated uncertainty

VLM이 “80% 확실하다”고 말하는 verbal confidence를 그대로 사용하면 안 된다. 실제 correctness와 맞는 measurable probability 또는 uncertainty score가 필요하다.

후보 방법:

- held-out calibration set의 temperature scaling
- repeated VLM sampling 또는 ensemble disagreement
- multi-view consistency
- Dirichlet evidence accumulation
- Bayesian update across observations
- conformal prediction 또는 prediction set
- geometry-based relation confidence

Calibration은 단독 main novelty라기보다 MPC가 의미 있는 probability를 사용하도록 만드는 필수 supporting contribution으로 보는 것이 현실적이다.

## 10.3 Novelty 3: Task-risk-aware information-seeking MPC

가장 중요한 controller contribution은 단순 graph entropy 감소가 아니라 wrong action 또는 final task failure probability를 줄이는 action을 선택하는 것이다.

Conceptual MPC objective:

`J = task cost + expected future graph uncertainty + wrong-action risk + collision risk + motion/time cost`

Candidate action:

- wrist camera viewpoint change
- arm motion without grasping
- lightweight cover removal
- occluder movement
- container opening
- final grasp

Action은 task-relevant belief를 개선하고 failure risk를 줄일 것으로 예상될 때 선택된다.

## 10.4 Novelty 4: Re-observation과 final manipulation의 integrated closed loop

두 개의 disconnected system을 만들면 안 된다.

의도한 loop:

1. observation 획득
2. task-conditioned graph belief 생성 또는 update
3. target 및 relation uncertainty 계산
4. candidate action 평가
5. MPC가 control 또는 information-gathering action 선택
6. 첫 action 실행
7. 새로운 observation 획득
8. graph belief update
9. confidence와 risk가 threshold를 만족할 때까지 반복
10. final target retrieval

## 10.5 Optional stronger novelty: Action-conditioned future belief prediction

각 candidate action이 uncertainty를 얼마나 줄일지 action 전에 예측하는 model은 강한 extension이다.

가능한 방법:

- Isaac Sim geometry와 rendering으로 viewpoint visibility 예측
- learned graph transition model
- approximate information-gain model
- action-conditioned relation update model

하지만 이 기능은 core closed-loop가 안정된 후에만 추가해야 한다. 불완전한 prediction model 때문에 전체 시스템이 실패하면 오히려 논문이 약해진다.

## 10.6 단독 novelty로 사용하면 약한 주장

다음 항목만으로 novelty를 주장하면 부족할 가능성이 높다.

- VLM 사용
- Scene Graph 사용
- MPC 사용
- VLM과 MPC 연결
- camera viewpoint 변경
- container 열기
- off-the-shelf model로 효율성 향상
- 일반 pick-and-place 수행

차별점은 uncertainty representation이 mathematical/actionable하고, 그 값이 실제 MPC action과 wrong-action prevention을 바꾸는 데 있다.

---

# 11. Proposed technical architecture

## 11.1 Input

- natural-language instruction
- RGB 또는 RGB-D image
- robot joint state
- camera pose
- current dynamic Scene Graph
- optional task/action history

## 11.2 Perception and grounding layer

역할 기반으로 component를 나눈다.

- VLM: instruction-conditioned object 및 relation reasoning
- open-vocabulary grounding model: named object bounding box
- segmentation model: object mask
- RGB-D geometry: 3D position, container boundary, occlusion, camera-to-world transform

VLM 한 모델에 localization, segmentation, calibration, reasoning, control을 모두 맡기지 않는다.

정확한 model/version은 아직 고정되지 않았다. Grounding DINO 계열과 Qwen 계열 VLM이 후보지만 current benchmark와 hardware를 확인해야 한다.

## 11.3 Scene Graph representation

Object node 예시 field:

- `id`
- `label_distribution`
- `target_probability`
- `position_mean`
- `position_covariance`
- `bounding_box`
- `mask_reference`
- `visible`
- `occluded_probability`
- `container_state`
- `graspable`
- `last_observed_time`
- `source_view_id`

Relation edge 예시 field:

- `source_id`
- `target_id`
- `relation_type`
- `probability`
- `uncertainty`
- `geometric_evidence`
- `semantic_evidence`
- `last_updated_time`

Task state 예시:

- `instruction`
- `target_description`
- `active_subgoal`
- `completion_probability`
- `risk_threshold`
- `observation_budget`

## 11.4 Uncertainty 계산 초기 버전

첫 구현은 다음 signal을 조합할 수 있다.

- grounding confidence
- VLM relation output probability 또는 repeated-sampling agreement
- RGB-D relation geometry score
- multi-view consistency
- small held-out scenario set에서 학습한 calibration parameter

Mutually exclusive relation에 대해서는 normalized probability distribution을 만든다.

예:

Initial observation:

- inside: 0.48
- behind: 0.37
- outside: 0.15

Re-observation 후:

- inside: 0.91
- behind: 0.06
- outside: 0.03

확률뿐 아니라 어떤 evidence로 update되었는지 log에 저장한다.

## 11.5 MPC state 및 cost

MPC state 후보:

- robot joint position 및 velocity
- end-effector pose
- camera pose
- container 및 object geometry
- target/relation belief vector
- active sub-goal
- collision geometry

Cost 후보:

- goal progress
- expected task success
- expected future target uncertainty
- expected future relation uncertainty
- wrong-action risk
- collision 및 joint-limit penalty
- motion effort
- time/path length
- information-gathering action count

초기 controller는 Isaac Sim에서 안정적으로 사용할 수 있는 sampling-based MPC 또는 robust controller로 시작할 수 있다. Exact solver는 확정된 박신규 교수님 연구실 robot model, control interface, simulation stability 확인 후 결정한다.

## 11.6 Final execution threshold

Robot은 다음 조건을 만족할 때만 final grasp를 수행한다.

- target probability가 threshold 이상
- 필요한 relation probability가 threshold 이상
- collision 및 reachability check 통과
- predicted wrong-action risk가 threshold 이하

그렇지 않으면 re-observation 또는 information-gathering action을 선택한다.

---

# 12. Isaac Sim 구현 계획

## 12.1 Phase 0: Repository 및 environment audit

Codex는 feature를 만들기 전에 다음을 확인해야 한다.

- 현재 repository file 목록
- Isaac Sim 버전과 installation path
- Python environment
- `nvidia-smi`
- CUDA 및 driver compatibility
- 박신규 교수님 연구실의 정확한 robot model과 해당 URDF, USD, MJCF, mesh, joint name
- robot scale, joint limit, articulation control
- camera 생성, image capture, headless execution
- 현재 실행 가능한 최소 script

확인 후 Git checkpoint를 만든다.

## 12.2 Phase 1: Minimal deterministic tabletop scene

가장 작은 재현 가능한 scene을 만든다.

- ground plane
- table
- robot model 확정 전 replaceable robot mount/interface, 확정 후 실제 실험과 동일한 manipulator
- fixed 또는 wrist-mounted camera
- open basket 또는 tray 1개
- simple object 2~3개
- target 1개와 distractor 1개 이상

최소 성공 기준:

- robot scale과 joint limit이 정확함
- camera RGB 또는 RGB-D frame 저장 가능
- object pose 접근 가능
- scripted pick trajectory 또는 simple end-effector movement 가능
- GUI와 headless 모두 scene 실행 가능

## 12.3 Phase 2: Controlled relation scene generator

다음 variant를 code로 생성한다.

- target inside container
- target outside container
- target near boundary
- target behind container
- target partially occluded
- target absent from expected container
- visually similar distractor

Randomization 대상:

- object position
- orientation
- distractor count
- occlusion level
- lighting
- camera pose
- container pose

한 번에 모든 것을 randomize하지 말고 factor별로 validation한다.

## 12.4 Phase 3: Perception 및 graph stub

처음부터 heavy VLM을 연결하지 않는다. Ground truth 또는 rule-based perception stub으로 동일한 Scene Graph interface를 먼저 만든다.

필수 output:

- object node list
- relation belief distribution
- target probability
- uncertainty score
- graph JSON
- image와 camera metadata

이 단계에서 controller loop를 독립적으로 test한다. 이후 실제 grounding/VLM component로 교체한다.

## 12.5 Phase 4: Active viewpoint action

Table 주변에 finite set의 safe viewpoint를 정의한다. 초기 active perception controller는 viewpoint 후보 중 하나를 선택한다.

요구사항:

- 모든 viewpoint가 reachable해야 함
- camera pose log 저장
- movement 이후 새 observation 획득
- graph update
- before/after uncertainty 비교
- occluded 또는 invalid view가 episode를 crash시키지 않음

## 12.6 Phase 5: Lightweight cover interaction

Hinge lid보다 먼저 removable lightweight cover를 사용한다.

- cover가 graspable함
- cover를 들어 fixed safe location에 놓음
- container interior를 re-observe
- inside/absent relation update
- search 또는 grasp 계속

Hinge lid는 basic loop가 안정된 뒤 extension으로만 취급한다.

## 12.7 Phase 6: Information-seeking MPC

Fixed 또는 greedy viewpoint를 proposed MPC objective로 교체한다.

MPC가 비교할 항목:

- predicted target/relation uncertainty reduction
- task progress
- movement cost
- collision risk
- wrong-action risk

첫 action만 실행하고 새 observation에서 다시 planning한다.

## 12.8 Phase 7: Real-robot transfer

실제 robot은 simulation에서 검증된 가장 단순한 scenario를 재현한다.

우선순위:

1. Open-container relation ambiguity + active viewpoint
2. Lightweight cover removal + re-observation

Object geometry와 lighting은 처음에는 controlled condition으로 시작한다. Failure와 sim-to-real gap을 숨기지 않고 기록한다.

---

# 13. Baseline 및 ablation

## 13.1 Main baselines

### Direct VLM + Execution

가장 가능성이 높은 target/relation을 선택하고 uncertainty를 무시한 채 바로 action한다. Uncertainty를 무시했을 때 발생하는 wrong action cost를 측정한다.

### Deterministic Scene Graph + MPC

Graph가 relation을 하나의 hard label로 저장한다. Probabilistic graph belief의 필요성을 분리해 검증한다.

### Uncertainty-Aware Graph + Fixed Re-observation

Uncertainty는 알지만 항상 predetermined viewpoint를 사용한다. Intelligent action selection의 효과를 검증한다.

### Uncertainty-Aware Graph + Greedy Viewpoint

Immediate information gain이 가장 큰 view를 고르지만 multi-step 또는 motion/risk cost를 고려하지 않는다. MPC optimization의 효과를 검증한다.

### MPC without Uncertainty/Task-Risk Cost

동일 controller에서 uncertainty 및 wrong-action cost term을 제거한다. Proposed objective의 효과를 검증한다.

### Proposed Full Method

Calibrated target/relation uncertainty, graph update, task-risk-aware information-seeking MPC를 모두 사용한다.

## 13.2 필수 ablation

- relation-edge uncertainty 제거
- target-node uncertainty 제거
- calibration 제거
- MPC uncertainty term 제거
- wrong-action risk term 제거
- camera re-observation만 허용하고 physical interaction 제거
- closed loop 대신 one observation만 사용

## 13.3 Related system comparison 주의사항

대형 external system을 완전히 재현하기 어렵다면 strategy-level baseline을 구현할 수 있다. 다만 reproduced baseline과 conceptual comparison을 논문에서 명확히 구분해야 한다.

---

# 14. Evaluation metrics

## 14.1 Primary Metric 1: Task Success Rate

요청한 정확한 target을 회수하고 정의된 terminal condition을 완료한 episode 비율이다. 가장 중요한 robotics metric이다.

## 14.2 Primary Metric 2: Wrong Action Rate

다음 action을 count한다.

- wrong object pick
- wrong container open
- relation confidence가 충분하지 않은 상태에서 grasp
- irrelevant occluder 이동
- disproven location을 반복 탐색
- task relation을 위반하는 action

정의는 실험 전에 고정해야 한다. Episode당 wrong action 수 또는 전체 high-level action 중 wrong action 비율로 사용할 수 있다.

## 14.3 Primary Metric 3: Information-Gathering Efficiency

가능한 지표:

- 성공 전 re-observation/interaction action 수
- action당 entropy reduction
- meter, second 또는 action당 task-risk reduction
- decision threshold까지 걸린 total time 또는 path length

Main paper에서는 하나의 단순한 정의를 사용하고, 대안은 supplementary에 둘 수 있다.

## 14.4 Secondary metrics

- Expected Calibration Error
- Brier score
- target grounding accuracy
- relation classification accuracy
- Scene Graph entropy before/after action
- collision rate
- planning time per step
- total execution time
- path length
- replanning count
- graph update accuracy
- occlusion level별 success
- distractor count별 success

## 14.5 통계 보고

- randomized episode를 충분히 반복
- training/stochastic sampling이 있으면 최소 3 seed
- real-robot success rate에 confidence interval 보고 가능
- raw episode-level CSV/JSON 보존
- failure와 qualitative example 기록

---

# 15. Experimental scale 및 data strategy

## 15.1 Dataset strategy

새로운 대규모 dataset을 만들 필요는 없다.

Main validation:

- scenario code가 생성한 randomized Isaac Sim episode
- calibration/component validation에 필요한 관련성이 높은 external dataset 1~2개만 선택적으로 사용
- 박신규 교수님 연구실의 real-robot trial

Dataset 수를 늘리기 위해 관련 없는 benchmark를 추가하지 않는다.

## 15.2 Simulation episode target

Compute와 시간에 따라 main scenario 전체에서 수백 episode를 목표로 한다.

추천 구조:

- open container + active viewpoint: main simulation benchmark
- covered container 또는 removable cover: second benchmark
- occluder removal: optional third scenario 또는 supplementary

## 15.3 Real-robot trial target

각 main method 및 baseline, scenario combination마다 최소 10~15 trial을 현실적인 하한으로 보고, 가능하면 20 trial을 목표로 한다. 실제 lab time과 statistical meaning을 고려하여 최종 수를 정한다.

---

# 16. Timeline 및 urgency

연구 계획은 9월 15일 deadline에서 역산한다.

- 5월: 연구 방향, literature review, problem definition, 1차 미팅
- 6월: method 구조, prototype, Scene Graph representation, VLM-MPC interface
- 7월 초: scope, uncertainty, baseline, scenario selection
- 7월 중후반: 작동하는 simulation, main experiment, paper skeleton
- 8월 초: experiment stabilization, ablation, result organization
- 8월 중후반: result freeze, figure, table, full draft
- 9월 초: revision, failure analysis, supplementary, formatting
- 9월 14일: intended submission buffer
- 9월 15일: deadline

이응주 교수님은 사용자가 처음 ICRA paper를 쓰므로 8월 말까지 substantial result가 준비되어야 writing 및 revision 시간이 확보된다고 강조했다.

---

# 17. 추천 repository 구조

```text
efficient-robotics/
├── AGENTS.md
├── README.md
├── docs/
│   ├── PROJECT_CONTEXT.md
│   ├── DECISIONS.md
│   ├── STATUS.md
│   ├── MEETING_NOTES/
│   ├── EXPERIMENT_PROTOCOL.md
│   ├── RUNBOOK.md
│   └── PAPER_CLAIMS.md
├── assets/
│   ├── robots/so_arm_101/
│   ├── containers/
│   └── objects/
├── configs/
│   ├── sim/
│   ├── robot/
│   ├── perception/
│   ├── scene_graph/
│   ├── uncertainty/
│   ├── mpc/
│   └── experiments/
├── src/
│   ├── sim/
│   ├── robot/
│   ├── perception/
│   ├── scene_graph/
│   ├── uncertainty/
│   ├── mpc/
│   ├── integration/
│   └── evaluation/
├── scripts/
│   ├── setup/
│   ├── run_gui.py
│   ├── run_headless.py
│   ├── generate_scenarios.py
│   ├── calibrate_uncertainty.py
│   └── evaluate.py
├── tests/
├── experiments/
│   ├── manifests/
│   └── summaries/
├── results/
│   ├── raw/
│   ├── processed/
│   ├── figures/
│   └── videos/
└── paper/
```

사용자는 과거 다른 프로젝트에서 임의의 `logs` 및 `results` folder를 싫어한 적이 있으므로 최종 naming은 현재 선호를 확인해 정할 수 있다. 다만 robotics research에서 reproducibility를 위해 raw output과 source code를 분리하는 명확한 experiment-output location이 반드시 필요하다.

---

# 18. Logging 및 reproducibility

각 episode에서 저장해야 할 정보:

- Git commit hash
- machine identifier
- Isaac Sim version
- Python/package version
- GPU 정보
- scenario seed
- object 및 camera pose
- natural-language instruction
- grounding output
- action 전후 Scene Graph
- target/relation probability distribution
- selected action 및 candidate cost
- MPC planning time
- collision/reachability status
- episode success/failure reason
- image 또는 video sequence

각 experiment batch는 manifest file을 가져야 한다. Terminal scrollback만 유일한 기록으로 사용하면 안 된다.

---

# 19. Codex가 지켜야 하는 절대 제약과 현재 결정

## 19.1 절대 제약

- 최종 simulator robot은 박신규 교수님 연구실의 실제 실험 manipulator와 동일한 model
- simulator는 Isaac Sim
- generic pick-and-place tutorial이 아님
- target 또는 relation uncertainty가 robot action에 영향을 주어야 함
- Scene Graph는 MPC decision에 영향을 주어야 하며 display용이면 안 됨
- robot은 항상 사람에게 질문하기보다 action으로 uncertainty를 줄여야 함
- real-robot validation이 중요한 목표
- 새 대규모 dataset을 만들지 않음
- 많은 미완성 benchmark보다 좁고 완전한 scope를 사용

## 19.2 현재 결정

- open container와 active viewpoint change부터 시작
- partial occlusion 추가
- complex hinge lid보다 removable cover 우선
- target 및 relation uncertainty를 graph node/edge에 저장
- MPC action selection에 task-failure risk와 expected uncertainty reduction 사용
- primary metric은 task success, wrong action, information-gathering efficiency

## 19.3 아직 미정인 사항

- 정확한 VLM model/version
- uncertainty calibration algorithm
- exact MPC solver
- learned future belief model 또는 geometry-based prediction 사용 여부
- 박신규 교수님 연구실 robot의 정확한 model, gripper, camera configuration
- 박신규 교수님 연구실에서 사용할 실제 hardware
- 최종 paper title

Codex는 미정 항목을 임의로 확정된 것으로 취급하면 안 된다.

---

# 20. Codex CLI에 요청할 즉시 구현 순서

## Step 1: Inspect only

요청 예시:

`AGENTS.md와 docs/PROJECT_CONTEXT.md를 읽고 repository와 Isaac Sim setup을 조사해라. 아직 파일을 수정하지 마라. Robot asset, simulator entry point, Python environment, missing dependency, 현재 실행 가능한 가장 작은 scene을 보고해라.`

## Step 2: Plan과 checkpoint 정의

`SO-ARM 101 open-container active-viewpoint simulation을 위한 implementation plan을 작성해라. 변경할 파일, 추가할 파일, test, expected output, rollback checkpoint를 정의하고 승인 전에는 구현하지 마라.`

## Step 3: Minimal scene 구현

`SO-ARM 101, table, open basket, object 두 개, camera가 있는 가장 작은 deterministic Isaac Sim scene을 구현해라. RGB image와 scene metadata를 한 번 저장하는 script를 추가하고 exact command와 output을 보고해라.`

## Step 4: Scenario generation 추가

`inside, outside, near, behind, partial occlusion variant를 controllable하게 추가하고 각 episode의 ground-truth relation과 config를 저장해라.`

## Step 5: Graph stub 추가

`PROJECT_CONTEXT.md에 정의된 Scene Graph JSON schema를 ground truth 기반으로 먼저 구현하고 viewpoint 및 object movement 후 graph update test를 추가해라.`

## Step 6: Uncertainty와 active observation 추가

`baseline belief distribution과 candidate viewpoint set을 구현하고 simple uncertainty-reduction heuristic으로 view를 선택해라. Action 전후 uncertainty를 log해라.`

## Step 7: Heuristic을 MPC로 교체

`Task-risk-aware MPC objective를 추가하고 direct execution, fixed viewpoint, greedy viewpoint, MPC without uncertainty cost, full method를 비교해라.`

## Step 8: 실제 perception model 통합

`Ground-truth perception stub을 선택한 grounding/VLM component로 교체하되 graph interface와 test는 유지해라. Calibration과 multi-view update를 추가해라.`

---

# 21. 여러 기기에서 Codex 및 작업 기록을 관리하는 방법

## 21.1 Codex가 로컬에 저장하는 기록

Codex CLI는 각 machine의 local `CODEX_HOME` 아래에 session 및 history를 저장한다. 같은 machine에서는 `codex resume`을 사용해 이전 session을 이어갈 수 있다.

그러나 desktop, laptop, server는 서로 다른 local Codex history를 가질 수 있다. Local CLI transcript를 cross-machine project database로 간주하면 안 된다.

## 21.2 공유해야 하는 영구 기록

공유 기록의 중심은 Git repository여야 한다.

Commit 및 push 대상:

- source code
- config
- `AGENTS.md`
- detailed project context
- decision log
- experiment protocol
- lightweight result summary
- environment lock file

일반 Git repository에 commit하지 말아야 할 것:

- password
- access token
- auth file
- large checkpoint
- raw video 전체
- 개인 정보

Large file은 Git LFS, NAS 또는 합의된 storage를 사용한다.

## 21.3 Recommended Git workflow

1. Desktop, laptop, server가 같은 remote repository를 clone한다.
2. 어느 machine에서 작업을 시작하든 먼저 `git pull`한다.
3. 하나의 coherent task마다 branch를 만든다.
4. Codex에게 큰 변경을 요청하기 전에 commit한다.
5. 작동이 확인된 milestone마다 commit한다.
6. 다른 machine으로 이동하기 전에 push한다.
7. 새 machine에서 pull한 뒤 commit hash를 확인한다.

## 21.4 Documentation workflow

Repository에 다음 파일을 유지한다.

- `AGENTS.md`: Codex가 자동으로 읽을 짧고 안정적인 지시
- `docs/PROJECT_CONTEXT.md`: 전체 연구 history와 architecture
- `docs/DECISIONS.md`: 결정 날짜, 이유, alternative, consequence
- `docs/EXPERIMENT_PROTOCOL.md`: episode 및 metric 정의
- `docs/RUNBOOK.md`: desktop/server command
- `docs/STATUS.md`: 현재 완료 상태와 next task

Codex session을 끝내기 전에 `STATUS.md`에 다음을 적게 한다.

- 완료한 작업
- 사용한 exact command
- 변경한 파일
- 통과/실패한 test
- next step
- known issue
- current Git commit

## 21.5 AGENTS.md 사용

Codex는 project hierarchy에서 `AGENTS.md`를 자동으로 읽는다. 이 파일은 짧고 stable하게 유지한다. 전체 research history는 길기 때문에 `docs/PROJECT_CONTEXT.md`에 두고, 첫 prompt에서 반드시 읽도록 지시한다.

## 21.6 Local history backup

CLI transcript가 반드시 필요하면 각 machine의 `.codex` 또는 `CODEX_HOME` 관련 state를 별도 backup할 수 있다. 그러나 authentication file은 Git에 넣지 않는다. Repository만 복사하면 local Codex history까지 복사되는 것은 아니다.

## 21.7 VS Code server 사용

Server에서 실행할 때는 다음 방식이 가장 명확하다.

1. Laptop/desktop에서 VS Code Remote SSH로 server 접속
2. Server에 clone된 동일 repository open
3. `git pull`
4. Server terminal에서 Codex CLI 실행
5. `AGENTS.md`와 `docs/PROJECT_CONTEXT.md`를 읽게 함
6. Headless Isaac Sim 및 batch job 실행
7. 결과 summary와 code를 commit/push
8. Large result/video는 NAS 또는 designated storage에 저장

Laptop local Codex session과 server Codex session은 별개일 수 있으므로 `STATUS.md`와 Git commit이 handoff 역할을 한다.

---

# 22. 최종 논문이 전달해야 할 하나의 story

사용자가 robot에게 특정 object를 가져오라고 요청한다. Object는 ambiguous하거나 partially hidden되어 있으며 basket 또는 container와의 relation이 불확실하다. VLM과 grounding component는 object와 relation hypothesis를 생성한다. System은 이를 task-conditioned probabilistic Scene Graph로 저장한다.

MPC는 target으로 바로 이동하는 trajectory만 계산하지 않는다. Candidate robot motion과 interaction이 task-relevant uncertainty, wrong-action risk, collision risk, effort를 얼마나 줄이는지 평가한다. 필요하면 robot은 viewpoint를 바꾸고, lightweight cover를 제거하거나, occluder를 이동한다. 각 action 뒤에는 새로운 observation을 얻고 Scene Graph를 update하며 다음 MPC decision을 다시 계산한다. Uncertainty와 risk가 충분히 낮을 때만 final grasp를 실행한다.

논문은 direct execution, deterministic graph, fixed re-observation, greedy re-observation, uncertainty cost가 없는 MPC와 비교하여 proposed loop가 correct target retrieval을 높이고 wrong action을 줄인다는 것을 증명해야 한다.

Scenario는 testbed다. Contribution은 uncertainty를 graph에 표현하고, 그 uncertainty와 task risk를 information-seeking MPC에 직접 사용하는 방법이다.

---

# 23. 자료 출처 및 신뢰도 주의사항

이 문서는 다음 자료를 종합하여 작성했다.

- `Efficient VLM-MPC Framework for Robotic Manipulation with Scene Graph-Based World State Tracking` 발표 자료
- 한국어 VLM-MPC 발표 outline
- `Grounded Scene Graphs and Planning-Aware Perception for Neural MPC` internal plan
- ICRA submission schedule 문서
- 2026년 7월 6일 4차 미팅 녹취
- 시나리오 후보 Word 문서
- `Scenario Selection and Initial Experimental Plan` PDF
- 박신규 교수님의 scenario feedback, 이응주 교수님의 response, 사용자의 reply가 포함된 이메일 캡처
- RT-2, SO-ARM 101, Isaac Sim, server, uncertainty, metric, ICRA 방향과 관련된 이전 대화 기록

1차부터 3차까지의 미팅 section은 당시 발표 자료와 discussion history를 바탕으로 상세 재구성한 것이다. 4차 미팅은 transcript 기반이다. 교수님의 정확한 직접 인용이 필요한 논문 또는 meeting record에서는 이 문서 문장을 그대로 quotation으로 쓰지 말고 original transcript 또는 email screenshot을 다시 확인한다.
