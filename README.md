# DQN_Multiagent_RL

## 1) Agents-Landmarks
```shell
DQN_Multiagent_RL
    /rspt
```

### 프로젝트 주제 및 목표
K x K 그리드 환경에서 N개의 랜드마크 세트에 도달하기 위해 협력해야 하는 N개의 로봇(에이전트)이 목표에 잘 도달하도록 강화학습으로 해결

- Multi agent Learning 을 이해하기 위해
    [Deep Q-learning (DQN) for Multi-agent Reinforcement Learning (RL)](https://github.com/mohammadasghari/dqn-multi-agent-rl) Technical Report 를 참고하여 학습하는 것이 목표

- 그대로 코드를 사용하지 않고, Keras 기반의 코드를 Pytorch로 변경하여 진행

- 위 Technical Report에 설명된 2가지 예시중 Agent-Landmarks 예시를 통해 Multi-agent DQN으로 문제해결

- Reward Function 에 따른 학습결과 비교와 알고리즘 (DQN, DDQN)에 따른 학습결과 비교를 진행

@TODO



## 2) Retrieval of surgical phase transitions
```shell
DQN_Multiagent_RL
    /surgical_rspt
```
### 프로젝트 주제 및 목표
수술영상의 수술단계를 구분하기 위해 단계가 transition 되는 부분을 강화학습으로 탐색

- [Retrieval of surgical phase transitions using reinforcement learning; Yitong Zhang et al; MICCAI 2022](https://arxiv.org/abs/2208.00902) 논문을 구현하는 것이 목표

- 수술단계를 인식하기 위해 비디오 프레임을 각 단계에 맞도록 분류하는 Frame based Video Classification Task가 존재

- 하지만 기존의 프레임 기반의 메소드는 프레임 기반으로 인식 하므로 Noisy Transition이 일어날 가능성이 크게 존재

- 이에 강화학습을 사용하여 Agents가 비디오를 탐색하며 Phase 가 변하는 부분에 대해 Retrieval 하도록 학습진행



@TODO
