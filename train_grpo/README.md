# 介绍

在gsm8k上面基于qwen1.5B复现一下deepseek的R1流程，本项目使用的是open-R1的代码：https://github.com/huggingface/open-r1

使用的资源是4张40G的A100

更多讲解见：https://zhuanlan.zhihu.com/p/22797726638

# 结果
![image](https://github.com/Mryangkaitong/deepseek-r1-gsm8k/blob/main/log/log.png)

笔者训练的全部日志见log/trainer_state.json

可以看到accuracy_reward、format_reward都在增加，而且思维链确实在不断增加。在qwen1.5B上训练完后在gsm8k上最终的acc为74.29。

下面是一个例子

```
<think> Let's break down the problem step by step. First, we need to find Dora's father's age. We know that Dora's father's age is eight more than twice Dora's age. Since Dora is 15 years old, we can calculate Dora's father's age as follows: Father's age = 2 * Dora's age + 8 = 2 * 15 + 8 = 30 + 8 = 38 years old. Next, we need to find Dora's mother's age. We know that Dora's mother is four years younger than Dora's father. So, Dora's mother's age = Father's age - 4 = 38 - 4 = 34 years old. Now, we can calculate the total combined age of Dora, her father, and her mother. Total combined age = Dora's age + Father's age + Mother's age = 15 + 38 + 34 = 87 years old. </think><answer>87</answer>
```

# 开始

（1）训练开始：

```
sh run_test.sh
```

（2）日志可视化

```
python visualization.py
```
（3）推理

```
python infer.py
```
