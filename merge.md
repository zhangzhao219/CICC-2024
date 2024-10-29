# 模型集成的模型选择

## 背景和原理

模型集成是提升模型效果的最简单方法之一

对于分类问题，原始的模型输出的结果是取不同类别的Softmax输出分数，选择分数最大的为当前类别，而模型集成就是获取多个模型的Softmax输出分数并相加（或取平均），然后再选择分数最大的作为当前类别

在前期实验中，如果集成12个左右的模型，在最强的单模型的效果为0.38的情况下，集成的结果可以达到0.43，简单尝试抓了一下内鬼（删除某些模型会不会让结果更好）没有成功，即看起来模型越多效果越好

因此从实验记录中找出了0.34以上的模型都进行了推理，目前总共44个模型，仍在训练中，预计最终的模型总数在50左右。

不过一起集成44个模型得到的结果仅有0.39左右，因此这里面一定有内鬼，甚至数量可能还不少


## 任务：探索用哪些模型进行集成可以在验证集上达到更好的效果

代码：```merge.py```

数据：```result```文件夹里面的```val_data_*.csv```

最优的方式就是遍历，尝试各种组合以找出最优的结果，不过这个时间复杂度是无法承受的

例如44个模型，选取n个模型进行集成就有C44^n种可能性，即n为2-44时的总数为C44^2 + C44^3 + C44^4 +......+C44^44 = 很大很大的数字，实际运行中，下面的代码

```python
def combine(temp_list):
    end_list = []
    for i in range(2, len(MERGE_FILE_LIST)):
        print(i)
        temp_list2 = []
        for c in combinations(temp_list, i):
            temp_list2.append(list(c))
        end_list.extend(temp_list2)
    return end_list
```

当```len(MERGE_FILE_LIST)=7```时就已经无法在几十秒内得到结果了，更不用说后面还得集成去算（运行946次集成大概为1分钟左右）

因此需要设计一个启发式算法，可以尽量在少尝试的情况下得到更好的结果（感觉0.45+应该是没有问题的）
