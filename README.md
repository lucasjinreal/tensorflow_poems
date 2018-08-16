# LiBai AI Composer

> An ai powered automatically generats poems in Chinese.

很久以来，我们都想让机器自己创作诗歌，当无数作家、编辑还没有抬起笔时，AI已经完成了数千篇文章。现在，这里是第一步....


# Updates

#### 2018-8-16

We are now officially announced a new project started: **StrangeAI School** - An artificial intelligence learning school and advanced algorithm exchange platform! What we believed in is: AI should made to change people's life, rather than controlled by Gaint Companies.
Here you can get some previews about our projects: http://ai.loliloli.pro (strangeai.pro availiable soon)

#### 2018-3-12

**tensorflow_poems**来诈尸了，许久没有更新这个项目，不知不觉已经有了上千个star，感觉大家对这个还是很感兴趣，在这里我非常荣幸大家关注这个项目，但是我们不能因此而停止不前，这也是我来诈尸的目的。我会向大家展示一下我最新的进展，首先非常希望大家关注一下我倾心做的知乎专栏，人工智能从入门到逆天杀神以及每周一个黑科技，我们不仅仅要关注人工智能，还有区块链等前沿技术：

- 人工智能从入门到逆天杀神(知乎专栏)： https://zhuanlan.zhihu.com/ai-man
- 每周一项目黑科技-TrackTech(知乎专栏):  https://zhuanlan.zhihu.com/tracktech
If you want talk about AI, visit our website (for now):  http://ai.loliloli.pro (strangeai.pro availiable soon)
 , **subscribe** our WeChat channel: 奇异人工智能学院

#### 2017-11-8

貌似距离上一次更新这个repo已经很久了，这段时间很多童鞋通过微信找到了我，甚至包括一些大佬。当时这个项目只是一个练手的东西，那个时候我的手法还不是非常老道。让各位踩坑了。现在**李白**强势归来。在这次的更新中增加了这些改进：

- 对数据预处理脚本进行了前所未有的简化，现在连小学生都能了解了
- 训练只需要运行train.py，数据和预训练模型都已经备好
- 可以直接compose_poem.py 作诗，这次不会出现死循环的情况了。

#### 2017-6-1 ~~可能是最后一次更新~~

我决定有时间的时候重构这个项目了，古诗，源自在下骨子里的文艺之风，最近搞得东西有点乱，所以召集大家，对这个项目感兴趣的欢迎加入扣扣群：
```
 292889553
```


#### 2017-3-22 重磅更新，推出藏头诗功能

一波小更新，下面的问题已经解决了：
* 训练完成作诗时出现一直不出现的情况，实际上是陷入了一直作诗的死循环，已修复
* 新增pretty print功能，打印出的古诗标准，接入第三方APP或者其他平台可以直接获取到标准格式的诗词
* Ternimal disable了tensorflow默认的debug信息
  最后最后最重要的是： **我们的作诗机器人（暂且叫李白）已经可以根据你的指定的字作诗了哦！！**
  欢迎大家继续来踩，没有star的快star！！保持更新！！永远开源！！！
  让我们来看看李白做的藏头诗吧：

```
# 最近一直下雨，就作一首雨字开头的吧
雨霁开门中，山听淮水流。
落花遍霜霰，金壶横河湟。
年年忽息世，径远谁论吟。
惊舟望秋月，应柳待晨围。
人处山霜月，萧萧广野虚。

# 李白人工智能作诗机器人的作者长得比较帅，以帅开头做一首吧
帅主何幸化，自日兼春连。
命钱犯夕兴，职馀玄赏圣。
君有不知益，浮于但神衍。
（浓浓的怀才不遇之风...）
```

![PicName](http://ofwzcunzi.bkt.clouddn.com/VMBUVeqLjlXA6cUJ.png)


# Quick Start

using **LiBai** is very simple:

```
git clone https://github.com/jinfagang/tensorflow_poems
# train on poems
python3 train.py
# compose poems
python3 compose_poem.py
```

When you kick it off, you will see something like this:

![](https://i.loli.net/2018/03/12/5aa5fd903c041.jpeg)





# 它已经不仅仅能够作古诗，还能模仿周杰伦创作歌词！！

这是2017-03-9更新的功能，模仿周杰伦歌曲创作歌词，大家先来感受一下它创作的歌词：

```
我的你的她
蛾眉脚的泪花
乱飞从慌乱
笛卡尔的悲伤
迟早在是石板上
荒废了晚上
夜你的她不是她
....
```

怎么说，目前由于缺乏训练文本，导致我们的AI做的歌词有点....额，还好啦，有那么一点忧郁之风，这个周杰伦完全不是一种风格呀。
然而没有关系，目前它训练的文本还太少，只有112首歌，在这里我来呼吁大家一起来整理 **中国歌手的语料文本！！！**
如果你喜欢周杰伦的歌，可以把他的歌一首一行，每首歌句子空格分开保存到txt中，大家可以集中发到我的邮箱：
[jinfagang19@163.com](http://mail.163.com/)
相信如果不断的加入训练文本我们的歌词创作机器人会越来越牛逼！当然我会及时把数据集更新到github上，大家可以star一下跟进本项目的更新。

# 阅遍了近4万首唐诗



```
龙舆迎池里，控列守龙猱。
几岁芳篁落，来和晚月中。
殊乘暮心处，麦光属激羁。
铁门通眼峡，高桂露沙连。
倘子门中望，何妨嶮锦楼。
择闻洛臣识，椒苑根觞吼。
柳翰天河酒，光方入胶明。
```

这诗做的很有感觉啊，这都是勤奋的结果啊，基本上学习了全唐诗的所有精华才有了这么牛逼的能力，这一般人能做到？
本博客讲讲解一些里面实现的技术细节，如果有未尽之处，大家可以通过微信找到我，那个头像很神奇的男人。闲话不多说，先把github链接放上来，这个作诗机器人我会一直维护，如果大家因为时间太紧没有时间看，可以给这个项目star一下或者fork，
我一推送更新你就能看到，主要是为了修复一些api问题，tensorflow虽然到了1.0，但是api还是会变化。
把星星加起来，让更多人可以看到我们创造这个作诗机器人，后期会加入更多牛逼掉渣天的功能，比如说押韵等等。

![PicName](http://ofwzcunzi.bkt.clouddn.com/m6fvfm6s0aZzVoni.png)

# Copyright

This repo implement by Jin Fagang. Using this under Apache License.

```
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

