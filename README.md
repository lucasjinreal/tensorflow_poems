![PicName](http://ofwzcunzi.bkt.clouddn.com/c6NquwsEIMWg8xVm.png)

# 2017-3-22 重磅更新，推出藏头诗功能
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

# Install tensorflow_poems

* 安装要求：
```
tensorflow 1.0
python3.5
all platform
```

* 安装作诗机器人， 简单粗暴，一顿clone：
```
git clone https://github.com/jinfagang/tensorflow_poems.git
```
由于数据大小的原因我没有把数据放到repo里面，大家🏠我的QQ： 1195889656 或者微信： jintianiloveu 我发给你们把，顺便给我们的项目点个赞哦！～

* 使用方法：
```
# for poem train
python3 main.py -w poem --train
# for lyric train
python3 main.py -w lyric --train

# for generate poem
python3 main.py -w poem --no-train
# for generate lyric
python3 main.py -w lyric --no-train

```

* 参数说明
`-w or --write`: 设置作诗还是创作歌词，poem表示诗，lyric表示歌词
`--train`: 训练标识位，首次运行请先train一下...
`--no-train`: 生成标识位

训练的时候有点慢，有GPU就更好啦，最后gen的时候你就可以看到我们牛逼掉渣天的诗啦！

这是它做的诗：

```
龙舆迎池里，控列守龙猱。
几岁芳篁落，来和晚月中。
殊乘暮心处，麦光属激羁。
铁门通眼峡，高桂露沙连。
倘子门中望，何妨嶮锦楼。
择闻洛臣识，椒苑根觞吼。
柳翰天河酒，光方入胶明
```
感觉有一种李白的豪放风度！

这是它作的歌词：

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


# Author & Cite
This repo implement by Jin Fagang.
(c) Jin Fagang.
Blog: [jinfagang.github.io](https://jinfagang.github.io)
BlogNewest: [lewisjin.oschina.io](https://lewisjin.oschina.io)
