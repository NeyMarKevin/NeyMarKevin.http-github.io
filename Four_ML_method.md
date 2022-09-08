# Four methods/stuctures for machine learning

In this blog, four methods will be introduced in machine learning area, Independent Component Analysis (ICA), Locally Linear Embedding (LLE), Autoencoders and Long-short term memory (LSTM)

## Independent Conponent Analysis (ICA)

Let's assume that you have a dog and a cat, today is a rainy weekend, and you have nothing to do so you decided to watch a movie at home.

The movie is splendid, you are really enjoying it, however, it seems like your dog and cat is boring too, so they decide to have a fight with each other to have some fun. So at this time, they are shouting at each other, and that really impacts you to hear the sound of the movie. You know what? Your computer can also hear the sound, and he is annoyed too,(but he cannot speak to you, and he is annoyed because of the movie sound! He is curious to the world and wants to hear more about the fight).

![They are really in a big fight!](https://cdn.jsdelivr.net/gh/NeyMarKevin/pic@main/202209080925072.png)

You are annoyed, your computer is annoying, but there's nothing you and he can do to stop the noise because the cat and dog will fight again and the computer cannot close the movie by itself, so you and your computer are trying to ignore them, and you pay all your attention to the movie, luckily, that really works!! You can hear the characters talking in the movie! In this case, you think: Let these two small devils fight then, I can ignore their noise and enjoy the movie. But your computer failed, he is still suffering from the movie noise because he cannot ignore it, he really wants some help.

In this case, the noise is not disappeared or become small, but in your feelings, the noise disappeared and doesn't influence you to enjoy the movie. Thanks to the brain's help, the brain helps you eliminate most of the noise though it still exists. The noise cannot affect you any more! However, we cannot give the computer a brain (the computer will be unstoppable to control the world with a brain!), so the only thing we can do is develop an algorithm for him.

This type of problem called Blind Source Separation problem (BSS), which refers to the fact that the data we measure (or hear) is a mixture of data from multiple different sources. It should be noted that the data we measure is not a mixture of target data and noise, but a mixture of multiple different target data. (cat and dog's fight is not noise! because your computer wants to hear their barking!!)

So let do your computer a favour and we can help them from the root.

### Sounds add up linearly
We know that sounds add up linearly. What the microphone of your computer or our eardrum picks up is a linear superposition of multiple sound signals. Like the figure shows. So in this case, the sounds we heard can be seen as the combination of different types of sound (dog's woof, cat's mew and human's talking). The only different between the sound you heard and the sound computer heard is the size of sound. 

![The linear sounds add up](https://cdn.jsdelivr.net/gh/NeyMarKevin/pic@main/202209080920238.jpeg)

### Some basic mathematics
We are almost there! To understand what is ICA and how it works, we need a little mathematics (Just a little...)
***
Like we discussed before, the sounds we heard is a combination of different types of sound, we can use a linear vector to represent different types of the sound.
> $$\vec{S}= [x_{dog}, x_{cat}, x_{movie}]^T$$

Also, to show the combination of these types of sounds, we need another martix to do the combination.
> $$\vec{A} = 
> \begin{bmatrix} 
> w_1 & w_2 & w_3\\
> w_4 & w_5 & w_6\\
> w_7 & w_8 & w_9
> \end{bmatrix}
> $$ 

From linear algebra, we can know that we need three equations for a three-degree equation to find a result, which means we need to have a 3 by 1 measurement matrix $\vec{X}$ to record three different places' sounds. And these three places need to be linearly independent.

So now we get:
> $$ \vec{X} = \vec{A}\vec{S}$$
> 
> $\vec{X}$ is the measurement we have, $\vec{A}$ is the superposition parameter matrix. And $\vec{S}$ is what we really want.

We can put the inverse A to the left side to get the result expression.

>$$\vec{A}^{-1}\vec{X} = \vec{S}$$

Luckily, your computer has two mics, so there are three measurement data here (You ears and computer's data)

So what now? actually we want the sepearte sounds in $\vec{S}$, it looks like we need to find $\vec{A}$

$\vec{A}$ is a 3 by 3 matrix to show that the different amplitudes of different sounds, unfortunately, the linear algebra also tells us it's impossible to find a specific solution for a one-degree equation which has two unknown varables.

So it seems like we have reached to a dead end, the solutions are not specific for this equation, maybe you will have a weird solution about the sound type (maybe the result tells you it a cow has a fight with a horse!!) This is definetely possible because both of these two matrixs are changeable, you can have various $\vec{X}$ by changing the value in $\vec{A}$.

### Find one of the possibility

So now we lack information to separate the sound (to find the matrix $\vec{A}$), and the only thing we can do is assume some of the situations (or add some limitations) to "create" more information for our equation. Anyway, the results might still be different from the original sound's signal. But we can say this result is one of the possibilities and we can use this because it seems similar to the original one (even if they are not the same, but they are similar!). 

#### What we assume?

Okay, the fist thing we need to assume is that **all of the sounds we have are independent**, they are not related to each other. The another assume is the sound is pre-whiten, which means the signal we have now will have this relationship.

> $$\vec{S} \vec{S}^T = \vec{I}$$

$\vec{I}$ is a unit matrix. So now we can do the singular value decomposition for $\vec{A}$, and we get:

> $$\vec{A} = \vec{U}\vec{\sum}\vec{V}^T $$

Do not be confusing about these three new guys, they are just some symbols :->, all we need to know is that **any matrix can be decomposed as a multiplication of three matrices**, and one of the matrices is a orthogonal matrix ( $\vec{U}$ ) , the other one is a rotation matrix ( $\vec{V}$ )

The rotation matrix has the following properties:

> $$\vec{V}^T = \vec{V}^{-1}$$

Now we can do some interesting mutiple for our matrix (no worries, it's very easy)

> $$\vec{X}\vec{X}^T = (\vec{A}\vec{S})(\vec{A}\vec{S})^T $$
> 
> $$ = \vec{A}\vec{A}^T $$ 
>
> $$ = (\vec{U}\vec{\sum}\vec{V}^T)(\vec{U}\vec{\sum}\vec{V}^T)^T $$
>
> $$ = \vec{U}\vec{\sum}\vec{V}^T\vec{V}\vec{\sum}^T\vec{U}^T $$
>
> $$ = \vec{U}\vec{\sum}^2\vec{U}^T $$

And luckily, we know that $\vec{X}\vec{X}^T$ is a symmetric matrix, according to the properties of symmetric matrices, we know that symmetric matrices can be orthogonally decomposed by their singular vectors, namely:
> $$ \vec{X}\vec{X}^T = \vec{E}\vec{D}\vec{E}^T $$

Do you see the same thing between these two expressions? Yep! actually
> $$ \vec{U} = \vec{E} $$
> 
> $$ \vec{\sum} = \sqrt{\vec{D}} $$

Now, we have $\vec{U}$ and $\vec{\sum}$, the only thing we need is $\vec{V}$

And now the $\vec{S}$ can be written as:

> $$ \vec{S} = \vec{V}\sqrt{\vec{D}}\vec{E}^T\vec{X} $$

From what we know, let's make the long equation become simple and elegant.

> $$ \vec{X_w} = \sqrt{\vec{D}}\vec{E}^T\vec{X} $$

So now, we can rewritten our expression as :
> $$ \vec{S} = \vec{V}\vec{X_w}$$

#### Some information knowledge...
We're going to use information theory here. There are many theories in information theory that measure the statistical independence of a distribution. For example, the simplest one, mutual information, measures the independence of two variables. As generalized mutual information, multi-information can measure the statistical independence of multiple variables.
If all variables are statistically independent, then the $\vec{I}$(**y**) will be 0, otherwise, $\vec{I}$(**y**) > 0

The expression about the muti-information is like:
> $$ \vec{I}(**y**) = \int_{}^{}{**P**(y)log_2\frac{**P**(y)}{\quad\prod_i P(y_i)}}d**y** $$

Nevermind about this long ugly expression, the all thing we need know is that this expression is related with our beautiful expression in the past. Let $\vec{S}$ be the **P**(y), and the $\vec{I}$(**y**) will depend on the $X_w$, and the $\vec{V}$.

We already know the $\vec{X_w}$, so we need to find the $\vec{V}$ to make the result become 0 (or minimize it). There are two methods to do it:

By using the properties of the rotation matrix, start from 0 degrees to 180 degrees, and observe which Angle has the smallest multi-information.

Or sometimes we cannot see the graph, in this case, we can transform the multi-information expression in the expression into a function of entropy. Entropy is a well-known measure of uncertainty. And find the minimize of the new expression, we find the V martrix. After a long period of math calculation, we can get this expression:
> $$ \vec{V} = argmin\sum_{i}H[(\vec{V}X_w)_i]$$

Anyway, we can use a lot of method to optimize this "loss function" and get the V vector. So finally! We get the $\vec{V}$. Now we have all of the three martrix, we can get our $\vec{A}$ and then find the inverse of it. Then we mutiple it with the $\vec{X}$ and get the result about one of the possible result $\vec{S}$.

Now your computer is happy too, he can use this ICA method to seperate the sounds and choose the fight sound to hear. Here is an example of his work on brain wave.

![](https://cdn.jsdelivr.net/gh/NeyMarKevin/pic@main/202209081416292.gif)

It seems like there is still some problems about ICA, they are:

The Gaussian distribution signal cannot be recovered;

Each signal is assumed to be independent of each other;

Unable to distinguish the displaced source;

Simultaneous increases and decreases in signal strength cannot be identified.

So if you really want to use this method, it's necessary to consider about all of that. Just be careful.

# LLE (In progress)
