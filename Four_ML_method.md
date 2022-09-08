# Four methods/stuctures for machine learning

In this blog, four methods will be introduced in machine learning area, Independent Component Analysis (ICA), Locally Linear Embedding (LLE), Autoencoders and Long-short term memory (LSTM)

## Independent Conponent Analysis (ICA)

Let's assume that you have a dog and a cat, today is a rainy weekend, and you have nothing to do so you decided to watch a movie at home.

The movie is splendid, you are really enjoying it, however, it seems like your dog and cat is boring too, so they decide to have a fight with each other to have some fun. So at this time, they are shouting at each other, and that really impacts you to hear the sound of the movie. You know what? Your computer can also hear the sound, and he is annoyed too,(but he cannot speak to you, and he is annoyed because of the movie sound! He is curious to the world and wants to hear more about the fight).

You are annoyed, your computer is annoying, but there's nothing you and he can do to stop the noise because the cat and dog will fight again and the computer cannot close the movie by itself, so you and your computer are trying to ignore them, and you pay all your attention to the movie, luckily, that really works!! You can hear the characters talking in the movie! In this case, you think: Let these two small devils fight then, I can ignore their noise and enjoy the movie. But your computer failed, he is still suffering from the movie noise because he cannot ignore it, he really wants some help.

In this case, the noise is not disappeared or become small, but in your feelings, the noise disappeared and doesn't influence you to enjoy the movie. Thanks to the brain's help, the brain helps you eliminate most of the noise though it still exists. The noise cannot affect you any more! However, we cannot give the computer a brain (the computer will be unstoppable to control the world with a brain!), so the only thing we can do is develop an algorithm for him.

This type of problem called Blind Source Separation problem (BSS), which refers to the fact that the data we measure (or hear) is a mixture of data from multiple different sources. It should be noted that the data we measure is not a mixture of target data and noise, but a mixture of multiple different target data. (cat and dog's fight is not noise! because your computer wants to hear their barking!!)

So let do your computer a favour and we can help them from the root.

### Sounds add up linearly
We know that sounds add up linearly. What the microphone of your computer or our eardrum picks up is a linear superposition of multiple sound signals. Like the figure shows. So in this case, the sounds we heard can be seen as the combination of different types of sound (dog's woof, cat's mew and human's talking). The only different between the sound you heard and the sound computer heard is the size of sound. 

### Some basic mathematics
We are almost there! To understand what is ICA and how it works, we need a little mathematics (Just a little...)
***
Like we discussed before, the sounds we heard is a combination of different types of sound, we can use a linear vector to represent different type of sound.
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

So what now? actually we want two sepearte sounds in $\vec{S}$, it looks like we need to find $\vec{A}$

$\vec{A}$ is a 3 by 3 matrix to show that the different amplitudes of different sounds, unfortunately, the linear algebra also tells us it's impossible to find a specific solution for a one-degree equation which has two unknown varables.

So it seems like we have reached to a dead end, there is no solution for this equation, maybe you will have a weird solution (maybe a cow has a fight with a horse!!) This is definetely possible because both of these two matrixs are changeable, you can have the same $\vec{X}$ by changing the value in $\vec{A}$ 

### Find one of the possibility

So now we lack of information to seperate the sound (to find the matrix $\vec{A}$), the only thing we can do is assume some of the situations (or limitations) to "create" more information for our equation. Anyway, the results might be different from the original sound's signal. But we can say this result is one of the possibility and we can use this result because it similar to the original one (even if they are not same, but they are similar!). 

#### What we assume?

Okay, the fist thing we need to assume is that all of the sounds we have are independent, they are not related to each other
