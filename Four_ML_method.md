# Four methods/stuctures for machine learning

In this blog, four methods will be introduced in machine learning area, Independent Component Analysis (ICA), Locally Linear Embedding (LLE), Autoencoders nad Long-short term memory (LSTM)

## Independent Conponent Analysis (ICA)

Let's assume that you have a dog and a cat, today is a rainy weekend, and you have nothing to do so you decided to watch a movie at home.

The movie is splendid, you are really enjoying it, however, it seems like your dog and cat is boring too, so they decide to have a fight with each other to have some fun. So in this time, they are shouting to each other and that really impacts you to hear the sound of the movie.

You are annoying, but there's nothing you can do to stop them because they will fight again, so you are trying to ignore them, and pay all your attention to the movie, luckily, that really works!! You can hear the characters talking in the movie! In this case, you think: Let these two small devils fight then, I can ignore their noise and enjoy the movie.

In this case, the noise is not disappeared or become small, but in your feelings, the noise disappeared and doesn't influence you to enjoy the movie. Thanks to the brain's help, the brain helps you eliminate most of the noise though it still exists. The noise cannot affect you any more! 

This type of problem called Blind Source Separation problem (BSS), which refers to the fact that the data we measure (or hear) is a mixture of data from multiple different sources. It should be noted that the data we measure is not a mixture of target data and noise, but a mixture of multiple different target data. (not like the cat and dog's noise, this time you want to hear their barking too!!)

### Sounds add up linearly
We know that sounds add up linearly. What the microphone or our eardrum picks up is a linear superposition of multiple sound signals. Like the figure shows. So in this case, the sounds we heard can be seen as the combination of different types of sound (dog's woof, cat's mew and human's talking). The only different between the sound you heard and the sound computer heard is the size of sound. 

### Some basic mathematics
We are almost there! To understand what is ICA and how it works, we need a little mathematics (Just a little...)
***
Like what I said before, the sounds we heard is a combination of different types of sound, we can use a linear vector to represent different type of sound.
> $\vec{S}$= [ $x_{dog}$, $x_{cat}$, $x_{movie}$]

Also, to show the combination of these types of sounds, we need another martix to do the combination.
> $\vec{A}$ = 
> $$\begin{bmatrix} 
> $w_1$ & $w_2$ & $w_3$\\
> $w_4$ & $w_5$ & $w_6$ \\
> $w_7$ & $w_8$ $w_9$ 
