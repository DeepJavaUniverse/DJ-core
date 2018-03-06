# DeepJava


Read this in other languages: [Русский](README.ru.md)

DeepJava (DJ) is a DeepLearning framework. One might ask: why do we need yet another DeepLearning framework? Good question. There is, at least, one thing that makes DJ different: it is, mainly, for the educational purpose. What does this mean exactly:

* Codebase should be understandable (not fast). Anyone who has read a book about DeepLearning should be able to map main concepts from the book to the code in this framework (if concept already implemented);
* You can experiment. Educational purpose means that the framework open for the experiments. Do you have an idea how to represent a computational graph in a non-canonical way? You can try it here!
* Simple to use. DJ prioritizing the simplicity over the speed, such priority would not be possible with other frameworks.

# Usage Example

## 1 Layer Perceptron With Sigmoid Activation Function 

```java
InputNeuron inputFriend = new InputNeuron("friend");
InputNeuron inputVodka = new InputNeuron("vodka");
InputNeuron inputSunny = new InputNeuron("sunny");

ConnectedNeuron outputNeuron
        = new ConnectedNeuron.Builder()
            .bias(0.1)
            .activationFunction(new Sigmoid())
            .learningRate(0.2)
            .build();

inputFriend.connect(outputNeuron, wFriend);
inputVodka.connect(outputNeuron, wVodka);
inputSunny.connect(outputNeuron, wSunny);

// Sending input signal to the graph:
inputFriend.forwardSignalReceived(null, 1.);
inputVodka.forwardSignalReceived(null, 1.);
inputSunny.forwardSignalReceived(null, 1.);

// Sending error back to the graph:
outputNeuron.backwardSignalReceived(dA);
```

# How To Contribute

There are several ways:
* open bug/issue if you have found something or want us to do something;
* submit a PR;
