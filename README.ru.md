# DeepJava

DeepJava (DJ) - фреймворк глубокого обучения на Java. Зачем нужен еще один такой фрейворк? Есть как минимум одна причина: он предназначен преимущественно для образовательных целей. А это значит:
* Код должен быть понятным (даже иногда в ущерб производительности). Читатель книги по глубокому обучению должен иметь возможность связать основные концепты книги с кодом этого фреймворка (если он уже был реализован);
* Фреймворк позволяет экспериментировать. У вас есть идея, как представить вычислительный граф необычным способом? Вы можете попробовать сделать это здесь;
* Легко использовать. DJ прост, так как он не ставит своей основной целью скорость работы в отличие от других фреймворков.

# Пример Использования

## Однослойный персептрон с сигмоидом в качестве функции активации

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

// Посылаем входные сигналы:
inputFriend.forwardSignalReceived(null, 1.);
inputVodka.forwardSignalReceived(null, 1.);
inputSunny.forwardSignalReceived(null, 1.);

// Посылаем обратно ошибку:
outputNeuron.backwardSignalReceived(dA);
```

# Помощь проекту

Вы можете:
* открыть баг/таск, если вы нашли ошибку или хотите, чтобы мы что-то исправили/добавили;
* сделать pull request.
