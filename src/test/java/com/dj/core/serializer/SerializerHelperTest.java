package com.dj.core.serializer;


import com.dj.core.model.activation.Sigmoid;
import com.dj.core.model.graph.ConnectedNeuron;
import com.dj.core.model.graph.Context;
import com.dj.core.model.graph.InputNeuron;
import com.dj.core.model.graph.Neuron;
import org.junit.Test;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertFalse;

public class SerializerHelperTest {

    private static final String TMP_DIR_PATH = System.getProperty("java.io.tmpdir");

    @Test
    public void testSerializeToFileAndDeserialize() {
        Context context = new Context(.2, false, 2);
        ConnectedNeuron outputNeuron = new ConnectedNeuron.Builder()
                .bias(-1.)
                .activationFunction(new Sigmoid())
                .context(context)
                .build();
        InputNeuron inputFriend = new InputNeuron();
        InputNeuron inputVodka = new InputNeuron();
        InputNeuron inputSunny = new InputNeuron();

        inputFriend.connect(outputNeuron, 1.);
        inputVodka.connect(outputNeuron, 1.);
        inputSunny.connect(outputNeuron, 1.);


        inputFriend.forwardSignalReceived(null, 1.);
        inputVodka.forwardSignalReceived(null, 1.);
        inputSunny.forwardSignalReceived(null, 1.);
        assertTrue(outputNeuron.getForwardResult()[0] > .7);

        List<Neuron> inputLayer = List.of(inputFriend, inputVodka, inputSunny);
        List<Neuron> outputLayer = List.of(outputNeuron);

        ModelWrapper modelWrapper = new ModelWrapper.Builder().inputLayer(inputLayer).outputLayer(outputLayer).build();

        Path tmpModelFilePath = Paths.get(TMP_DIR_PATH, "test.dj");
        File tmpModelFile = tmpModelFilePath.toFile();
        if (tmpModelFile.exists()) {
            tmpModelFile.delete();
        }

        assertFalse(tmpModelFile.exists());

        SerializerHelper.serializeToFile(modelWrapper, tmpModelFilePath.toString());

        assertTrue(tmpModelFile.exists());

        ModelWrapper afterSerialization = SerializerHelper.deserializeFromFile(tmpModelFilePath.toString());
        List<Neuron> inputLayerAfterSeialization = afterSerialization.getInputLayer();
        List<Neuron> outputLayerAfterSeialization = afterSerialization.getOutputLayer();

        Neuron outputNeuronToTest = outputLayerAfterSeialization.get(0);
        Neuron inputFriendToTest = inputLayerAfterSeialization.get(0);
        Neuron inputVodkaToTest = inputLayerAfterSeialization.get(1);
        Neuron inputSunnyToTest = inputLayerAfterSeialization.get(2);

        inputFriendToTest.forwardSignalReceived(null, 1.);
        inputVodkaToTest.forwardSignalReceived(null, 1.);
        inputSunnyToTest.forwardSignalReceived(null, 1.);
        assertTrue(outputNeuronToTest.getForwardResult()[0] > .7);
    }
}
