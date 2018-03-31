package com.dj.core.serializer;

import java.io.*;

public class SerializerHelper {

    private SerializerHelper() { } // no-op

    public static void serializeToFile(
            final ModelWrapper model,
            final String path) {
        try(var fileOut = new FileOutputStream(path);
            var outStream = new ObjectOutputStream(fileOut)) {
            outStream.writeObject(model);
        } catch (IOException  e) {
            e.printStackTrace();
            throw new RuntimeException("Serialization failed", e);
        }
    }

    public static ModelWrapper deserializeFromFile(
            final String path) {
        try(var fileIn = new FileInputStream(path);
            var outStream = new ObjectInputStream(fileIn)) {
            return (ModelWrapper) outStream.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            throw new RuntimeException("Serialization failed", e);
        }
    }
}
