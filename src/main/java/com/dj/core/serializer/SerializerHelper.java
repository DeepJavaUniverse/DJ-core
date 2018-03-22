package com.dj.core.serializer;

import java.io.*;

public class SerializerHelper {

    private SerializerHelper() { } // no-op

    public static void serializeToFile(
            final ModelWrapper model,
            final String path) {
        try(FileOutputStream fileOut = new FileOutputStream(path);
            ObjectOutputStream outStream = new ObjectOutputStream(fileOut)) {
            outStream.writeObject(model);
        } catch (IOException  e) {
            e.printStackTrace();
            throw new RuntimeException("Serialization failed", e);
        }
    }

    public static ModelWrapper deserializeFromFile(
            final String path) {
        try(FileInputStream fileIn = new FileInputStream(path);
            ObjectInputStream outStream = new ObjectInputStream(fileIn)) {
            return (ModelWrapper) outStream.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            throw new RuntimeException("Serialization failed", e);
        }
    }
}
