package dkgames.neuralnet;

import dkgames.neuralnet.neuron.BiasNeuron;
import dkgames.neuralnet.neuron.InputNeuron;
import dkgames.neuralnet.neuron.LeakReLU;
import dkgames.neuralnet.neuron.Neuron;

import java.io.Serializable;
import java.util.Iterator;

public class FeedForwardNet implements Iterable<Neuron>, Serializable{
    private static final int HIDDEN_LAYERS = 1;
    private static final int HIDDEN_LAYER_SIZE = 2;
    private static final int INPUTS = 2;
    private static final int OUTPUTS = 2;

    private boolean forwardIteration = true;

    private Neuron[][] network;

    public FeedForwardNet() {
        network = new Neuron[HIDDEN_LAYERS + 2][];
        Neuron[] outputs = new Neuron[OUTPUTS];
        Neuron[] inputs = new InputNeuron[INPUTS + 1];

        network[0] = inputs;
        for (int i = 1; i < HIDDEN_LAYERS + 1; i++) {
            network[i] = new Neuron[HIDDEN_LAYER_SIZE + 1];
        }
        network[network.length - 1] = outputs;

        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = new InputNeuron();
        }

        for (int i = 1; i < network.length - 1; i++) {
            for (int j = 0; j < network[i].length; j++) {
                if (j == network[i].length - 1) {
                    network[i][j] = new BiasNeuron();
                } else {
                    LeakReLU neuron = new LeakReLU();
                    neuron.setInputs(network[i - 1]);
                    neuron.setOutputs(network[i + 1]);
                    network[i][j] = neuron;
                }
            }
        }

        for (int i = 0; i < outputs.length; i++) {
            LeakReLU neuron = new LeakReLU();
            neuron.setInputs(network[network.length - 2]);
            outputs[i] = neuron;
        }

        for (Neuron each : this) {
            each.initWeights();
        }
    }

    @Override
    public Iterator<Neuron> iterator() {
        return new NetIterator();
    }

    public double[] getOutputs() {
        double[] output = new double[OUTPUTS];
        Neuron[] outputLayer = network[network.length - 1];

        for (int i = 0; i < outputLayer.length ; i++) {
            output[i] = outputLayer[i].getOut();
        }

        return output;
    }

    public void setInputs(double[] inputs) {
        if (inputs.length != network[0].length - 1) {
            throw new IllegalArgumentException("Wrong size of input vector");
        }

        InputNeuron[] inputLayer = (InputNeuron[]) network[0];
        for (int i = 0; i < inputs.length; i++) {
            inputLayer[i].setOut(inputs[i]);
        }
        inputLayer[inputLayer.length - 1].setOut(1);
    }

    public void activate() {
        forwardIteration = true;
        for (Neuron each : this) {
            each.activate();
        }
    }

    public void backwardPass(double[] target) {
        forwardIteration = false;
        Neuron[] outputLayer = network[network.length - 1];

        for (int i = 0; i < outputLayer.length; i++) {
            Neuron output = outputLayer[i];
            double out = output.getOut();
            double error = Math.pow(target[i] - out, 2);
            System.out.printf("Error - %.10f    ", error);
            double dErrorDout = out - target[i];
            output.addDErrorDOut(dErrorDout);
        }

        for (Neuron each : this) {
            each.backwardPass();
        }
        for (Neuron each : this) {
            each.applyDeltas();
        }
    }

    private class NetIterator implements Iterator<Neuron> {
        private int direction;
        private int layer;
        private int index = -1;

        private NetIterator() {
            if (forwardIteration) {
                layer = 0;
                direction = 1;
            } else {
                layer = network.length - 1;
                direction = -1;
            }
        }

        @Override
        public boolean hasNext() {
            if (forwardIteration) {
                if (layer != network.length - 1 || index != network[layer].length - 1) {
                    return true;
                }
            } else {
                if (layer != 0 || index != network[layer].length - 1) {
                    return true;
                }
            }
            return false;
        }

        @Override
        public Neuron next() {
            index++;
            if (index == network[layer].length) {
                layer += direction;
                index = 0;
            }
            return network[layer][index];
        }
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < network.length; i++) {
            for (int j = 0; j < network[i].length; j++) {
                sb.append(network[i][j].toString());
            }
            sb.append("\r\n");
        }

        return sb.toString();
    }
}
