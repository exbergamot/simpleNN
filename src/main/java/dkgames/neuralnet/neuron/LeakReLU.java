package dkgames.neuralnet.neuron;

import java.util.Random;

public class LeakReLU implements Neuron {
    private static final Random random = new Random();
    private final static double LEAKY_COEF = 0.01;
    private final static double LEARNING_RATE = 0.1;

    private Neuron[] inputs;
    private Neuron[] outputs;
    private double[] weights;
    private double[] weightDeltas;
    private double out;
    private double eout;

    @Override
    public double getOut() {
        return out;
    }

    @Override
    public void activate() {
        out = out();
    }

    @Override
    public void initWeights() {
        weights = new double[inputs.length];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = random.nextGaussian();
        }
        weightDeltas = new double[inputs.length];
    }

    private double summator() {
        double z = 0;
        for (int i = 0; i < inputs.length; i++) {
            z += inputs[i].getOut() * weights[i];
        }
        return z;
    }

    double out() {
        double z = summator();
        if (z > 0) {
            return  z;
        } else {
            return z * LEAKY_COEF;
        }
    }

    private double zDerivative() {
        if (out > 0) {
            return 1;
        } else {
            return LEAKY_COEF;
        }
    }


    private void calculateWeightDelta(double dErrorDz) {
        for (int i = 0; i < inputs.length; i++) {
            weightDeltas[i] = LEARNING_RATE * dErrorDz * inputs[i].getOut();
        }
    }

    public void addDErrorDOut(double dErrorDOut) {
        this.eout += dErrorDOut;
    }

    public void backwardPass() {
        double dErrorDz = eout * zDerivative();
        calculateWeightDelta(dErrorDz);
        updatePreviousLayerErrors(dErrorDz);
    }

    private void updatePreviousLayerErrors(double dErrorDz) {
        for (int i = 0; i < inputs.length; i++) {
            inputs[i].addDErrorDOut(dErrorDz * weights[i]);
        }
    }

    public void applyDeltas() {
        for (int i = 0; i < weights.length; i++) {
            weights[i] -= weightDeltas[i];
            weightDeltas[i] = 0;
            eout = 0;
        }
    }

    public void setInputs(Neuron[] inputs) {
        this.inputs = inputs;
    }

    public void setOutputs(Neuron[] outputs) {
        this.outputs = outputs;
    }

    @Override
    public String toString() {
        double sum = 0;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i];
        }
        return String.format(" %05.5f ", sum);
    }
}
