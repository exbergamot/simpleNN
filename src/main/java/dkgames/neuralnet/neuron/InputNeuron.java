package dkgames.neuralnet.neuron;

public class InputNeuron implements Neuron {
    private double out;

    @Override
    public double getOut() {
        return out;
    }

    @Override
    public void activate() {

    }

    @Override
    public void initWeights() {

    }

    @Override
    public void addDErrorDOut(double dErrorDOut) {

    }

    @Override
    public void backwardPass() {

    }

    @Override
    public void applyDeltas() {

    }

    public void setOut(double out) {
        this.out = out;
    }

    @Override
    public String toString() {
        return "";
    }
}
