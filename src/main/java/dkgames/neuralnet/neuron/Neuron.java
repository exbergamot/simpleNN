package dkgames.neuralnet.neuron;

import java.io.Serializable;

public interface Neuron extends Serializable {
    public double getOut();
    public void activate();
    public void initWeights();
    public void addDErrorDOut(double dErrorDOut);
    public void backwardPass();
    public void applyDeltas();
}
