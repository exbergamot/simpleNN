package dkgames.neuralnet;

public class TrainingData {
    private double[] input;
    private double[] target;

    public TrainingData(double[] input, double[] target) {
        this.input = input;
        this.target = target;
    }

    public double[] getInput() {
        return input;
    }

    public double[] getTarget() {
        return target;
    }
}
