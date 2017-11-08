package dkgames.neuralnet;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Runner {
    public static void main(String[] args) {
        List<TrainingData> trainingData = new ArrayList<>();

        double[] input = new double[] {0, 1};
        double[] target = new double[] {1, 0};
        trainingData.add(new TrainingData(input, target));

        double[] input1 = new double[] {1, 1};
        double[] target1 = new double[] {0, 1};
        trainingData.add(new TrainingData(input1, target1));

        double[] input2 = new double[] {1, 0};
        double[] target2 = new double[] {1, 0};
        trainingData.add(new TrainingData(input2, target2));

        double[] input3 = new double[] {0, 0};
        double[] target3 = new double[] {0, 1};
        trainingData.add(new TrainingData(input3, target3));

        FeedForwardNet ffn = new FeedForwardNet();

        for (int i = 0; i < 100000; i++) {
            Collections.shuffle(trainingData);
            for (TrainingData each : trainingData) {
                ffn.setInputs(each.getInput());
                ffn.activate();

                double[] outputs = ffn.getOutputs();
                System.out.printf(" %.10f, %.10f %n", outputs[0], outputs[1]);

                ffn.backwardPass(each.getTarget());
            }
        }
        System.out.println(ffn);
    }
}
