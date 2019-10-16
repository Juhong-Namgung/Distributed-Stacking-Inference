package dke.cs.knu.v3;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.storm.task.OutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.base.BaseRichBolt;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Tuple;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.util.Map;

public class Level1Bolt extends BaseRichBolt {
    private static Log LOG = LogFactory.getLog(Level1Bolt.class);
    OutputCollector collector;
    float[] test = new float[3];
    float[][] level0Result = new float[1][3];
    private String modelPath;       // Deep Learning Model Path
    private float[][] result_v = new float[1][1];

    public Level1Bolt(String path) {
        this.modelPath = path;
    }
    @Override
    public void prepare(Map stormConf, TopologyContext context, OutputCollector collector) {
        this.collector = collector;
    }

    @Override
    public void execute(Tuple input) {
//        float[][] level0Result = (float[][])input.getValueByField("level0");
        float[][] resultA = (float[][])input.getValueByField("level0_a");
        float[][] resultB = (float[][])input.getValueByField("level0_b");
        float[][] resultC = (float[][])input.getValueByField("level0_c");

        level0Result[0][0] = resultA[0][0];
        level0Result[0][1] = resultB[0][0];
        level0Result[0][2] = resultC[0][0];


        try (SavedModelBundle b = SavedModelBundle.load(modelPath, "serve")) {

            Session sess = b.session();

            Tensor finalTensor = Tensor.create(level0Result);
            Tensor finalResult = sess.runner()
                    .feed("final_input", finalTensor)
                    .fetch("final_output/BiasAdd")
                    .run()
                    .get(0);
            System.out.print("Stacking Final Result: ");
            printTensor(finalResult);
        }
    }

    @Override
    public void declareOutputFields(OutputFieldsDeclarer declarer) {
    }

    public void printTensor(Tensor tensor) {
        result_v = (float[][]) tensor.copyTo(new float[1][1]);
        for (int i = 0; i < result_v.length; i++) {
            System.out.println(result_v[i][0]);
        }
    }
}