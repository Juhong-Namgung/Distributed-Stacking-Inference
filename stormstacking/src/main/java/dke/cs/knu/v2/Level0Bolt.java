package dke.cs.knu.v2;

import dke.cs.knu.PreProcessor;
import dke.cs.knu.v1.StackingBolt;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.storm.task.OutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.base.BaseRichBolt;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Tuple;
import org.apache.storm.tuple.Values;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.util.Map;

public class Level0Bolt extends BaseRichBolt {
    private static Log LOG = LogFactory.getLog(StackingBolt.class);
    OutputCollector collector;

    private float[][] urlTensor = new float[1][21];
    private String modelPath;       // Deep Learning Model Path
    private PreProcessor printable;
    private float[][] result_v = new float[1][1];

    public Level0Bolt(String path) {
        this.modelPath = path;
    }
    @Override
    public void prepare(Map stormConf, TopologyContext context, OutputCollector collector) {
        this.collector = collector;
        printable = new PreProcessor();
    }


    @Override
    public void execute(Tuple input) {
        String validURL = (String) input.getValueByField("str");
        String detectResult;

        try (SavedModelBundle b = SavedModelBundle.load(modelPath, "serve")) {
            urlTensor = printable.convert(validURL);

            //create an input Tensor
            Tensor x = Tensor.create(urlTensor);

            Session sess = b.session();

            float[][] resultLevel0 = new float[1][3];

            Tensor result = sess.runner()
                    .feed("nn1_input", x)
                    .fetch("nn1_output/BiasAdd")
                    .run()
                    .get(0)
                    ;

            float[][] value = (float[][]) result.copyTo(new float[1][1]);
            resultLevel0[0][0] = value[0][0];

            System.out.print("NN1 result: ");
            printTensor(result);

            Tensor result2 = sess.runner()
                    .feed("nn2_input", x)
                    .fetch("nn2_output/BiasAdd")
                    .run()
                    .get(0);

            value = (float[][]) result2.copyTo(new float[1][1]);
            resultLevel0[0][1] = value[0][0];

            System.out.print("NN2 result: ");
            printTensor(result2);

            Tensor result3 = sess.runner()
                    .feed("nn3_input", x)
                    .fetch("nn3_output/BiasAdd")
                    .run()
                    .get(0);

            value = (float[][]) result3.copyTo(new float[1][1]);
            resultLevel0[0][2] = value[0][0];

            System.out.print("NN3 result: ");
            printTensor(result3);

            System.out.println("Level 0 models result: ");
            for (int i = 0; i < 3; i++) {
                System.out.print("[" +resultLevel0[0][i] + "] ");
            }

            collector.emit(new Values(resultLevel0));
//            Tensor finalTensor = Tensor.create(resultLevel0);
//            Tensor finalResult = sess.runner()
//                    .feed("final_input", finalTensor)
//                    .fetch("final_output/BiasAdd")
//                    .run()
//                    .get(0);
//            System.out.print("Stacking Final Result: ");
//            printTensor(finalResult);

        }
    }

    @Override
    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        declarer.declare(new Fields("level0"));
    }

    public void printTensor(Tensor tensor) {
        result_v = (float[][]) tensor.copyTo(new float[1][1]);
        for (int i = 0; i < result_v.length; i++) {
            System.out.println(result_v[i][0]);
        }
    }
}