package dke.cs.knu.v3;

import dke.cs.knu.PreProcessor;
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

public class Level0_C_ModelBolt extends BaseRichBolt {
    private static Log LOG = LogFactory.getLog(Level0_C_ModelBolt.class);
    OutputCollector collector;

    private float[][] inputTensor = new float[1][21];
    private String modelPath;       // Deep Learning Model Path
    private PreProcessor printable;
    private float[][] result_v = new float[1][1];

    public Level0_C_ModelBolt(String path) {
        this.modelPath = path;
    }
    @Override
    public void prepare(Map stormConf, TopologyContext context, OutputCollector collector) {
        this.collector = collector;
        printable = new PreProcessor();
    }


    @Override
    public void execute(Tuple input) {
        String inputValue = (String) input.getValueByField("input");

        try (SavedModelBundle b = SavedModelBundle.load(modelPath, "serve")) {
            inputTensor = printable.convert(inputValue);

            //create an input Tensor
            Tensor x = Tensor.create(inputTensor);

            Session sess = b.session();

            float[][] resultLevel0 = new float[1][3];

            Tensor result3 = sess.runner()
                    .feed("nn3_input", x)
                    .fetch("nn3_output/BiasAdd")
                    .run()
                    .get(0);

            float[][] value = (float[][]) result3.copyTo(new float[1][1]);
            resultLevel0[0][2] = value[0][0];

            System.out.print("NN3 result: ");
            printTensor(result3);

            collector.emit(new Values(input.getValueByField("level0_a"), input.getValueByField("level0_b"), resultLevel0));
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
        declarer.declare(new Fields("level0_a", "level0_b", "level0_c"));
    }

    public void printTensor(Tensor tensor) {
        result_v = (float[][]) tensor.copyTo(new float[1][1]);
        for (int i = 0; i < result_v.length; i++) {
            System.out.println(result_v[i][0]);
        }
    }
}