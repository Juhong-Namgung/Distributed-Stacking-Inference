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

public class Level0_A_ModelBolt extends BaseRichBolt {
    private static Log LOG = LogFactory.getLog(Level0_A_ModelBolt.class);
    OutputCollector collector;

    private float[][] inputTensor = new float[1][21];
    private String modelPath;       // Deep Learning Model Path
    private PreProcessor printable;
    private float[][] result_v = new float[1][1];
    private SavedModelBundle b;

    public Level0_A_ModelBolt(String path) {
        this.modelPath = path;
    }

    @Override
    public void prepare(Map stormConf, TopologyContext context, OutputCollector collector) {
        this.collector = collector;
        printable = new PreProcessor();
        b = SavedModelBundle.load(modelPath, "serve");
    }


    @Override
    public void execute(Tuple input) {
        String inputValue = (String) input.getValueByField("str");

        inputTensor = printable.convert(inputValue);

        //create an input Tensor
        Tensor x = Tensor.create(inputTensor);

        Session sess = b.session();

        float[][] resultLevel0 = new float[1][3];

        Tensor result = sess.runner()
                .feed("nn1_input", x)
                .fetch("nn1_output/BiasAdd")
                .run()
                .get(0);

        float[][] value = (float[][]) result.copyTo(new float[1][1]);
        resultLevel0[0][0] = value[0][0];


        collector.emit(new Values(inputValue, resultLevel0));
        this.collector.ack(input);


    }

    @Override
    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        declarer.declare(new Fields("input", "level0_a"));
    }

    public void printTensor(Tensor tensor) {
        result_v = (float[][]) tensor.copyTo(new float[1][1]);
        for (int i = 0; i < result_v.length; i++) {
            System.out.println(result_v[i][0]);
        }
    }
}