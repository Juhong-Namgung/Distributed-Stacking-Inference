package dke.cs.knu;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.storm.task.OutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.base.BaseRichBolt;
import org.apache.storm.tuple.Tuple;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.util.Map;

public class StackingBolt extends BaseRichBolt {
    private static Log LOG = LogFactory.getLog(StackingBolt.class);
    OutputCollector collector;

    private float[][] urlTensor = new float[1][21];
    private String modelPath;       // Deep Learning Model Path
    private PreProcessor printable;    //

    public StackingBolt(String path) {
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

            Tensor result = sess.runner()
                    .feed("main_input", x)
                    .fetch("main_output/BiasAdd")
                    .run()
                    .get(0);

            float[][] result_v = (float[][]) result.copyTo(new float[1][1]);
            //print the result
            for(int i=0; i<result_v.length;i++)
                System.out.println(result_v[i][0]);
        }
    }

    @Override
    public void declareOutputFields(OutputFieldsDeclarer declarer) {
    }
}