package dke.cs.knu.perf;

import org.apache.storm.task.OutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.base.BaseRichBolt;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Tuple;
import org.apache.storm.tuple.Values;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class MiddleBolt extends BaseRichBolt {
    private OutputCollector collector;
    private static int ROW = 0;
    private static int FEATURE = 0;
    private String modelDir = "/home/team1/juhong/kepco/tensorflowforjava/resultmodel/PythonModel/result/regressionModel";
    private String raw;
    private SavedModelBundle b;
    private Session sess;
   

    public void prepare(Map stormConf, TopologyContext context, OutputCollector collector) {
        this.collector = collector;
        this.b = SavedModelBundle.load(modelDir, "serve");
        this.sess = b.session();
        
    }

    public void execute(Tuple tuple) {
    	
    	raw = (String) tuple.getValueByField("input");
    	getDataSize(raw);
    	
    	float[][] testInput = new float[ROW][FEATURE];
    	csvToMtrx(raw, testInput);    	
    	
    	Tensor x = Tensor.create(testInput);
     
    	float[][] y = sess.runner()
                .feed("x", x)
                .fetch("h")
                .run()
                .get(0)
                .copyTo(new float[ROW][1]);
    	
    	 for(int i=0; i<y.length; i++) {
             System.out.println(y[i][0]);          
         }
    	 System.out.println("====");
    	 
    	 this.collector.emit(tuple, new Values((Object)y));
    	 this.collector.ack(tuple);
    }

    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        declarer.declare(new Fields("result"));
    }           
    
    public static void getDataSize(String raw) {
		String[] field = null;
		field = raw.split(",");
		ROW = 1;
		FEATURE = field.length;
	}    

	public static void csvToMtrx(String raw, float[][] mtrx) {
		String[] field = null;
		field = raw.split(",");
		for(int j=0; j<field.length; j++)
			mtrx[0][j] = Float.parseFloat(field[j]);
	}
	
	 public static void printMatrix(float[][] mtrx) {
	        System.out.println("============ARRAY VALUES============");
	        for(int i=0; i<mtrx.length; i++) {
	            if(i==0)
	                System.out.print("[");
	            else
	                System.out.println();
	            for(int j =0; j<mtrx[i].length; j++) {
	                System.out.print("["+mtrx[i][j]+"]");
	            }
	        }
	        System.out.println("]");
	    }

}