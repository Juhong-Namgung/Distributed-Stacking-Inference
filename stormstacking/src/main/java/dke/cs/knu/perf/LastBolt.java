package dke.cs.knu.perf;

import org.apache.storm.task.OutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.base.BaseRichBolt;
import org.apache.storm.tuple.Tuple;

import java.util.Map;

public class LastBolt extends BaseRichBolt {
    
	private OutputCollector collector;
	float[][] result;

    public void prepare(Map stormConf, TopologyContext context, OutputCollector collector) {
        this.result = new float[4][];
    	this.collector = collector;
    }

    public void execute(Tuple tuple) {
    	
    	result = (float[][]) tuple.getValueByField("result");
    	 for(int i=0; i<result.length; i++) {
             System.out.println(result[i][0]);          
         }
        this.collector.ack(tuple);
    }

    public void declareOutputFields(OutputFieldsDeclarer declarer) {
    }
}