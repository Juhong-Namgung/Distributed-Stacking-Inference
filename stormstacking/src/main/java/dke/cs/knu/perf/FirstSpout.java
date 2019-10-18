package dke.cs.knu.perf;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import javax.imageio.ImageIO;

import org.apache.storm.spout.SpoutOutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.base.BaseRichSpout;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Values;

public class FirstSpout extends BaseRichSpout {

	private SpoutOutputCollector collector;
	private String input;	
	private int count = 0;
	private int interval;

	public FirstSpout(int interval) {
		this.interval = interval;
	}

	@Override
	public void open(Map conf, TopologyContext context, SpoutOutputCollector collector) {
		this.collector = collector;
		input = "";
	}
	public void nextTuple() {

		input = makeRandomCSV();
		collector.emit(new Values(input), this.count++);
		try {
			Thread.sleep(interval);
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		

	}

	public void declareOutputFields(OutputFieldsDeclarer declarer) {
		declarer.declare(new Fields("input"));
	}

	public void ack(Object msgId) {
		super.ack(msgId);
	}
	
	public static String makeRandomCSV() {
		double randomValue; 
		int intValue = 0;
		String output = "";
		
		randomValue = Math.random();
		intValue= (int)(randomValue * 2);
		
		output = output + intValue ;
		for(int i=1; i<4; i++) {
			randomValue = Math.random();
			intValue= (int)(randomValue * 2);
			output = output + "," + intValue;
		}		
		return output;
	}

}
