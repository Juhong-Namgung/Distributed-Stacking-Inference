package dke.cs.knu;

import org.apache.storm.spout.SpoutOutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.base.BaseRichSpout;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Values;

import java.util.Map;
import java.util.Random;

public class RandomTupleSpout extends BaseRichSpout {

    private SpoutOutputCollector collector;
    private int count = 0;
    private int interval = 0;
    private String[] inputs = {
            "2014,12,09,3,2.25,2570,7242,2.0,0,0,3,7,2170,400,1951,1991,98125,47.721000000000004,-122.319,1690,7639",
            "2014,12,09,4,3.0,1960,5000,1.0,0,0,5,7,1050,910,1965,0,98136,47.5208,-122.39299999999999,1360,5000",
            "2014,05,12,4,4.5,5420,101930,1.0,0,0,3,11,3890,1530,2001,0,98053,47.6561,-122.005,4760,101930",
            "2015,04,15,3,1.0,1780,7470,1.0,0,0,3,7,1050,730,1960,0,98146,47.5123,-122.337,1780,8113",
            "2015,03,12,3,2.5,1890,6560,2.0,0,0,3,7,1890,0,2003,0,98038,47.3684,-122.031,2390,7570",
            "2014,07,03,5,2.5,2270,6300,2.0,0,0,3,8,2270,0,1995,0,98092,47.3266,-122.169,2240,7005",
            "2014,06,24,3,1.75,1520,6380,1.0,0,0,3,7,790,730,1948,0,98115,47.695,-122.304,1520,6235",
            "2015,03,02,4,2.5,2570,7173,2.0,0,0,3,8,2570,0,2005,0,98052,47.7073,-122.11,2630,6026",
            "2014,12,01,2,1.5,1190,1265,3.0,0,0,3,7,1190,0,2005,0,98133,47.7274,-122.35700000000001,1390,1756",
            "2014,05,28,4,1.0,1660,34848,1.0,0,0,1,5,930,730,1933,0,98052,47.6621,-122.132,2160,11467"
    };
    private Random random;

    public RandomTupleSpout(int interval) {
        this.interval = interval;
    }

    public void ack(Object msgId) {
        super.ack(msgId);
    }

    @Override
    public void open(Map map, TopologyContext topologyContext, SpoutOutputCollector spoutOutputCollector) {
        this.collector = spoutOutputCollector;
        random = new Random();
    }

    @Override
    public void nextTuple() {
        this.collector.emit(new Values(inputs[random.nextInt(10)], this.count++));

        try {
            Thread.sleep(interval);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

    }

    @Override
    public void declareOutputFields(OutputFieldsDeclarer outputFieldsDeclarer) {
        outputFieldsDeclarer.declare(new Fields("str"));
    }
}
