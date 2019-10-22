package dke.cs.knu.v2;

import dke.cs.knu.RandomTupleSpout;
import dke.cs.knu.ReportBolt;
import dke.cs.knu.v1.StackingBolt;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.storm.Config;
import org.apache.storm.StormSubmitter;
import org.apache.storm.generated.StormTopology;
import org.apache.storm.kafka.*;
import org.apache.storm.perf.utils.Helper;
import org.apache.storm.spout.SchemeAsMultiScheme;
import org.apache.storm.topology.BoltDeclarer;
import org.apache.storm.topology.TopologyBuilder;
import org.apache.storm.utils.Utils;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.UUID;

public class PerfTopoV2 {

    private static final String TOPOLOGY_NAME = "PerformanceTopo";
    private static final String FIRST_SPOUT_ID = "random-spout";
    private static final String LEVEL0_BOLT_ID = "level0-bolt";
    private static final String LEVEL1_BOLT_ID = "level1-bolt";
    private static final String LAST_BOLT_ID = "last-bolt";

    private static Log LOG = LogFactory.getLog(PerfTopoV2.class);

    @Option(name = "--help", aliases = {"-h"}, usage = "print help message")
    private static boolean _help = false;

    @Option(name = "--topologyName", aliases = {"--name"}, metaVar = "TOPOLOGIE NAME", usage = "name of topology")
    private static String topologyName = "Topo";

    @Option(name = "--inputTopic", aliases = {"--input"}, metaVar = "INPUT TOPIC", usage = "name of input kafka topic")
    private static String inputTopic = "input";

    @Option(name = "--outputTopic", aliases = {"--output"}, metaVar = "OUTPUT TOPIC", usage = "name of output kafka topic")
    private static String outputTopic = "output";

    @Option(name = "--testTime", aliases = {"--t"}, metaVar = "TIME", usage = "how long should run topology")
    private static int testTime = 3;

    @Option(name = "--numWorkers", aliases = {"--workers"}, metaVar = "WORKERS", usage = "number of workers")
    private static int numWorkers = 8;

    @Option(name = "--zookeeperHosts", aliases = {"--zookeeper"}, metaVar = "ZOOKEEPER HOST", usage = "path of zookeeper host")
    private static String zkhosts = "MN:42181,SN01:42181,SN02:42181,SN03:42181,SN04:42181,SN05:42181,SN06:42181,SN07:42181,SN08:42181";

    @Option(name = "--brokerList", aliases = {"--broker"}, metaVar = "BROKER LIST", usage = "path of broker list, bootstrap servers")
    private static String bootstrap = "MN:9092,SN01:9092,SN02:9092,SN03:9092,SN04:9092,SN05:9092,SN06:9092,SN07:9092,SN08:9092";

    @Option(name = "--parallelismHint", aliases = {"--parm"}, metaVar = "PARALLELISM HINT", usage = "number of spout, bolts(KafkaSpout-StackingBolt")
    private static String paralleism = "1 4";

    @Option(name = "--modelPath", aliases = {"--model"} , metaVar = "TENSORFLOW MODEL PATH", usage ="path of deep learning model")
    private static String modelPath = "./models/";

    @Option(name = "--inputSource", aliases = {"--source"}, metaVar = "INPUT SOURCE TYPE", usage = "type of input source [spout | kafka]")
    private static String source = "spout";

    @Option(name = "--topoConfPath", aliases = {"--conf"}, metaVar = "TOPOLOGY CONFIG FILE PATH", usage = "path of topology config")
    private static String configPath = "./conf/perf.yaml";

    @Option(name = "--spoutInterval", aliases = {"--interval"}, metaVar = "SPOUT MESSAGE INTERVAL", usage = "spout message interval time(ms)")
    private static int interval = 100;

    private static StormTopology getTopology(Map conf) {

        TopologyBuilder builder = new TopologyBuilder();

        	/* Kafka Spout Configuration */
        BrokerHosts brokerHosts = new ZkHosts(zkhosts);

        SpoutConfig kafkaSpoutConfig = new SpoutConfig(brokerHosts, inputTopic, "/" + inputTopic,
                UUID.randomUUID().toString());
        kafkaSpoutConfig.scheme = new SchemeAsMultiScheme(new StringScheme());

			/* KafkaBolt Configuration */
        Properties props = new Properties();
        props.put("metadata.broker.list", bootstrap);
        props.put("bootstrap.servers", bootstrap);
        props.put("acks", "1");
        props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        props.put("value.serializer", "org.springframework.kafka.support.serializer.JsonSerializer");


        // 1 -  Setup Spout   --------
        KafkaSpout kafkaSpout = new KafkaSpout(kafkaSpoutConfig);
        RandomTupleSpout randomSpout = new RandomTupleSpout(Helper.getInt(conf, "message.interval", interval));

        builder.setSpout(FIRST_SPOUT_ID, randomSpout,  Helper.getInt(conf, "first.spout.executors.count", 1))
                .setNumTasks(Helper.getInt(conf, "first.spout.tasks.count", 1));
        String prevTaskID = FIRST_SPOUT_ID;

        // 2 -  Setup Middle Bolts   --------
        BoltDeclarer middleBolt = builder.setBolt(LEVEL0_BOLT_ID, new Level0Bolt(modelPath), Helper.getInt(conf, "middle.bolt.executors.count", 1))
                .setNumTasks(Helper.getInt(conf,"middle.bolt.tasks.count", 1));
        setGrouping(middleBolt, conf, prevTaskID);
        prevTaskID = LEVEL0_BOLT_ID;

        BoltDeclarer middle2Bolt = builder.setBolt(LEVEL1_BOLT_ID, new Level1Bolt(modelPath), Helper.getInt(conf, "middle.bolt.executors.count", 1))
                .setNumTasks(Helper.getInt(conf,"middle.bolt.tasks.count", 1));
        setGrouping(middle2Bolt, conf, prevTaskID);
        prevTaskID = LEVEL1_BOLT_ID;

        // 3 -  Setup Last Bolt   --------
        BoltDeclarer lastBolt = builder.setBolt(LAST_BOLT_ID, new ReportBolt(), Helper.getInt(conf, "last.bolt.executors.count", 1))
                .setNumTasks(Helper.getInt(conf, "last.bolt.tasks.count", 1));
        setGrouping(lastBolt, conf, prevTaskID);

        return builder.createTopology();
    }

    private static void setGrouping(BoltDeclarer bolt, Map conf, String prevTaskID) {
        List<String> zookeeperServers = Utils.getStrings(Utils.get(conf, "storm.zookeeper.servers", "localhost"));
        int zookeeperPort = Helper.getInt(conf, "storm.zookeeper.port", 42181);
        StringBuilder sb = new StringBuilder();
        for (String server : zookeeperServers) {
            sb.append(server + ":" + zookeeperPort + ",");
        }
        String zkConnection = sb.substring(0, sb.length() - 1);
        switch (Helper.getStr(conf, "grouping")) {
            case "shuffle":
                bolt.shuffleGrouping(prevTaskID);
                break;
            case "local":
                bolt.localOrShuffleGrouping(prevTaskID);
                break;
            case "all":
                bolt.allGrouping(prevTaskID);
                break;
            default:
                bolt.shuffleGrouping(prevTaskID);
                break;
        }
    }

    public static void main(String[] args) throws Exception {
        new PerfTopoV2().topoMain(args);
    }

    public void topoMain(String[] args) throws Exception {

        CmdLineParser parser = new CmdLineParser(this);
        parser.setUsageWidth(150);

        try {
            // parse the arguments.
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            // if there's a problem in the command line,
            // you'll get this exception. this will report
            // an error message.
            System.err.println(e.getMessage());
            _help = true;
        }
        if (_help) {
            parser.printUsage(System.err);
            System.err.println();
            return;
        }
        if (numWorkers <= 0) {
            throw new IllegalArgumentException("Need at least one worker");
        }
        if (topologyName == null || topologyName.isEmpty()) {
            throw new IllegalArgumentException("Topology Name must be something");
        }

        // submit to real cluster

        Map<String, Object> topoConf = Utils.findAndReadConfigFile(configPath);
        topoConf.put(Config.TOPOLOGY_DISABLE_LOADAWARE_MESSAGING, true);
        topoConf.put("benchmark.label", topologyName);

        //  Submit topology to storm cluster
        int countJob = Helper.getInt(topoConf, "topology.count", 1);
        while (countJob-- > 1)
            StormSubmitter.submitTopology(TOPOLOGY_NAME + "-" + countJob, topoConf, getTopology(topoConf));

        Helper.runOnClusterAndPrintMetrics((testTime * 60), TOPOLOGY_NAME + "-" + countJob, topoConf, getTopology(topoConf));


    }
}
