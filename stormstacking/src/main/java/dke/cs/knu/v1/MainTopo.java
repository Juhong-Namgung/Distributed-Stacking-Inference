package dke.cs.knu.v1;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.storm.Config;
import org.apache.storm.StormSubmitter;
import org.apache.storm.generated.*;
import org.apache.storm.kafka.*;
import org.apache.storm.spout.SchemeAsMultiScheme;
import org.apache.storm.thrift.TException;
import org.apache.storm.topology.TopologyBuilder;
import org.apache.storm.utils.NimbusClient;
import org.apache.storm.utils.Utils;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import java.util.ArrayList;
import java.util.Map;
import java.util.Properties;
import java.util.UUID;

public class MainTopo {
    private static Log LOG = LogFactory.getLog(MainTopo.class);

    @Option(name = "--help", aliases = {"-h"}, usage = "print help message")
    private boolean _help = false;

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

    public static void main(String[] args) throws NotAliveException, InterruptedException, TException {
        new MainTopo().topoMain(args);
    }

    public void topoMain(String[] args) throws InterruptedException, NotAliveException, TException {
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

        KafkaSpout kafkaSpout = new KafkaSpout(kafkaSpoutConfig);

        StackingBolt stackingBolt = new StackingBolt(modelPath);


			/* Topology Build */
        TopologyBuilder builder = new TopologyBuilder();

        ArrayList<Integer> parameters = new ArrayList<Integer>();
        String[] params = paralleism.split(" ");

        for(String p : params) {
            parameters.add(Integer.parseInt(p));
        }

        builder.setSpout("kafka-spout", kafkaSpout, parameters.get(0));
        builder.setBolt("detect-bolt", stackingBolt, parameters.get(1)).shuffleGrouping("kafka-spout");


        Config config = new Config();
        config.setNumWorkers(numWorkers);

        StormSubmitter.submitTopology(topologyName, config, builder.createTopology());

        try {
            Thread.sleep(testTime * 60 * 1000);

            Map<String, Object> conf = Utils.readStormConfig();
            Nimbus.Client client = NimbusClient.getConfiguredClient(conf).getClient();
            KillOptions killOpts = new KillOptions();
            killOpts.set_wait_secs(0);
            client.killTopologyWithOpts(topologyName, killOpts);
        } catch (AlreadyAliveException ae) {
            LOG.info(ae.get_msg());
        } catch (InvalidTopologyException ie) {
            LOG.info(ie.get_msg());
        }
    }
}
