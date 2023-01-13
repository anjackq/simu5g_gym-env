// 
//                  Simu5G
//
// Authors: Giovanni Nardini, Giovanni Stea, Antonio Virdis (University of Pisa)
// 
// This file is part of a software released under the license included in file
// "license.pdf". Please read LICENSE and README files before using it.
// The above files and the present reference are part of the software itself, 
// and cannot be removed from it.
//

//package simu5g.simulations.NR.ai4mobile_thosa;

import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.networklayer.ipv4.RoutingTableRecorder;
import inet.node.inet.AdhocHost;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import inet.node.ethernet.Eth10G;

import simu5g.world.radio.LteChannelControl;
import simu5g.common.carrierAggregation.CarrierAggregation;
import simu5g.nodes.Upf;
import simu5g.common.binder.Binder;

//import simu5g.nodes.NR.gNodeB;
//import simu5g.nodes.cars.NRCar;
import simu5g.nodes.NR.gNodeB;
import simu5g.nodes.cars.NRCarGym;


// Replace teh VeinsInetManager with TraCImanager, including related modules
// import org.car2x.veins.subprojects.veins_inet.VeinsInetManager;
//import org.car2x.veins.base.connectionManager.ConnectionManager;
import org.car2x.veins.base.modules.BaseWorldUtility;
import org.car2x.veins.modules.mobility.traci.TraCIScenarioManager*;
// import org.car2x.veins.modules.obstacle.ObstacleControl;
// import org.car2x.veins.modules.obstacle.VehicleObstacleControl;
//import org.car2x.veins.modules.world.annotations.AnnotationManager;
//import org.car2x.veins.nodes.Car;
// import org.car2x.veinsvlc.VlcConnectionManager;

//import thosa_highway.GymConnection;

network ThosaHighwayScenario
{
    parameters:
        double playgroundSizeX @unit(m); // x size of the area the nodes are in (in meters)
        double playgroundSizeY @unit(m); // y size of the area the nodes are in (in meters)
        double playgroundSizeZ @unit(m); // z size of the area the nodes are in (in meters)
        //@display("bgb=$play1500,1500");
        @display("bgb=$playgroundSizeX,$playgroundSizeY");

    submodules:

        routingRecorder: RoutingTableRecorder {
            @display("p=50,75;is=s");
        }
        configurator: Ipv4NetworkConfigurator {
            @display("p=50,125");
            config = xmldoc("demo.xml");
        }
        
        
//         connectionManager: ConnectionManager {
//             parameters:
//                 @display("p=150,0;i=abstract/multicast");
//         }
        world: BaseWorldUtility {
            parameters:
                playgroundSizeX = veins_eval_by_version(veins_omnetpp_buildnum(), "playgroundSizeX", 1525, "parent.playgroundSizeX");
                playgroundSizeY = veins_eval_by_version(veins_omnetpp_buildnum(), "playgroundSizeY", 1525, "parent.playgroundSizeY");
                playgroundSizeZ = veins_eval_by_version(veins_omnetpp_buildnum(), "playgroundSizeZ", 1525, "parent.playgroundSizeZ");
                @display("p=30,0;i=misc/globe");
        }
        
        //# Obstacle module
//         obstacles: ObstacleControl {
//             @display("p=240,50");
//         }
//         vehicleObstacles: VehicleObstacleControl {}
//         annotations: AnnotationManager {
//             @display("p=260,50");
//         }

        //# Veins manager module
//         veinsManager: VeinsInetManager {
//             @display("p=50,227;is=s");
//         }

        // # Veins manager module, replace the VeinsInetManager with TraCIManager
        manager: TraCIScenarioManagerForker {
            parameters:
                @display("p=512,128");
        }
        
        //# Gym modules
        //gym_connection: GymConnection {}

        //# LTE modules
        channelControl: LteChannelControl {
            @display("p=50,25;is=s");
        }
        binder: Binder {
            @display("p=50,175;is=s");
        }
        carrierAggregation: CarrierAggregation {
            @display("p=50.993748,258.7;is=s");
        }
        server: StandardHost {
            @display("p=660,136;is=n;i=device/server");
        }
        router: Router {
            @display("p=561,135;i=device/smallrouter");
        }
        upf: Upf {
            @display("p=462,136");
        }
        gNodeB1: gNodeB {
            @display("p=500,500;is=vl");
        }
        gNodeB2: gNodeB {
            @display("p=4500,-500;is=vl");
        }
        
        //# submodule vector to store all cars - will be filled dynamically during simulations
        node[0]: NRCarGym;
        
        
        connections allowunconnected:
        server.pppg++ <--> Eth10G <--> router.pppg++;
        router.pppg++ <--> Eth10G <--> upf.filterGate;
        upf.pppg++ <--> Eth10G <--> gNodeB1.ppp;
        upf.pppg++ <--> Eth10G <--> gNodeB2.ppp;

        //# X2 connections
        gNodeB1.x2++ <--> Eth10G <--> gNodeB2.x2++;
}

