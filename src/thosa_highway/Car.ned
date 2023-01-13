//
// Copyright (C) 2019 Max Schettler <schettler@ccs-labs.org>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
//


// Adapt NR Car module from simu5g into simu5g_gym


package thosa_highway;

import inet.applications.contract.IApp;
//import inet.mobility.contract.IMobility;
import inet.networklayer.common.InterfaceTable;
import inet.networklayer.contract.IRoutingTable;
import inet.networklayer.contract.INetworkLayer;
import inet.networklayer.configurator.ipv4.HostAutoConfigurator;
import inet.transportlayer.tcp.Tcp;
import inet.transportlayer.udp.Udp;
import inet.common.MessageDispatcher;
//import simu5g.nodes.Ue;
//import simu5g.nodes.NR.NRUe;
import simu5g.nodes.UeGym;
import simu5g.nodes.NR.NRUeGym;

import org.car2x.veins.base.modules.*;

// import org.car2x.veins.base.modules.IBaseApplLayer;
// import org.car2x.veins.modules.mobility.traci.TraCIMobility;
// import org.car2x.veins.modules.nic.Nic80211p;
// 
// import org.car2x.veinsvlc.NicVlc;
// import org.car2x.veinsvlc.ISplitter;

//
// Car module
//

module CarGym extends UeGym
{
    parameters:

        @display("i=device/car");
        
        //# Mobility
        //string veinsmobilityType = default("org.car2x.veins.modules.mobility.traci.TraCIMobility")
        //mobility.typename = default("VeinsInetMobility");

        //# Network Layer specs
        *.routingTableModule = default(absPath(".ipv4.routingTable"));
        *.interfaceTableModule = default(absPath(".interfaceTable"));
        
        ipv4.configurator.typename = "HostAutoConfigurator";
        ipv4.configurator.interfaces = "cellular lo0";
}


// 
// Car Module
//
module NRCarGym extends NRUeGym
{
    parameters:

        @display("i=device/car");

        //# Mobility
        //mobility.typename = default("VeinsInetMobility");

        //# Network Layer specs
        *.routingTableModule = default(absPath(".ipv4.routingTable"));
        
        ipv4.configurator.typename = "HostAutoConfigurator";
        ipv4.configurator.interfaces = "cellular lo0";
}
// module Car
// {
//     parameters:
//         string applType; //type of the application layer
//         string splitterType;
//         string veinsmobilityType = default("org.car2x.veins.modules.mobility.traci.TraCIMobility"); //type of the mobility module
// 
//     gates:
//         input radioIn; // gate for sendDirect
//         input headLightIn;
//         input tailLightIn;
//     submodules:
//         appl: <applType> like org.car2x.veins.base.modules.IBaseApplLayer {
//             parameters:
//                 @display("p=60,50");
//         }
// 
//         splitter: <splitterType>  like ISplitter {
//             @display("p=163,127");
//         }
// 
//         nic: Nic80211p {
//             parameters:
//                 @display("p=60,400");
//         }
// 
//         nicVlcHead: NicVlc {
//             parameters:
//                 @display("p=163,243");
//         }
// 
//         nicVlcTail: NicVlc {
//             parameters:
//                 @display("p=253,243");
//         }
// 
//         veinsmobility: <veinsmobilityType> like org.car2x.veins.base.modules.IMobility {
//             parameters:
//                 @display("p=130,172;i=block/cogwheel");
//         }
// 
//     connections allowunconnected:
//         appl.lowerLayerOut --> splitter.applicationIn;
//         appl.lowerLayerIn <-- splitter.applicationOut;
// 
//         splitter.nicOut --> nic.upperLayerIn;
//         splitter.nicIn <-- nic.upperLayerOut;
//         splitter.nicVlcHeadOut --> nicVlcHead.upperLayerIn;
//         splitter.nicVlcHeadIn <-- nicVlcHead.upperLayerOut;
//         splitter.nicVlcTailOut --> nicVlcTail.upperLayerIn;
//         splitter.nicVlcTailIn <-- nicVlcTail.upperLayerOut;
// 
//         nic.radioIn <-- radioIn;
//         nicVlcHead.radioIn <-- headLightIn;
//         nicVlcTail.radioIn <-- tailLightIn;
// }
