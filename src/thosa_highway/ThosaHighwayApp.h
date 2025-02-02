//
// Copyright (C) 2020 Dominik S. Buse <buse@ccs-labs.org>
//
// Documentation for these modules is at http://veins.car2x.org/
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//

#pragma once

#include "veins/veins.h"

#include "veins/base/modules/BaseApplLayer.h"
#include "veins/modules/utility/TimerManager.h"

namespace veins {

class BaseMobility;

namespace thosa_highway {

class ThosaHighwayApp : public BaseApplLayer {
public:
    ~ThosaHighwayApp() override = default;

    // OMNeT++ module interface implementation
    void initialize(int stage) override;
    void finish() override;
    void handleSelfMsg(cMessage* msg) override;

    // CAM sending
    void beacon();
    void handleLowerMsg(cMessage* msg) override;

protected:
    TimerManager timerManager{this};
    BaseMobility* mobility;

private:
    bool isFollower = false;
};

} // namespace serpentine
} // namespace veins
