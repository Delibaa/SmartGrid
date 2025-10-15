// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;

contract EnergyMarket {
    // info struct for bids and asks
    struct Info {
        address owner;
        uint256 price;
        uint256 amount;
        uint256 date;
        bytes32 orderId; // unique hash ID
    }

    Info[] public Bids; // demand info
    Info[] public Asks; // supply info

    // ---- Reads ----
    // get bid by index
    function getBid(
        uint256 index
    ) public view returns (address, uint256, uint256, uint256, bytes32) {
        Info storage b = Bids[index];
        return (b.owner, b.price, b.amount, b.date, b.orderId);
    }

    // get ask by index
    function getAsk(
        uint256 index
    ) public view returns (address, uint256, uint256, uint256, bytes32) {
        Info storage a = Asks[index];
        return (a.owner, a.price, a.amount, a.date, a.orderId);
    }

    // get number of bids
    function getBidsCount() public view returns (uint256) {
        return Bids.length;
    }

    // get number of asks
    function getAsksCount() public view returns (uint256) {
        return Asks.length;
    }

    // ---- Writes ----
    // add bid
    function addBid(
        uint256 price,
        uint256 amount,
        uint256 timestamp,
        bytes32 orderId
    ) public returns (bool) {
        Bids.push(
            Info({
                owner: msg.sender,
                price: price,
                amount: amount,
                date: timestamp,
                orderId: orderId
            })
        );
        return true;
    }

    // add ask
    function addAsk(
        uint256 price,
        uint256 amount,
        uint256 timestamp,
        bytes32 orderId
    ) public returns (bool) {
        Asks.push(
            Info({
                owner: msg.sender,
                price: price,
                amount: amount,
                date: timestamp,
                orderId: orderId
            })
        );
        return true;
    }

    // update bid amount by orderId
    function updateBidAmount(bytes32 orderId, uint256 newAmount) public returns (bool) {
        for (uint256 i = 0; i < Bids.length; i++) {
            if (Bids[i].orderId == orderId) {
                require(Bids[i].owner == msg.sender, "not order owner");
                Bids[i].amount = newAmount;
                return true;
            }
        }
        return false;
    }

    // update ask amount by orderId
    function updateAskAmount(bytes32 orderId, uint256 newAmount) public returns (bool) {
        for (uint256 i = 0; i < Asks.length; i++) {
            if (Asks[i].orderId == orderId) {
                require(Asks[i].owner == msg.sender, "not order owner");
                Asks[i].amount = newAmount;
                return true;
            }
        }
        return false;
    }

    // remove bid by orderId
    function removeBid(bytes32 orderId) public returns (bool) {
        for (uint256 i = 0; i < Bids.length; i++) {
            if (Bids[i].orderId == orderId) {
                require(Bids[i].owner == msg.sender, "not order owner");
                uint256 last = Bids.length - 1;
                if (i != last) {
                    Bids[i] = Bids[last]; // swap with last
                }
                Bids.pop();
                return true;
            }
        }
        return false;
    }

    // remove ask by orderId
    function removeAsk(bytes32 orderId) public returns (bool) {
        for (uint256 i = 0; i < Asks.length; i++) {
            if (Asks[i].orderId == orderId) {
                require(Asks[i].owner == msg.sender, "not order owner");
                uint256 last = Asks.length - 1;
                if (i != last) {
                    Asks[i] = Asks[last];
                }
                Asks.pop();
                return true;
            }
        }
        return false;
    }

    // clear all bids and asks
    function clearMarket() public {
        delete Bids;
        delete Asks;
    }
}

