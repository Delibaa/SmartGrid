// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;

contract ScoreBillboard {
    // so we name the field "err" to store the error value.
    struct Leader {
        uint256 id; // leader id (e.g., house id)
        uint256 day; // day index (1-based or your own convention)
        uint256 err; // error value (e.g., abs(real_total - pred_total))
    }

    Leader[] public Leaders;

    // --- Write APIs ---

    // Append a new leader record.
    function addLeader(
        uint256 id,
        uint256 day,
        uint256 err
    ) public returns (bool) {
        Leaders.push(Leader({id: id, day: day, err: err}));
        return true;
    }

    function getLeaderIdByDay(uint256 day) public view returns (bool found, uint256 id) {
        for (uint256 i = Leaders.length; i > 0; i--) {
            Leader storage L = Leaders[i - 1];
            if (L.day == day) {
                return (true, L.id);
            }
        }
        return (false, 0);
    }

    function getLeadersCount() public view returns (uint256) {
        return Leaders.length;
    }

    function getLeaderByIndex(
        uint256 index
    ) public view returns (uint256 id, uint256 day, uint256 err) {
        Leader storage L = Leaders[index];
        return (L.id, L.day, L.err);
    }

    // Clear the entire leaders array.
    function clearLeaders() public {
        delete Leaders;
    }


}
