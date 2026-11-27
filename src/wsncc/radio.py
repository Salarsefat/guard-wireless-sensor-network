"""PHY/MAC timing constants, link model and radio energy models.

All timing values follow IEEE 802.15.4 for the 2.4 GHz O-QPSK PHY
(62.5 ksymbol/s, 16 us per symbol, 250 kbit/s).
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# IEEE 802.15.4 timing (2.4 GHz O-QPSK)
# ---------------------------------------------------------------------------
SYMBOL = 16e-6                 # s
UNIT_BACKOFF = 20 * SYMBOL     # aUnitBackoffPeriod = 20 symbols = 320 us
T_CCA = 8 * SYMBOL             # CCA detection time = 8 symbols = 128 us
T_TURN = 12 * SYMBOL           # aTurnaroundTime = 12 symbols = 192 us
T_ACK = 22 * SYMBOL            # 11-byte ACK PPDU = 22 symbols = 352 us
T_ACKWAIT = 54 * SYMBOL        # macAckWaitDuration = 20+12+10+12 symbols = 864 us
BITRATE = 250_000.0            # bit/s

MAC_MIN_BE = 3
MAC_MAX_BE = 5
MAC_MAX_CSMA_BACKOFFS = 4
MAC_MAX_FRAME_RETRIES = 3

# ---------------------------------------------------------------------------
# Frame format (bytes)
# ---------------------------------------------------------------------------
PHY_OVERHEAD = 6       # preamble (4) + SFD (1) + PHR (1)
MAC_OVERHEAD = 11      # FCF 2 + seq 1 + PAN 2 + dst 2 + src 2 + FCS 2
NET_HEADER = 4         # frame type, origin, hop count, sequence
READING_BYTES = 20     # one sensor reading (id, timestamp, value, class)
MAX_PSDU = 127         # aMaxPhyPacketSize
ACK_BITS = 11 * 8
BEACON_BITS = (PHY_OVERHEAD + MAC_OVERHEAD + 8) * 8   # 8-byte beacon payload


def frame_bits(n_readings: int, extra_header: int = 0) -> int:
    """Number of bits on air (PPDU) for a data frame carrying n readings."""
    psdu = MAC_OVERHEAD + NET_HEADER + extra_header + READING_BYTES * n_readings
    if psdu > MAX_PSDU:
        raise ValueError(f"PSDU of {psdu} bytes exceeds aMaxPhyPacketSize")
    return (PHY_OVERHEAD + psdu) * 8


def airtime(bits: int) -> float:
    return bits / BITRATE


def max_pack(extra_header: int = 0) -> int:
    """Largest number of readings that fit in one 127-byte PSDU."""
    return (MAX_PSDU - MAC_OVERHEAD - NET_HEADER - extra_header) // READING_BYTES


# Nominal per-frame service cycle of a full-size frame on an idle channel:
# mean initial backoff (3.5 units) + CCA + turnaround + airtime + turnaround + ACK
def service_cycle(bits: int) -> float:
    return 3.5 * UNIT_BACKOFF + T_CCA + T_TURN + airtime(bits) + T_TURN + T_ACK


# ---------------------------------------------------------------------------
# Link model: logistic packet-reception ratio with log-normal shadowing
# ---------------------------------------------------------------------------
def prr_of_distance(d_eff: float, r50: float, width_frac: float) -> float:
    """PRR as a logistic function of (shadowed) distance.

    r50 is the distance at which PRR = 0.5 (the nominal communication range);
    width_frac * r50 sets the width of the transitional region.
    """
    z = (d_eff - r50) / (width_frac * r50)
    if z > 40:
        return 0.0
    if z < -40:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))


# ---------------------------------------------------------------------------
# Energy models
# ---------------------------------------------------------------------------
class FirstOrderEnergy:
    """First-order radio model (free space d^2 / multipath d^4 with crossover d0).

    E_tx(k, d) = k*E_elec + k*eps_fs*d^2   (d <  d0)
                 k*E_elec + k*eps_mp*d^4   (d >= d0)
    E_rx(k)    = k*E_elec
    Idle listening and CCA are not charged (convention of this model family).
    """

    name = "first_order"

    def __init__(self, e_elec: float = 50e-9, eps_fs: float = 10e-12,
                 eps_mp: float = 0.0013e-12) -> None:
        self.e_elec = e_elec
        self.eps_fs = eps_fs
        self.eps_mp = eps_mp
        self.d0 = math.sqrt(eps_fs / eps_mp)
        self.idle_power = 0.0
        self.cca = 0.0
        self.ack_listen = 0.0

    def tx(self, bits: int, d: float) -> float:
        if d < self.d0:
            return bits * (self.e_elec + self.eps_fs * d * d)
        d2 = d * d
        return bits * (self.e_elec + self.eps_mp * d2 * d2)

    def rx(self, bits: int) -> float:
        return bits * self.e_elec


class CC2420Energy:
    """State-based model with CC2420 datasheet currents at a 3 V supply.

    TX 17.4 mA (0 dBm), RX/listen 18.8 mA.  Energy is charged for transmit and
    receive airtime, every CCA, the ACK-wait listening interval after a failed
    transmission and a duty-cycled idle-listening baseline.
    """

    name = "cc2420"

    def __init__(self, voltage: float = 3.0, i_tx: float = 17.4e-3,
                 i_rx: float = 18.8e-3, duty: float = 0.01) -> None:
        self.p_tx = voltage * i_tx
        self.p_rx = voltage * i_rx
        self.idle_power = duty * self.p_rx
        self.cca = self.p_rx * T_CCA
        self.ack_listen = self.p_rx * T_ACKWAIT

    def tx(self, bits: int, d: float) -> float:  # noqa: ARG002 (fixed output power)
        return self.p_tx * bits / BITRATE

    def rx(self, bits: int) -> float:
        return self.p_rx * bits / BITRATE


def make_energy_model(name: str):
    if name == "first_order":
        return FirstOrderEnergy()
    if name == "cc2420":
        return CC2420Energy()
    raise ValueError(f"unknown energy model {name!r}")
