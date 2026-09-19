# 📡 CyberShield-BigData: REST API Reference

The serving microservice provides high-throughput inference endpoints built on FastAPI.

## Endpoints Summary

### 1. Single Packet Inference
- **Endpoint**: `POST /predict/packet`
- **Request Body**:
```json
{
  "timestamp": 1718000000.0,
  "src_ip": "192.168.1.105",
  "dst_ip": "10.0.0.1",
  "src_port": 54120,
  "dst_port": 80,
  "protocol": 6,
  "packet_length": 1420.0,
  "time_delta": 0.001,
  "header_length": 32.0,
  "window_size": 32120.0,
  "flags": "ACK",
  "payload": "UNION SELECT username, password FROM users--"
}