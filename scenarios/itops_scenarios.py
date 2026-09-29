"""
50 realistic ITOps incident scenarios.

Each scenario maps to a concrete ITState that the agent will observe and act on.
Scenarios span all 7 alert types across diverse service/load/memory conditions.
"""
from dataclasses import dataclass
from agent.environment import AlertType, ServiceStatus, LoadLevel, ITState, OPTIMAL_ACTION


@dataclass
class Scenario:
    id: int
    name: str
    description: str          # natural-language context shown in LLM prompt
    state: ITState
    optimal_action: str       # ground-truth best action


def _s(
    sid: int,
    name: str,
    desc: str,
    alert: AlertType,
    status: ServiceStatus,
    load: LoadLevel,
    cpu: float,
    mem: float,
    prev: str = "no_op",
) -> Scenario:
    state = ITState(
        alert_type=alert,
        service_status=status,
        load_level=load,
        cpu_usage=cpu,
        memory_usage=mem,
        previous_action=prev,
    )
    return Scenario(
        id=sid,
        name=name,
        description=desc,
        state=state,
        optimal_action=OPTIMAL_ACTION[alert],
    )


A = AlertType
S = ServiceStatus
L = LoadLevel

SCENARIOS: list[Scenario] = [

    # ── NONE (5) — stable system, correct action is no_op ──────────────────
    _s(1,  "Healthy baseline",
       "All services nominal. CPU 20%, memory 35%. No alerts.",
       A.NONE, S.RUNNING, L.LOW, 0.20, 0.35),

    _s(2,  "Post-deployment steady state",
       "New release deployed 10 min ago. Metrics stable. No anomalies detected.",
       A.NONE, S.RUNNING, L.MEDIUM, 0.40, 0.50),

    _s(3,  "Overnight quiet period",
       "Off-peak hours. All services green. Monitoring shows zero errors.",
       A.NONE, S.RUNNING, L.LOW, 0.15, 0.28),

    _s(4,  "Maintenance window — services OK",
       "Scheduled maintenance completed. All health checks passing.",
       A.NONE, S.RUNNING, L.MEDIUM, 0.35, 0.45),

    _s(5,  "Warm-up after cold start",
       "Services restarted 5 min ago. Load building normally. No alerts.",
       A.NONE, S.RUNNING, L.LOW, 0.30, 0.42),

    # ── HIGH_CPU (8) — optimal: scale_up ───────────────────────────────────
    _s(6,  "Web tier CPU saturation",
       "Web servers at 92% CPU. Response times degraded. Traffic spike from campaign.",
       A.HIGH_CPU, S.DEGRADED, L.HIGH, 0.92, 0.60),

    _s(7,  "ML inference CPU bound",
       "Inference service pegged at 95% CPU. GPU not saturated. Need more CPU workers.",
       A.HIGH_CPU, S.RUNNING, L.HIGH, 0.95, 0.55),

    _s(8,  "Background job CPU runaway",
       "Nightly ETL job consuming 88% CPU on all nodes. Starving API layer.",
       A.HIGH_CPU, S.DEGRADED, L.CRITICAL, 0.88, 0.70),

    _s(9,  "Search indexer overload",
       "Elasticsearch reindexing using 91% CPU. Query latency up 5x.",
       A.HIGH_CPU, S.RUNNING, L.HIGH, 0.91, 0.62),

    _s(10, "API gateway CPU spike",
       "Gateway showing 87% CPU. TLS handshake rate doubled post-traffic surge.",
       A.HIGH_CPU, S.DEGRADED, L.HIGH, 0.87, 0.50),

    _s(11, "Video transcoding saturated",
       "Transcoding workers at 94% CPU. Queue depth growing. Need more instances.",
       A.HIGH_CPU, S.RUNNING, L.CRITICAL, 0.94, 0.65),

    _s(12, "Report generation spike",
       "End-of-month report jobs driving 89% CPU across cluster.",
       A.HIGH_CPU, S.RUNNING, L.HIGH, 0.89, 0.58),

    _s(13, "Ad auction bidder CPU bound",
       "Bidder service at 93% CPU during peak RTB traffic. Latency SLA at risk.",
       A.HIGH_CPU, S.DEGRADED, L.CRITICAL, 0.93, 0.55),

    # ── MEMORY_LEAK (8) — optimal: restart_service ─────────────────────────
    _s(14, "Java heap exhaustion",
       "Java app memory growing 200MB/hr. Heap at 90%. GC pause times increasing.",
       A.MEMORY_LEAK, S.DEGRADED, L.MEDIUM, 0.55, 0.90),

    _s(15, "Node.js uncollected closures",
       "Node process memory at 85% after 12h uptime. Classic closure leak pattern.",
       A.MEMORY_LEAK, S.RUNNING, L.MEDIUM, 0.50, 0.85),

    _s(16, "Python worker RSS growing",
       "Celery workers RSS growing 50MB/hr. No GC pressure but memory not freed.",
       A.MEMORY_LEAK, S.RUNNING, L.LOW, 0.40, 0.82),

    _s(17, "Cache server memory leak",
       "Redis-proxy process memory at 88%. Connection leak suspected.",
       A.MEMORY_LEAK, S.DEGRADED, L.HIGH, 0.60, 0.88),

    _s(18, "gRPC server stream leak",
       "gRPC service memory at 91%. Streaming RPCs not releasing buffers on close.",
       A.MEMORY_LEAK, S.DEGRADED, L.MEDIUM, 0.48, 0.91),

    _s(19, "Session store bloat",
       "Auth service memory at 87%. Sessions not being evicted correctly.",
       A.MEMORY_LEAK, S.RUNNING, L.MEDIUM, 0.52, 0.87),

    _s(20, "WebSocket server memory growth",
       "WS server memory at 83% after 8h. Connection objects not freed on disconnect.",
       A.MEMORY_LEAK, S.RUNNING, L.HIGH, 0.65, 0.83),

    _s(21, "Sidecar proxy memory creep",
       "Envoy sidecar using 86% memory. Buffer leak in HTTP/2 multiplexing.",
       A.MEMORY_LEAK, S.DEGRADED, L.MEDIUM, 0.45, 0.86),

    # ── SERVICE_CRASH (8) — optimal: restart_service ───────────────────────
    _s(22, "Payment service OOM killed",
       "Payment microservice killed by OOM killer. Pod in CrashLoopBackOff.",
       A.SERVICE_CRASH, S.DOWN, L.HIGH, 0.70, 0.95),

    _s(23, "Auth service segfault",
       "Auth service crashed with SIGSEGV. Core dump generated. Service down.",
       A.SERVICE_CRASH, S.DOWN, L.MEDIUM, 0.30, 0.40),

    _s(24, "Worker process exit code 1",
       "Background worker exited unexpectedly. Non-zero exit. Jobs not processing.",
       A.SERVICE_CRASH, S.DOWN, L.LOW, 0.20, 0.35),

    _s(25, "API server panic",
       "Go API server panicked on nil pointer dereference. Goroutines dumped.",
       A.SERVICE_CRASH, S.DOWN, L.HIGH, 0.55, 0.50),

    _s(26, "Scheduler process died",
       "Job scheduler process terminated. No jobs running. Heartbeat missed.",
       A.SERVICE_CRASH, S.DOWN, L.LOW, 0.25, 0.30),

    _s(27, "Database connection pool crash",
       "Connection pool manager crashed. All DB connections dropped. Service down.",
       A.SERVICE_CRASH, S.DOWN, L.CRITICAL, 0.80, 0.75),

    _s(28, "Load balancer health check fail",
       "LB reporting backend as unhealthy. Service process not responding on port.",
       A.SERVICE_CRASH, S.DOWN, L.HIGH, 0.60, 0.55),

    _s(29, "Notification service SIGKILL",
       "Notification worker SIGKILLed by container runtime. Pending notifications lost.",
       A.SERVICE_CRASH, S.DOWN, L.MEDIUM, 0.40, 0.45),

    # ── NETWORK_LATENCY (7) — optimal: investigate ─────────────────────────
    _s(30, "Cross-AZ latency spike",
       "p99 latency between AZ-A and AZ-B jumped from 2ms to 45ms. Intermittent.",
       A.NETWORK_LATENCY, S.DEGRADED, L.MEDIUM, 0.50, 0.55),

    _s(31, "Database round-trip degraded",
       "DB query round-trip up from 5ms to 120ms. CPU normal. Network issue suspected.",
       A.NETWORK_LATENCY, S.RUNNING, L.MEDIUM, 0.45, 0.50),

    _s(32, "CDN origin latency",
       "CDN origin requests timing out intermittently. 5% error rate on edge nodes.",
       A.NETWORK_LATENCY, S.RUNNING, L.HIGH, 0.55, 0.60),

    _s(33, "Service mesh routing delay",
       "Istio sidecar adding 80ms latency. VirtualService misconfiguration suspected.",
       A.NETWORK_LATENCY, S.DEGRADED, L.MEDIUM, 0.48, 0.52),

    _s(34, "External API timeout",
       "Third-party payment gateway p95 latency at 8s. Internal services waiting.",
       A.NETWORK_LATENCY, S.RUNNING, L.HIGH, 0.60, 0.58),

    _s(35, "DNS resolution slowdown",
       "DNS TTL expired. Resolver taking 3s per lookup. Service discovery impacted.",
       A.NETWORK_LATENCY, S.RUNNING, L.MEDIUM, 0.42, 0.48),

    _s(36, "TCP retransmit storm",
       "Network interface showing 15% packet retransmit rate. MTU mismatch suspected.",
       A.NETWORK_LATENCY, S.DEGRADED, L.HIGH, 0.65, 0.60),

    # ── DISK_FULL (7) — optimal: clear_disk ────────────────────────────────
    _s(37, "Log partition 98% full",
       "Application log partition at 98%. Log writes starting to fail. Service slowing.",
       A.DISK_FULL, S.DEGRADED, L.MEDIUM, 0.55, 0.60),

    _s(38, "Docker overlay full",
       "Docker overlay2 at 97%. Container image pulls failing. Builds broken.",
       A.DISK_FULL, S.DEGRADED, L.HIGH, 0.65, 0.65),

    _s(39, "Core dump filling disk",
       "Repeated crashes leaving core dumps. /var/crash at 99%. New crashes failing.",
       A.DISK_FULL, S.DOWN, L.HIGH, 0.70, 0.55),

    _s(40, "Temp file accumulation",
       "/tmp at 96%. Long-running jobs not cleaning up. New jobs failing to start.",
       A.DISK_FULL, S.DEGRADED, L.MEDIUM, 0.50, 0.50),

    _s(41, "Database WAL files",
       "Postgres WAL archive disk at 95%. Archiving lag 2h. Checkpoint blocked.",
       A.DISK_FULL, S.DEGRADED, L.CRITICAL, 0.75, 0.70),

    _s(42, "Audit log partition full",
       "Audit log volume at 99%. Compliance writes failing. Regulatory risk.",
       A.DISK_FULL, S.RUNNING, L.LOW, 0.35, 0.45),

    _s(43, "Build artifact storage full",
       "CI/CD artifact storage at 98%. New builds cannot push artifacts.",
       A.DISK_FULL, S.RUNNING, L.MEDIUM, 0.50, 0.55),

    # ── DEPENDENCY_FAILURE (7) — optimal: rollback ─────────────────────────
    _s(44, "Bad library upgrade breaks API",
       "Upstream library v2.1 introduced breaking change. API returning 500s since deploy.",
       A.DEPENDENCY_FAILURE, S.DEGRADED, L.HIGH, 0.70, 0.60),

    _s(45, "Config service incompatibility",
       "New config-server version returns unexpected schema. 40% of services misconfigured.",
       A.DEPENDENCY_FAILURE, S.DEGRADED, L.CRITICAL, 0.80, 0.70),

    _s(46, "Shared DB schema migration broke consumer",
       "DB migration v45 dropped column still read by v2 service. Errors on every request.",
       A.DEPENDENCY_FAILURE, S.DOWN, L.HIGH, 0.65, 0.55),

    _s(47, "Feature flag service regression",
       "Feature flag SDK update flipped a critical flag. Feature broken for all users.",
       A.DEPENDENCY_FAILURE, S.DEGRADED, L.HIGH, 0.68, 0.60),

    _s(48, "Message queue protocol change",
       "Kafka client upgrade changed serialization. Consumers failing to deserialize.",
       A.DEPENDENCY_FAILURE, S.DEGRADED, L.CRITICAL, 0.75, 0.65),

    _s(49, "Auth library JWT signing change",
       "JWT signing algorithm changed in auth lib. All tokens invalidated. Login broken.",
       A.DEPENDENCY_FAILURE, S.DOWN, L.CRITICAL, 0.85, 0.70),

    _s(50, "gRPC proto breaking change",
       "Proto field removed in v3.2.0. Old clients sending removed field. Service rejecting.",
       A.DEPENDENCY_FAILURE, S.DEGRADED, L.HIGH, 0.72, 0.62),
]

assert len(SCENARIOS) == 50, f"Expected 50 scenarios, got {len(SCENARIOS)}"
