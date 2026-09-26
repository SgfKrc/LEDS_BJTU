from llama_rpc_contract import RpcShardLeaseBook
from cluster_fence import ControlFence
from cluster_control_contract import (
    ControlPlaneAuthority, QuorumCertificate, VoterIdentity, VoterSet,
    sign_certificate,
)


def _fence():
    import base64
    import time
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    keys = [Ed25519PrivateKey.generate() for _ in range(3)]
    voters = []
    for index, key in enumerate(keys):
        public = base64.urlsafe_b64encode(key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )).decode("ascii").rstrip("=")
        voters.append(VoterIdentity(voter_id=f"voter-{index}", public_key=public))
    voter_set = VoterSet(cluster_id="fence-test", voter_set_epoch=1, voters=voters)
    unsigned = QuorumCertificate(
        cluster_id="fence-test", voter_set_epoch=1, term=3,
        leader_id="voter-0", lease_id="lease-3",
        expires_at_ms=int(time.time() * 1000) + 60_000, signatures=tuple(),
    )
    certificate = unsigned.with_signatures(tuple(
        sign_certificate(unsigned, voter_id=f"voter-{index}", private_key=keys[index])
        for index in (0, 1)
    ))
    authority = ControlPlaneAuthority(cluster_id=voter_set.cluster_id, voter_set=voter_set)
    fence = ControlFence(authority, required=True)
    fence.install_certificate(certificate)
    return fence, certificate


def test_reassignment_fences_old_epoch_and_accepts_new_worker():
    book = RpcShardLeaseBook()
    first = book.assign("shard-0", "rpc-a", "abc", {"gpu_layers": 1})
    second = book.reassign("shard-0", "local-fallback", "abc", {"gpu_layers": 0})

    stale = book.commit(first.lease_id, first.epoch, "old")
    accepted = book.commit(second.lease_id, second.epoch, "new")
    duplicate = book.commit(second.lease_id, second.epoch, "new-again")

    assert second.epoch == first.epoch + 1
    assert second.attempt == first.attempt + 1
    assert stale.accepted is False
    # ★ 精确断言（R-R6 复盘，2026-09-25）：`reassign` 必然换 lease_id，而 `commit` 里
    #   **lease_id 先于 epoch 比较**（`src/llama_rpc_contract.py:184-187`）⇒ 这里只能是 `stale_lease`。
    #   原先的 `in {"stale_lease", "stale_epoch"}` 是宽松集合，等于放过了"到底命中哪个分支"。
    assert stale.reason == "stale_lease"
    assert accepted.accepted is True
    assert accepted.result_digest
    assert duplicate.accepted is False
    assert duplicate.reason == "lease_committed"


def test_commit_rejects_stale_epoch_on_the_current_lease():
    """★ 补 `commit` 路径上的 `stale_epoch` 断言（R-R6 复盘，2026-09-25）。

    为什么必须单独钉住：`commit`/`renew`/`check` 都是**先比 lease_id、再比 epoch**
    （`src/llama_rpc_contract.py:184-187`），而 epoch 与 lease_id 由**同一次 `assign` 成对产生**、
    `renew` 又不改 epoch ⇒「旧 lease_id + 旧 epoch」永远命中 `stale_lease`，**到不了 `stale_epoch`**。
    探针（`scripts/llama_pc_rpc.py`）与 `tests/test_llama_pc_rpc*.py` 都只传**配对的**
    `(lease_id, epoch)` ⇒ 该分支在端到端路径上**不可达**，只能在此钉住。
    全仓此前唯一精确断言 `stale_epoch` 的地方是 `check` 路径（本文件后面的用例）。
    """
    book = RpcShardLeaseBook()
    first = book.assign("shard-0", "rpc-a", "abc", {})                  # epoch=1
    second = book.reassign("shard-0", "local-fallback", "abc", {})      # epoch=2

    # (a) 旧 lease_id ⇒ 先撞 lease_id 比较 ⇒ stale_lease（无论 epoch 传哪个）
    old_lease = book.commit(first.lease_id, second.epoch, "old-lease")
    assert old_lease.accepted is False
    assert old_lease.reason == "stale_lease"

    # (b) **当前 lease_id + 过期 epoch** ⇒ 唯一能命中 commit 的 stale_epoch 分支
    stale_epoch = book.commit(second.lease_id, first.epoch, "old-epoch")
    assert stale_epoch.accepted is False
    assert stale_epoch.reason == "stale_epoch"
    assert stale_epoch.lease is not None
    assert stale_epoch.lease.lease_id == second.lease_id     # 拒绝时回传的是 **current**
    assert stale_epoch.result_digest == ""                   # 拒绝不产出 digest

    # (c) fencing 之后当前 lease 仍可提交 —— 防止负向断言过宽（把"全都拒绝"当成通过）
    assert book.commit(second.lease_id, second.epoch, "ok").accepted is True


def test_renew_rejects_old_lease_after_reassignment():
    book = RpcShardLeaseBook()
    first = book.assign("shard-0", "rpc-a", "abc", {})
    second = book.reassign("shard-0", "rpc-b", "abc", {})

    assert book.renew(first.lease_id, first.epoch).reason == "stale_lease"
    assert book.renew(second.lease_id, second.epoch).accepted is True


def test_renew_extends_an_active_lease():
    book = RpcShardLeaseBook()
    lease = book.assign("shard-0", "rpc-a", "abc", {}, lease_seconds=5)

    renewed = book.renew(lease.lease_id, lease.epoch)

    assert renewed.accepted is True
    assert renewed.lease.lease_expires_at > lease.lease_expires_at
    assert renewed.lease.lease_ttl_seconds == 5


def test_read_only_fencing_check_rejects_reassigned_lease():
    book = RpcShardLeaseBook()
    first = book.assign("shard-0", "rpc-a", "abc", {})

    assert book.check(first.lease_id, first.epoch).reason == "current"
    second = book.reassign("shard-0", "rpc-b", "abc", {})

    assert book.check(first.lease_id, first.epoch).reason == "stale_lease"
    assert book.check(second.lease_id, first.epoch).reason == "stale_epoch"
    assert book.check(second.lease_id, second.epoch).accepted is True


def test_control_term_is_bound_to_lease_renew_and_commit():
    fence, certificate = _fence()
    book = RpcShardLeaseBook(control_fence=fence)
    lease = book.assign("shard-0", "rpc-a", "abc", {}, certificate=certificate)

    assert lease.control_term == certificate.term
    assert lease.certificate_digest == certificate.digest()
    assert book.renew(lease.lease_id, lease.epoch, certificate=certificate).accepted is True
    assert book.commit(lease.lease_id, lease.epoch, "result", certificate=certificate).accepted is True
