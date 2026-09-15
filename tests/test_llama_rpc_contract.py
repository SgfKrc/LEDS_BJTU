from src.llama_rpc_contract import RpcShardLeaseBook


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
    assert stale.reason in {"stale_lease", "stale_epoch"}
    assert accepted.accepted is True
    assert accepted.result_digest
    assert duplicate.accepted is False
    assert duplicate.reason == "lease_committed"


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
