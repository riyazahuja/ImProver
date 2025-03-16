/-- Lazily deduplicate a lazy list, using a stored `HashMap`. -/
-- We choose `HashMap` here instead of `RBSet` as the initial application is `Expr`.
@[deprecated "See deprecation note in module documentation." (since := "2024-08-22")]
def dedupBy (L : MLList m α) (f : α → m β) : MLList m α :=
  ((L.liftM : MLList (StateT (HashMap β Unit) m) α) >>= fun a => do
      let b ← f a
      guard !(← get).contains b
      modify fun s => s.insert b ()
      pure a)
  |>.runState' ∅


/-- Lazily deduplicate a lazy list, using a stored `HashMap`. -/
@[deprecated "See deprecation note in module documentation." (since := "2024-08-22")]
def dedup (L : MLList m β) : MLList m β := L.dedupBy pure


