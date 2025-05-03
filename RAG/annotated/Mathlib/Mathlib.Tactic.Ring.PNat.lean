instance : CSLift ℕ+ Nat where
  lift := PNat.val
  inj := PNat.coe_injective

-- FIXME: this `no_index` seems to be in the wrong place, but
-- #synth CSLiftVal (3 : ℕ+) _ doesn't work otherwise

instance {n} : CSLiftVal (no_index (OfNat.ofNat (n+1)) : ℕ+) (n + 1) := ⟨rfl⟩


instance {n h} : CSLiftVal (Nat.toPNat n h) n := ⟨rfl⟩


instance {n} : CSLiftVal (Nat.succPNat n) (n + 1) := ⟨rfl⟩


instance {n} : CSLiftVal (Nat.toPNat' n) (n.pred + 1) := ⟨rfl⟩


instance {n k} : CSLiftVal (PNat.divExact n k) (n.div k + 1) := ⟨rfl⟩


instance {n n' k k'} [h1 : CSLiftVal (n : ℕ+) n'] [h2 : CSLiftVal (k : ℕ+) k'] :
                                       /-
                                         n : PNat
                                         n' : Nat
                                         k : PNat
                                         k' : Nat
                                         h1 : Mathlib.Tactic.Ring.CSLiftVal n n'
                                         h2 : Mathlib.Tactic.Ring.CSLiftVal k k'
                                         ⊢ Eq (HAdd.hAdd n' k') (Mathlib.Tactic.Ring.CSLift.lift (HAdd.hAdd n k))
                                       -/
    CSLiftVal (n + k) (n' + k') := ⟨by simp [h1.1, h2.1, CSLift.lift]⟩
                                       /-
                                         🎉 no goals
                                       -/


instance {n n' k k'} [h1 : CSLiftVal (n : ℕ+) n'] [h2 : CSLiftVal (k : ℕ+) k'] :
                                       /-
                                         n : PNat
                                         n' : Nat
                                         k : PNat
                                         k' : Nat
                                         h1 : Mathlib.Tactic.Ring.CSLiftVal n n'
                                         h2 : Mathlib.Tactic.Ring.CSLiftVal k k'
                                         ⊢ Eq (HMul.hMul n' k') (Mathlib.Tactic.Ring.CSLift.lift (HMul.hMul n k))
                                       -/
    CSLiftVal (n * k) (n' * k') := ⟨by simp [h1.1, h2.1, CSLift.lift]⟩
                                       /-
                                         🎉 no goals
                                       -/


instance {n n' k} [h1 : CSLiftVal (n : ℕ+) n'] :
                                      /-
                                        n : PNat
                                        n' k : Nat
                                        h1 : Mathlib.Tactic.Ring.CSLiftVal n n'
                                        ⊢ Eq (HPow.hPow n' k) (Mathlib.Tactic.Ring.CSLift.lift (HPow.hPow n k))
                                      -/
    CSLiftVal (n ^ k) (n' ^ k) := ⟨by simp [h1.1, CSLift.lift]⟩
                                      /-
                                        🎉 no goals
                                      -/


