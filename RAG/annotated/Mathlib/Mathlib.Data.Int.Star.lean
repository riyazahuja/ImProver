@[simp] lemma addSubmonoid_closure_range_pow {n : ℕ} (hn : Even n) :
    closure (range fun x : ℤ ↦ x ^ n) = nonneg _ := by
  /-
    n : Nat
    hn : Even n
    ⊢ Eq (AddSubmonoid.closure (Set.range fun x => HPow.hPow x n)) (AddSubmonoid.n …
  -/
  refine le_antisymm (closure_le.2 <| range_subset_iff.2 hn.pow_nonneg) fun x hx ↦ ?_
  /-
    n : Nat
    hn : Even n
    x : Int
    hx : Membership.mem (AddSubmonoid.nonneg Int) x
    ⊢ Membership.mem (AddSubmonoid.closure (Set.range fun x => HPow.hPow x n)) x
  -/
  have : x = x.natAbs • 1 ^ n := by simpa [eq_comm (a := x)] using hx
  /-
    n : Nat
    hn : Even n
    x : Int
    hx : Membership.mem (AddSubmonoid.nonneg Int) x
    this : Eq x (HSMul.hSMul x.natAbs (HPow.hPow 1 n))
    ⊢ Membership.mem (AddSubmonoid.closure (Set.range fun x => HPow.hPow x n)) x
  -/
  rw [this]
  /-
    n : Nat
    hn : Even n
    x : Int
    hx : Membership.mem (AddSubmonoid.nonneg Int) x
    this : Eq x (HSMul.hSMul x.natAbs (HPow.hPow 1 n))
    ⊢ Membership.mem (AddSubmonoid.closure (Set.range fun x => HPow.hPow x n)) (HS …
  -/
  exact nsmul_mem (subset_closure <| mem_range_self _) _
  /-
    🎉 no goals
  -/


@[simp]
lemma addSubmonoid_closure_range_mul_self : closure (range fun x : ℤ ↦ x * x) = nonneg _ := by
  /-
    ⊢ Eq (AddSubmonoid.closure (Set.range fun x => HMul.hMul x x)) (AddSubmonoid.n …
  -/
  simpa only [sq] using addSubmonoid_closure_range_pow even_two
  /-
    🎉 no goals
  -/


instance instStarOrderedRing : StarOrderedRing ℤ where
                   /-
                     a b : Int
                     ⊢ Iff (LE.le a b) (Exists fun p => And (Membership.mem (AddSubmonoid.closure ( …
                   -/
  le_iff a b := by simp [eq_comm, le_iff_exists_nonneg_add (a := a)]
                   /-
                     🎉 no goals
                   -/


