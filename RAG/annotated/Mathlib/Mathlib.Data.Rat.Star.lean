@[simp] lemma addSubmonoid_closure_range_pow {n : ℕ} (hn₀ : n ≠ 0) :
    closure (range fun x : ℚ≥0 ↦ x ^ n) = ⊤ := by
  /-
    n : Nat
    hn₀ : Ne n 0
    ⊢ Eq (AddSubmonoid.closure (Set.range fun x => HPow.hPow x n)) Top.top
  -/
  refine (eq_top_iff' _).2 fun x ↦ ?_
  suffices x = (x.num * x.den ^ (n - 1)) • (x.den : ℚ≥0)⁻¹ ^ n by
    rw [this]
    exact nsmul_mem (subset_closure <| mem_range_self _) _
  /-
    n : Nat
    hn₀ : Ne n 0
    x : NNRat
    ⊢ Eq x (HSMul.hSMul (HMul.hMul x.num (HPow.hPow x.den (HSub.hSub n 1))) (HPow. …
  -/
  rw [nsmul_eq_mul]
  /-
    n : Nat
    hn₀ : Ne n 0
    x : NNRat
    ⊢ Eq x (HMul.hMul (↑(HMul.hMul x.num (HPow.hPow x.den (HSub.hSub n 1)))) (HPow …
  -/
  push_cast
  rw [mul_assoc, pow_sub₀, pow_one, mul_right_comm, ← mul_pow, mul_inv_cancel₀, one_pow, one_mul,
    ← div_eq_mul_inv, num_div_den]
  /-
    n : Nat
    hn₀ : Ne n 0
    x : NNRat
    ⊢ Ne (↑x.den) 0
  -/
  all_goals simp [x.den_pos.ne', Nat.one_le_iff_ne_zero, *]
  /-
    🎉 no goals
  -/


@[simp] lemma addSubmonoid_closure_range_mul_self : closure (range fun x : ℚ≥0 ↦ x * x) = ⊤ := by
  /-
    ⊢ Eq (AddSubmonoid.closure (Set.range fun x => HMul.hMul x x)) Top.top
  -/
  simpa only [sq] using addSubmonoid_closure_range_pow two_ne_zero
  /-
    🎉 no goals
  -/


instance instStarOrderedRing : StarOrderedRing ℚ≥0 where
                   /-
                     a b : NNRat
                     ⊢ Iff (LE.le a b) (Exists fun p => And (Membership.mem (AddSubmonoid.closure ( …
                   -/
  le_iff a b := by simp [eq_comm, le_iff_exists_nonneg_add (a := a)]
                   /-
                     🎉 no goals
                   -/


@[simp] lemma addSubmonoid_closure_range_pow {n : ℕ} (hn₀ : n ≠ 0) (hn : Even n) :
    closure (range fun x : ℚ ↦ x ^ n) = nonneg _ := by
  /-
    n : Nat
    hn₀ : Ne n 0
    hn : Even n
    ⊢ Eq (AddSubmonoid.closure (Set.range fun x => HPow.hPow x n)) (AddSubmonoid.n …
  -/
  convert (AddMonoidHom.map_mclosure NNRat.coeHom <| range fun x ↦ x ^ n).symm
    /-
      case h.e'_2.h.e'_3
      n : Nat
      hn₀ : Ne n 0
      hn : Even n
      ⊢ Eq (Set.range fun x => HPow.hPow x n) (Set.image (⇑NNRat.coeHom) (Set.range  …
    -/
  · have (x : ℚ) : ∃ y : ℚ≥0, y ^ n = x ^ n := ⟨x.nnabs, by simp [hn.pow_abs]⟩
    /-
      case h.e'_2.h.e'_3
      n : Nat
      hn₀ : Ne n 0
      hn : Even n
      this : ∀ (x : Rat), Exists fun y => Eq (HPow.hPow (↑y) n) (HPow.hPow x n)
      ⊢ Eq (Set.range fun x => HPow.hPow x n) (Set.image (⇑NNRat.coeHom) (Set.range  …
    -/
    simp [subset_antisymm_iff, range_subset_iff, this]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      n : Nat
      hn₀ : Ne n 0
      hn : Even n
      ⊢ Eq (AddSubmonoid.nonneg Rat) (AddSubmonoid.map NNRat.coeHom (AddSubmonoid.cl …
    -/
  · ext
    /-
      case h.e'_3.h
      n : Nat
      hn₀ : Ne n 0
      hn : Even n
      x✝ : Rat
      ⊢ Iff (Membership.mem (AddSubmonoid.nonneg Rat) x✝) (Membership.mem (AddSubmon …
    -/
    simp [NNRat.addSubmonoid_closure_range_pow hn₀, NNRat.exists]
    /-
      🎉 no goals
    -/


@[simp]
lemma addSubmonoid_closure_range_mul_self : closure (range fun x : ℚ ↦ x * x) = nonneg _ := by
  /-
    ⊢ Eq (AddSubmonoid.closure (Set.range fun x => HMul.hMul x x)) (AddSubmonoid.n …
  -/
  simpa only [sq] using addSubmonoid_closure_range_pow two_ne_zero even_two
  /-
    🎉 no goals
  -/


instance instStarOrderedRing : StarOrderedRing ℚ where
                   /-
                     a b : Rat
                     ⊢ Iff (LE.le a b) (Exists fun p => And (Membership.mem (AddSubmonoid.closure ( …
                   -/
  le_iff a b := by simp [eq_comm, le_iff_exists_nonneg_add (a := a)]
                   /-
                     🎉 no goals
                   -/


