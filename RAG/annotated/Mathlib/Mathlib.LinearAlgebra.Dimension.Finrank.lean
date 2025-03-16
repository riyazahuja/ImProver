/-- The rank of a module as a natural number.

Defined by convention to be `0` if the space has infinite rank.

For a vector space `M` over a field `R`, this is the same as the finite dimension
of `M` over `R`.

Note that it is possible to have `M` with `¬(Module.Finite R M)` but `finrank R M ≠ 0`, for example
`ℤ × ℚ/ℤ` has `finrank` equal to `1`. -/
noncomputable def finrank (R M : Type*) [Semiring R] [AddCommGroup M] [Module R M] : ℕ :=
  Cardinal.toNat (Module.rank R M)


@[deprecated (since := "2024-10-01")] protected alias _root_.FiniteDimensional.finrank := finrank


theorem finrank_eq_of_rank_eq {n : ℕ} (h : Module.rank R M = ↑n) : finrank R M = n := by
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    h : Eq (Module.rank R M) ↑n
    ⊢ Eq (Module.finrank R M) n
  -/
  apply_fun toNat at h
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    h : Eq (Cardinal.toNat (Module.rank R M)) (Cardinal.toNat ↑n)
    ⊢ Eq (Module.finrank R M) n
  -/
  rw [toNat_natCast] at h
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    h : Eq (Cardinal.toNat (Module.rank R M)) n
    ⊢ Eq (Module.finrank R M) n
  -/
  exact mod_cast h
  /-
    🎉 no goals
  -/


lemma rank_eq_one_iff_finrank_eq_one : Module.rank R M = 1 ↔ finrank R M = 1 :=
  Cardinal.toNat_eq_one.symm


/-- This is like `rank_eq_one_iff_finrank_eq_one` but works for `2`, `3`, `4`, ... -/
lemma rank_eq_ofNat_iff_finrank_eq_ofNat (n : ℕ) [Nat.AtLeastTwo n] :
    Module.rank R M = OfNat.ofNat n ↔ finrank R M = OfNat.ofNat n :=
  Cardinal.toNat_eq_ofNat.symm


theorem finrank_le_of_rank_le {n : ℕ} (h : Module.rank R M ≤ ↑n) : finrank R M ≤ n := by
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    h : LE.le (Module.rank R M) ↑n
    ⊢ LE.le (Module.finrank R M) n
  -/
  rwa [← Cardinal.toNat_le_iff_le_of_lt_aleph0, toNat_natCast] at h
    /-
      case hc
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      n : Nat
      h : LE.le (Module.rank R M) ↑n
      ⊢ LT.lt (Module.rank R M) Cardinal.aleph0
    -/
  · exact h.trans_lt (nat_lt_aleph0 n)
    /-
      🎉 no goals
    -/
    /-
      case hd
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      n : Nat
      h : LE.le (Module.rank R M) ↑n
      ⊢ LT.lt (↑n) Cardinal.aleph0
    -/
  · exact nat_lt_aleph0 n
    /-
      🎉 no goals
    -/


theorem finrank_lt_of_rank_lt {n : ℕ} (h : Module.rank R M < ↑n) : finrank R M < n := by
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    h : LT.lt (Module.rank R M) ↑n
    ⊢ LT.lt (Module.finrank R M) n
  -/
  rwa [← Cardinal.toNat_lt_iff_lt_of_lt_aleph0, toNat_natCast] at h
    /-
      case hc
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      n : Nat
      h : LT.lt (Module.rank R M) ↑n
      ⊢ LT.lt (Module.rank R M) Cardinal.aleph0
    -/
  · exact h.trans (nat_lt_aleph0 n)
    /-
      🎉 no goals
    -/
    /-
      case hd
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      n : Nat
      h : LT.lt (Module.rank R M) ↑n
      ⊢ LT.lt (↑n) Cardinal.aleph0
    -/
  · exact nat_lt_aleph0 n
    /-
      🎉 no goals
    -/


theorem lt_rank_of_lt_finrank {n : ℕ} (h : n < finrank R M) : ↑n < Module.rank R M := by
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    h : LT.lt n (Module.finrank R M)
    ⊢ LT.lt (↑n) (Module.rank R M)
  -/
  rwa [← Cardinal.toNat_lt_iff_lt_of_lt_aleph0, toNat_natCast]
    /-
      case hc
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      n : Nat
      h : LT.lt n (Module.finrank R M)
      ⊢ LT.lt (↑n) Cardinal.aleph0
    -/
  · exact nat_lt_aleph0 n
    /-
      🎉 no goals
    -/
    /-
      case hd
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      n : Nat
      h : LT.lt n (Module.finrank R M)
      ⊢ LT.lt (Module.rank R M) Cardinal.aleph0
    -/
  · contrapose! h
    /-
      case hd
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      n : Nat
      h : LE.le Cardinal.aleph0 (Module.rank R M)
      ⊢ LE.le (Module.finrank R M) n
    -/
    rw [finrank, Cardinal.toNat_apply_of_aleph0_le h]
    /-
      case hd
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      n : Nat
      h : LE.le Cardinal.aleph0 (Module.rank R M)
      ⊢ LE.le 0 n
    -/
    exact n.zero_le
    /-
      🎉 no goals
    -/


theorem one_lt_rank_of_one_lt_finrank (h : 1 < finrank R M) : 1 < Module.rank R M := by
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    h : LT.lt 1 (Module.finrank R M)
    ⊢ LT.lt 1 (Module.rank R M)
  -/
  simpa using lt_rank_of_lt_finrank h
  /-
    🎉 no goals
  -/


theorem finrank_le_finrank_of_rank_le_rank
    (h : lift.{w} (Module.rank R M) ≤ Cardinal.lift.{v} (Module.rank R N))
    (h' : Module.rank R N < ℵ₀) : finrank R M ≤ finrank R N := by
  /-
    R : Type u
    M : Type v
    N : Type w
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    h : LE.le (Cardinal.lift.{w, v} (Module.rank R M)) (Cardinal.lift.{v, w} (Modu …
    h' : LT.lt (Module.rank R N) Cardinal.aleph0
    ⊢ LE.le (Module.finrank R M) (Module.finrank R N)
  -/
  simpa only [toNat_lift] using toNat_le_toNat h (lift_lt_aleph0.mpr h')
  /-
    🎉 no goals
  -/


/-- The dimension of a finite dimensional space is preserved under linear equivalence. -/
theorem finrank_eq (f : M ≃ₗ[R] M₂) : finrank R M = finrank R M₂ := by
  /-
    R : Type u_1
    M : Type u_2
    M₂ : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    f : LinearEquiv (RingHom.id R) M M₂
    ⊢ Eq (Module.finrank R M) (Module.finrank R M₂)
  -/
  unfold finrank
  /-
    R : Type u_1
    M : Type u_2
    M₂ : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    f : LinearEquiv (RingHom.id R) M M₂
    ⊢ Eq (Cardinal.toNat (Module.rank R M)) (Cardinal.toNat (Module.rank R M₂))
  -/
  rw [← Cardinal.toNat_lift, f.lift_rank_eq, Cardinal.toNat_lift]
  /-
    🎉 no goals
  -/


/-- Pushforwards of finite-dimensional submodules along a `LinearEquiv` have the same finrank. -/
theorem finrank_map_eq (f : M ≃ₗ[R] M₂) (p : Submodule R M) :
    finrank R (p.map (f : M →ₗ[R] M₂)) = finrank R p :=
  (f.submoduleMap p).finrank_eq.symm


/-- The dimensions of the domain and range of an injective linear map are equal. -/
theorem LinearMap.finrank_range_of_inj {f : M →ₗ[R] N} (hf : Function.Injective f) :
                                                      /-
                                                        R : Type u
                                                        M : Type v
                                                        N : Type w
                                                        inst✝⁴ : Ring R
                                                        inst✝³ : AddCommGroup M
                                                        inst✝² : Module R M
                                                        inst✝¹ : AddCommGroup N
                                                        inst✝ : Module R N
                                                        f : LinearMap (RingHom.id R) M N
                                                        hf : Function.Injective ⇑f
                                                        ⊢ Eq (Module.finrank R (Subtype fun x => Membership.mem (LinearMap.range f) x) …
                                                      -/
    finrank R (LinearMap.range f) = finrank R M := by rw [(LinearEquiv.ofInjective f hf).finrank_eq]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem Submodule.finrank_map_subtype_eq (p : Submodule R M) (q : Submodule R p) :
    finrank R (q.map p.subtype) = finrank R q :=
  (Submodule.equivSubtypeMap p q).symm.finrank_eq


@[simp]
theorem finrank_top : finrank R (⊤ : Submodule R M) = finrank R M := by
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Eq (Module.finrank R (Subtype fun x => Membership.mem Top.top x)) (Module.fi …
  -/
  unfold finrank
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Eq (Cardinal.toNat (Module.rank R (Subtype fun x => Membership.mem Top.top x …
  -/
  simp [rank_top]
  /-
    🎉 no goals
  -/

