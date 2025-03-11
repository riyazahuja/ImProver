variable {L} in
/-- An `sl₂` triple within a Lie ring `L` is a triple of elements `h`, `e`, `f` obeying relations
which ensure that the Lie subalgebra they generate is equivalent to `sl₂`. -/
structure IsSl2Triple (h e f : L) : Prop where
  h_ne_zero : h ≠ 0
  lie_e_f : ⁅e, f⁆ = h
  lie_h_e_nsmul : ⁅h, e⁆ = 2 • e
  lie_h_f_nsmul : ⁅h, f⁆ = - (2 • f)


lemma symm (ht : IsSl2Triple h e f) : IsSl2Triple (-h) f e where
                  /-
                    L : Type u_2
                    inst✝ : LieRing L
                    h e f : L
                    ht : IsSl2Triple h e f
                    ⊢ Ne (Neg.neg h) 0
                  -/
  h_ne_zero := by simpa using ht.h_ne_zero
                  /-
                    🎉 no goals
                  -/
                /-
                  L : Type u_2
                  inst✝ : LieRing L
                  h e f : L
                  ht : IsSl2Triple h e f
                  ⊢ Eq (Bracket.bracket f e) (Neg.neg h)
                -/
  lie_e_f := by rw [← neg_eq_iff_eq_neg, lie_skew, ht.lie_e_f]
                /-
                  🎉 no goals
                -/
                      /-
                        L : Type u_2
                        inst✝ : LieRing L
                        h e f : L
                        ht : IsSl2Triple h e f
                        ⊢ Eq (Bracket.bracket (Neg.neg h) f) (HSMul.hSMul 2 f)
                      -/
  lie_h_e_nsmul := by rw [neg_lie, neg_eq_iff_eq_neg, ht.lie_h_f_nsmul]
                      /-
                        🎉 no goals
                      -/
                      /-
                        L : Type u_2
                        inst✝ : LieRing L
                        h e f : L
                        ht : IsSl2Triple h e f
                        ⊢ Eq (Bracket.bracket (Neg.neg h) e) (Neg.neg (HSMul.hSMul 2 e))
                      -/
  lie_h_f_nsmul := by rw [neg_lie, neg_inj, ht.lie_h_e_nsmul]
                      /-
                        🎉 no goals
                      -/


@[simp] lemma symm_iff : IsSl2Triple (-h) f e ↔ IsSl2Triple h e f :=
  ⟨fun t ↦ neg_neg h ▸ t.symm, symm⟩


lemma lie_h_e_smul (t : IsSl2Triple h e f) : ⁅h, e⁆ = (2 : R) • e := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    h e f : L
    t : IsSl2Triple h e f
    ⊢ Eq (Bracket.bracket h e) (HSMul.hSMul 2 e)
  -/
  simp [t.lie_h_e_nsmul, two_smul]
  /-
    🎉 no goals
  -/


lemma lie_lie_smul_f (t : IsSl2Triple h e f) : ⁅h, f⁆ = -((2 : R) • f) := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    h e f : L
    t : IsSl2Triple h e f
    ⊢ Eq (Bracket.bracket h f) (Neg.neg (HSMul.hSMul 2 f))
  -/
  simp [t.lie_h_f_nsmul, two_smul]
  /-
    🎉 no goals
  -/


lemma e_ne_zero (t : IsSl2Triple h e f) : e ≠ 0 := by
  /-
    L : Type u_2
    inst✝ : LieRing L
    h e f : L
    t : IsSl2Triple h e f
    ⊢ Ne e 0
  -/
  have := t.h_ne_zero
  /-
    L : Type u_2
    inst✝ : LieRing L
    h e f : L
    t : IsSl2Triple h e f
    this : Ne h 0
    ⊢ Ne e 0
  -/
  contrapose! this
  /-
    L : Type u_2
    inst✝ : LieRing L
    h e f : L
    t : IsSl2Triple h e f
    this : Eq e 0
    ⊢ Eq h 0
  -/
  simpa [this] using t.lie_e_f.symm
  /-
    🎉 no goals
  -/


lemma f_ne_zero (t : IsSl2Triple h e f) : f ≠ 0 := by
  /-
    L : Type u_2
    inst✝ : LieRing L
    h e f : L
    t : IsSl2Triple h e f
    ⊢ Ne f 0
  -/
  have := t.h_ne_zero
  /-
    L : Type u_2
    inst✝ : LieRing L
    h e f : L
    t : IsSl2Triple h e f
    this : Ne h 0
    ⊢ Ne f 0
  -/
  contrapose! this
  /-
    L : Type u_2
    inst✝ : LieRing L
    h e f : L
    t : IsSl2Triple h e f
    this : Eq f 0
    ⊢ Eq h 0
  -/
  simpa [this] using t.lie_e_f.symm
  /-
    🎉 no goals
  -/


/-- Given a representation of a Lie algebra with distinguished `sl₂` triple, a vector is said to be
primitive if it is a simultaneous eigenvector for the action of both `h`, `e`, and the eigenvalue
for `e` is zero. -/
structure HasPrimitiveVectorWith (t : IsSl2Triple h e f) (m : M) (μ : R) : Prop where
  ne_zero : m ≠ 0
  lie_h : ⁅h, m⁆ = μ • m
  lie_e : ⁅e, m⁆ = 0


/-- Given a representation of a Lie algebra with distinguished `sl₂` triple, a simultaneous
eigenvector for the action of both `h` and `e` necessarily has eigenvalue zero for `e`. -/
lemma HasPrimitiveVectorWith.mk' [NoZeroSMulDivisors ℤ M] (t : IsSl2Triple h e f) (m : M) (μ ρ : R)
    (hm : m ≠ 0) (hm' : ⁅h, m⁆ = μ • m) (he : ⁅e, m⁆ = ρ • m) :
    HasPrimitiveVectorWith t m μ  where
  ne_zero := hm
  lie_h := hm'
  lie_e := by
    /-
      R : Type u_1
      L : Type u_2
      M : Type u_3
      inst✝⁷ : CommRing R
      inst✝⁶ : LieRing L
      inst✝⁵ : LieAlgebra R L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      inst✝¹ : LieModule R L M
      h e f : L
      inst✝ : NoZeroSMulDivisors Int M
      t : IsSl2Triple h e f
      m : M
      μ ρ : R
      hm : Ne m 0
      hm' : Eq (Bracket.bracket h m) (HSMul.hSMul μ m)
      he : Eq (Bracket.bracket e m) (HSMul.hSMul ρ m)
      ⊢ Eq (Bracket.bracket e m) 0
    -/
    suffices 2 • ⁅e, m⁆ = 0 by simpa using this
    rw [← nsmul_lie, ← t.lie_h_e_nsmul, lie_lie, hm', lie_smul, he, lie_smul, hm',
      smul_smul, smul_smul, mul_comm ρ μ, sub_self]


local notation "ψ" n => ((toEnd R L M f) ^ n) m

-- Although this is true by definition, we include this lemma (and the assumption) to mirror the API
-- for `lie_h_pow_toEnd_f` and `lie_e_pow_succ_toEnd_f`.

set_option linter.unusedVariables false in
@[nolint unusedArguments]
lemma lie_f_pow_toEnd_f (P : HasPrimitiveVectorWith t m μ) (n : ℕ) :
    ⁅f, ψ n⁆ = ψ (n + 1) := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    h e f : L
    m : M
    μ : R
    t : IsSl2Triple h e f
    P : t.HasPrimitiveVectorWith m μ
    n : Nat
    ⊢ Eq (Bracket.bracket f ((HPow.hPow ((LieModule.toEnd R L M) f) n) m)) ((HPow. …
  -/
  simp [pow_succ']
  /-
    🎉 no goals
  -/


lemma lie_h_pow_toEnd_f (n : ℕ) :
    ⁅h, ψ n⁆ = (μ - 2 * n) • ψ n := by
  induction n with
  | zero => simpa using P.lie_h
  | succ n ih =>
    rw [pow_succ', LinearMap.mul_apply, toEnd_apply_apply, Nat.cast_add, Nat.cast_one,
      leibniz_lie h, t.lie_lie_smul_f R, ← neg_smul, ih, lie_smul, smul_lie, ← add_smul]
    congr
    ring


lemma lie_e_pow_succ_toEnd_f (n : ℕ) :
    ⁅e, ψ (n + 1)⁆ = ((n + 1) * (μ - n)) • ψ n := by
  induction n with
  | zero =>
      simp only [zero_add, pow_one, toEnd_apply_apply, Nat.cast_zero, sub_zero, one_mul,
        pow_zero, LinearMap.one_apply, leibniz_lie e, t.lie_e_f, P.lie_e, P.lie_h, lie_zero,
        add_zero]
  | succ n ih =>
    rw [pow_succ', LinearMap.mul_apply, toEnd_apply_apply, leibniz_lie e, t.lie_e_f,
      lie_h_pow_toEnd_f P, ih, lie_smul, lie_f_pow_toEnd_f P, ← add_smul,
      Nat.cast_add, Nat.cast_one]
    congr
    ring


/-- The eigenvalue of a primitive vector must be a natural number if the representation is
finite-dimensional. -/
lemma exists_nat [IsNoetherian R M] [NoZeroSMulDivisors R M] [IsDomain R] [CharZero R] :
    ∃ n : ℕ, μ = n := by
  suffices ∃ n : ℕ, (ψ n) = 0 by
    obtain ⟨n, hn₁, hn₂⟩ := Nat.exists_not_and_succ_of_not_zero_of_exists P.ne_zero this
    refine ⟨n, ?_⟩
    have := lie_e_pow_succ_toEnd_f P n
    rw [hn₂, lie_zero, eq_comm, smul_eq_zero_iff_left hn₁, mul_eq_zero, sub_eq_zero] at this
    exact this.resolve_left <| Nat.cast_add_one_ne_zero n
  have hs : (range <| fun (n : ℕ) ↦ μ - 2 * n).Infinite := by
    rw [infinite_range_iff (fun n m ↦ by simp)]; infer_instance
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    h e f : L
    m : M
    μ : R
    t : IsSl2Triple h e f
    P : t.HasPrimitiveVectorWith m μ
    inst✝³ : IsNoetherian R M
    inst✝² : NoZeroSMulDivisors R M
    inst✝¹ : IsDomain R
    inst✝ : CharZero R
    hs : (Set.range fun n => HSub.hSub μ (HMul.hMul 2 ↑n)).Infinite
    ⊢ Exists fun n => Eq ((HPow.hPow ((LieModule.toEnd R L M) f) n) m) 0
  -/
  by_contra! contra
  exact hs ((toEnd R L M h).eigenvectors_linearIndependent
    {μ - 2 * n | n : ℕ}
    (fun ⟨s, hs⟩ ↦ ψ Classical.choose hs)
    (fun ⟨r, hr⟩ ↦ by simp [lie_h_pow_toEnd_f P, Classical.choose_spec hr, contra,
      Module.End.hasEigenvector_iff, Module.End.mem_eigenspace_iff])).finite


lemma pow_toEnd_f_ne_zero_of_eq_nat
    [CharZero R] [NoZeroSMulDivisors R M]
    {n : ℕ} (hn : μ = n) {i} (hi : i ≤ n) : (ψ i) ≠ 0 := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    h e f : L
    m : M
    μ : R
    t : IsSl2Triple h e f
    P : t.HasPrimitiveVectorWith m μ
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    n : Nat
    hn : Eq μ ↑n
    i : Nat
    hi : LE.le i n
    ⊢ Ne ((HPow.hPow ((LieModule.toEnd R L M) f) i) m) 0
  -/
  intro H
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    h e f : L
    m : M
    μ : R
    t : IsSl2Triple h e f
    P : t.HasPrimitiveVectorWith m μ
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    n : Nat
    hn : Eq μ ↑n
    i : Nat
    hi : LE.le i n
    H : Eq ((HPow.hPow ((LieModule.toEnd R L M) f) i) m) 0
    ⊢ False
  -/
  induction i
    /-
      case zero
      R : Type u_1
      L : Type u_2
      M : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      h e f : L
      m : M
      μ : R
      t : IsSl2Triple h e f
      P : t.HasPrimitiveVectorWith m μ
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors R M
      n : Nat
      hn : Eq μ ↑n
      hi : LE.le 0 n
      H : Eq ((HPow.hPow ((LieModule.toEnd R L M) f) 0) m) 0
      ⊢ False
    -/
  · exact P.ne_zero (by simpa using H)
    /-
      🎉 no goals
    -/
  · next i IH =>
    have : ((i + 1) * (n - i) : ℤ) • (toEnd R L M f ^ i) m = 0 := by
      have := congr_arg (⁅e, ·⁆) H
      simpa [← Int.cast_smul_eq_zsmul R, P.lie_e_pow_succ_toEnd_f, hn] using this
    rw [← Int.cast_smul_eq_zsmul R, smul_eq_zero, Int.cast_eq_zero, mul_eq_zero, sub_eq_zero,
      Nat.cast_inj, ← @Nat.cast_one ℤ, ← Nat.cast_add, Nat.cast_eq_zero] at this
    simp only [add_eq_zero, one_ne_zero, and_false, false_or] at this
    exact (hi.trans_eq (this.resolve_right (IH (i.le_succ.trans hi)))).not_lt i.lt_succ_self


lemma pow_toEnd_f_eq_zero_of_eq_nat
    [IsNoetherian R M] [NoZeroSMulDivisors R M] [IsDomain R] [CharZero R]
    {n : ℕ} (hn : μ = n) : (ψ (n + 1)) = 0 := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    h e f : L
    m : M
    μ : R
    t : IsSl2Triple h e f
    P : t.HasPrimitiveVectorWith m μ
    inst✝³ : IsNoetherian R M
    inst✝² : NoZeroSMulDivisors R M
    inst✝¹ : IsDomain R
    inst✝ : CharZero R
    n : Nat
    hn : Eq μ ↑n
    ⊢ Eq ((HPow.hPow ((LieModule.toEnd R L M) f) (HAdd.hAdd n 1)) m) 0
  -/
  by_contra h
  have : t.HasPrimitiveVectorWith (ψ (n + 1)) (n - 2 * (n + 1) : R) :=
    { ne_zero := h
      lie_h := (P.lie_h_pow_toEnd_f _).trans (by simp [hn])
      lie_e := (P.lie_e_pow_succ_toEnd_f _).trans (by simp [hn]) }
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    h✝ e f : L
    m : M
    μ : R
    t : IsSl2Triple h✝ e f
    P : t.HasPrimitiveVectorWith m μ
    inst✝³ : IsNoetherian R M
    inst✝² : NoZeroSMulDivisors R M
    inst✝¹ : IsDomain R
    inst✝ : CharZero R
    n : Nat
    hn : Eq μ ↑n
    h : Not (Eq ((HPow.hPow ((LieModule.toEnd R L M) f) (HAdd.hAdd n 1)) m) 0)
    this : t.HasPrimitiveVectorWith ((HPow.hPow ((LieModule.toEnd R L M) f) (HAdd. …
    ⊢ False
  -/
  obtain ⟨m, hm⟩ := this.exists_nat
  /-
    case intro
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    h✝ e f : L
    m✝ : M
    μ : R
    t : IsSl2Triple h✝ e f
    P : t.HasPrimitiveVectorWith m✝ μ
    inst✝³ : IsNoetherian R M
    inst✝² : NoZeroSMulDivisors R M
    inst✝¹ : IsDomain R
    inst✝ : CharZero R
    n : Nat
    hn : Eq μ ↑n
    h : Not (Eq ((HPow.hPow ((LieModule.toEnd R L M) f) (HAdd.hAdd n 1)) m✝) 0)
    this : t.HasPrimitiveVectorWith ((HPow.hPow ((LieModule.toEnd R L M) f) (HAdd. …
    m : Nat
    hm : Eq (HSub.hSub (↑n) (HMul.hMul 2 (HAdd.hAdd (↑n) 1))) ↑m
    ⊢ False
  -/
  have : (n : ℤ) < m + 2 * (n + 1) := by omega
  /-
    case intro
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    h✝ e f : L
    m✝ : M
    μ : R
    t : IsSl2Triple h✝ e f
    P : t.HasPrimitiveVectorWith m✝ μ
    inst✝³ : IsNoetherian R M
    inst✝² : NoZeroSMulDivisors R M
    inst✝¹ : IsDomain R
    inst✝ : CharZero R
    n : Nat
    hn : Eq μ ↑n
    h : Not (Eq ((HPow.hPow ((LieModule.toEnd R L M) f) (HAdd.hAdd n 1)) m✝) 0)
    this✝ : t.HasPrimitiveVectorWith ((HPow.hPow ((LieModule.toEnd R L M) f) (HAdd …
    m : Nat
    hm : Eq (HSub.hSub (↑n) (HMul.hMul 2 (HAdd.hAdd (↑n) 1))) ↑m
    this : LT.lt (↑n) (HAdd.hAdd (↑m) (HMul.hMul 2 (HAdd.hAdd (↑n) 1)))
    ⊢ False
  -/
  exact this.ne (Int.cast_injective (α := R) <| by simpa [sub_eq_iff_eq_add] using hm)
  /-
    🎉 no goals
  -/


