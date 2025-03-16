/-- A (left) ideal in a semiring `R` is an additive submonoid `s` such that
`a * b ∈ s` whenever `b ∈ s`. If `R` is a ring, then `s` is an additive subgroup. -/
abbrev Ideal (R : Type u) [Semiring R] :=
  Submodule R R


protected theorem zero_mem : (0 : α) ∈ I :=
  Submodule.zero_mem I


protected theorem add_mem : a ∈ I → b ∈ I → a + b ∈ I :=
  Submodule.add_mem I


theorem mul_mem_left : b ∈ I → a * b ∈ I :=
  Submodule.smul_mem I a


@[ext]
theorem ext {I J : Ideal α} (h : ∀ x, x ∈ I ↔ x ∈ J) : I = J :=
  Submodule.ext h


@[simp]
theorem unit_mul_mem_iff_mem {x y : α} (hy : IsUnit y) : y * x ∈ I ↔ x ∈ I := by
  /-
    α : Type u
    inst✝ : Semiring α
    I : Ideal α
    x y : α
    hy : IsUnit y
    ⊢ Iff (Membership.mem I (HMul.hMul y x)) (Membership.mem I x)
  -/
  refine ⟨fun h => ?_, fun h => I.mul_mem_left y h⟩
  /-
    α : Type u
    inst✝ : Semiring α
    I : Ideal α
    x y : α
    hy : IsUnit y
    h : Membership.mem I (HMul.hMul y x)
    ⊢ Membership.mem I x
  -/
  obtain ⟨y', hy'⟩ := hy.exists_left_inv
  /-
    case intro
    α : Type u
    inst✝ : Semiring α
    I : Ideal α
    x y : α
    hy : IsUnit y
    h : Membership.mem I (HMul.hMul y x)
    y' : α
    hy' : Eq (HMul.hMul y' y) 1
    ⊢ Membership.mem I x
  -/
  have := I.mul_mem_left y' h
  /-
    case intro
    α : Type u
    inst✝ : Semiring α
    I : Ideal α
    x y : α
    hy : IsUnit y
    h : Membership.mem I (HMul.hMul y x)
    y' : α
    hy' : Eq (HMul.hMul y' y) 1
    this : Membership.mem I (HMul.hMul y' (HMul.hMul y x))
    ⊢ Membership.mem I x
  -/
  rwa [← mul_assoc, hy', one_mul] at this
  /-
    🎉 no goals
  -/


/-- For two elements `m` and `m'` in an `R`-module `M`, the set of elements `r : R` with
equal scalar product with `m` and `m'` is an ideal of `R`. If `M` is a group, this coincides
with the kernel of `LinearMap.toSpanSingleton R M (m - m')`. -/
def Module.eqIdeal (R) {M} [Semiring R] [AddCommMonoid M] [Module R M] (m m' : M) : Ideal R where
  carrier := {r : R | r • m = r • m'}
                      /-
                        α : Type u
                        β : Type v
                        F : Type w
                        R : Type ?u.2649
                        M : Type ?u.2652
                        inst✝² : Semiring R
                        inst✝¹ : AddCommMonoid M
                        inst✝ : Module R M
                        m m' : M
                        a✝ b✝ : R
                        h : Membership.mem (setOf fun r => Eq (HSMul.hSMul r m) (HSMul.hSMul r m')) a✝
                        h' : Membership.mem (setOf fun r => Eq (HSMul.hSMul r m) (HSMul.hSMul r m')) b✝
                        ⊢ Membership.mem (setOf fun r => Eq (HSMul.hSMul r m) (HSMul.hSMul r m')) (HAd …
                      -/
  add_mem' h h' := by simpa [add_smul] using congr($h + $h')
                      /-
                        🎉 no goals
                      -/
                  /-
                    α : Type u
                    β : Type v
                    F : Type w
                    R : Type ?u.2649
                    M : Type ?u.2652
                    inst✝² : Semiring R
                    inst✝¹ : AddCommMonoid M
                    inst✝ : Module R M
                    m m' : M
                    ⊢ Membership.mem { carrier := setOf fun r => Eq (HSMul.hSMul r m) (HSMul.hSMul …
                  -/
  zero_mem' := by simp_rw [Set.mem_setOf, zero_smul]
                  /-
                    🎉 no goals
                  -/
                        /-
                          α : Type u
                          β : Type v
                          F : Type w
                          R : Type ?u.2649
                          M : Type ?u.2652
                          inst✝² : Semiring R
                          inst✝¹ : AddCommMonoid M
                          inst✝ : Module R M
                          m m' : M
                          x✝¹ x✝ : R
                          h : Membership.mem { carrier := setOf fun r => Eq (HSMul.hSMul r m) (HSMul.hSM …
                          ⊢ Membership.mem { carrier := setOf fun r => Eq (HSMul.hSMul r m) (HSMul.hSMul …
                        -/
  smul_mem' _ _ h := by simpa [mul_smul] using congr(_ • $h)
                        /-
                          🎉 no goals
                        -/


@[simp]
theorem mul_unit_mem_iff_mem {x y : α} (hy : IsUnit y) : x * y ∈ I ↔ x ∈ I :=
  mul_comm y x ▸ unit_mul_mem_iff_mem I hy


theorem mul_mem_right (h : a ∈ I) : a * b ∈ I :=
  mul_comm b a ▸ I.mul_mem_left b h


lemma mem_of_dvd (hab : a ∣ b) (ha : a ∈ I) : b ∈ I := by
  /-
    α : Type u
    a b : α
    inst✝ : CommSemiring α
    I : Ideal α
    hab : Dvd.dvd a b
    ha : Membership.mem I a
    ⊢ Membership.mem I b
  -/
  obtain ⟨c, rfl⟩ := hab; exact I.mul_mem_right _ ha
                          /-
                            🎉 no goals
                          -/


theorem pow_mem_of_mem (ha : a ∈ I) (n : ℕ) (hn : 0 < n) : a ^ n ∈ I :=
                              /-
                                α : Type u
                                a : α
                                inst✝ : CommSemiring α
                                I : Ideal α
                                ha : Membership.mem I a
                                n : Nat
                                hn : LT.lt 0 n
                                ⊢ Not (LT.lt 0 Nat.zero)
                              -/
  Nat.casesOn n (Not.elim (by decide))
                              /-
                                🎉 no goals
                              -/
    (fun m _hm => (pow_succ a m).symm ▸ I.mul_mem_left (a ^ m) ha) hn


theorem pow_mem_of_pow_mem {m n : ℕ} (ha : a ^ m ∈ I) (h : m ≤ n) : a ^ n ∈ I := by
  /-
    α : Type u
    a : α
    inst✝ : CommSemiring α
    I : Ideal α
    m n : Nat
    ha : Membership.mem I (HPow.hPow a m)
    h : LE.le m n
    ⊢ Membership.mem I (HPow.hPow a n)
  -/
  rw [← Nat.add_sub_of_le h, pow_add]
  /-
    α : Type u
    a : α
    inst✝ : CommSemiring α
    I : Ideal α
    m n : Nat
    ha : Membership.mem I (HPow.hPow a m)
    h : LE.le m n
    ⊢ Membership.mem I (HMul.hMul (HPow.hPow a m) (HPow.hPow a (HSub.hSub n m)))
  -/
  exact I.mul_mem_right _ ha
  /-
    🎉 no goals
  -/


protected theorem neg_mem_iff : -a ∈ I ↔ a ∈ I :=
  Submodule.neg_mem_iff I


protected theorem add_mem_iff_left : b ∈ I → (a + b ∈ I ↔ a ∈ I) :=
  Submodule.add_mem_iff_left I


protected theorem add_mem_iff_right : a ∈ I → (a + b ∈ I ↔ b ∈ I) :=
  Submodule.add_mem_iff_right I


protected theorem sub_mem : a ∈ I → b ∈ I → a - b ∈ I :=
  Submodule.sub_mem I


theorem mul_sub_mul_mem {R : Type*} [CommRing R] (I : Ideal R) {a b c d : R} (h1 : a - b ∈ I)
    (h2 : c - d ∈ I) : a * c - b * d ∈ I := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    a b c d : R
    h1 : Membership.mem I (HSub.hSub a b)
    h2 : Membership.mem I (HSub.hSub c d)
    ⊢ Membership.mem I (HSub.hSub (HMul.hMul a c) (HMul.hMul b d))
  -/
  rw [show a * c - b * d = (a - b) * c + b * (c - d) by rw [sub_mul, mul_sub]; abel]
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    a b c d : R
    h1 : Membership.mem I (HSub.hSub a b)
    h2 : Membership.mem I (HSub.hSub c d)
    ⊢ Membership.mem I (HAdd.hAdd (HMul.hMul (HSub.hSub a b) c) (HMul.hMul b (HSub …
  -/
  exact I.add_mem (I.mul_mem_right _ h1) (I.mul_mem_left _ h2)
  /-
    🎉 no goals
  -/


