instance : One (SubMulAction R M) where
  one :=
    { carrier := Set.range fun r : R => r • (1 : M)
      smul_mem' := fun r _ ⟨r', hr'⟩ => hr' ▸ ⟨r * r', mul_smul _ _ _⟩ }


theorem coe_one : ↑(1 : SubMulAction R M) = Set.range fun r : R => r • (1 : M) :=
  rfl


@[simp]
theorem mem_one {x : M} : x ∈ (1 : SubMulAction R M) ↔ ∃ r : R, r • (1 : M) = x :=
  Iff.rfl


theorem subset_coe_one : (1 : Set M) ⊆ (1 : SubMulAction R M) := fun _ hx =>
  ⟨1, (one_smul _ _).trans hx.symm⟩


instance : Mul (SubMulAction R M) where
  mul p q :=
    { carrier := Set.image2 (· * ·) p q
      smul_mem' := fun r _ ⟨m₁, hm₁, m₂, hm₂, h⟩ =>
        h ▸ smul_mul_assoc r m₁ m₂ ▸ Set.mul_mem_mul (p.smul_mem _ hm₁) hm₂ }


@[norm_cast]
theorem coe_mul (p q : SubMulAction R M) : ↑(p * q) = (p * q : Set M) :=
  rfl


theorem mem_mul {p q : SubMulAction R M} {x : M} : x ∈ p * q ↔ ∃ y ∈ p, ∃ z ∈ q, y * z = x :=
  Set.mem_mul


instance mulOneClass : MulOneClass (SubMulAction R M) where
  mul := (· * ·)
  one := 1
  mul_one a := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝⁴ : Monoid R
      inst✝³ : MulAction R M
      inst✝² : MulOneClass M
      inst✝¹ : IsScalarTower R M M
      inst✝ : SMulCommClass R M M
      a : SubMulAction R M
      ⊢ Eq (HMul.hMul a 1) a
    -/
    ext x
    /-
      case h
      R : Type u_1
      M : Type u_2
      inst✝⁴ : Monoid R
      inst✝³ : MulAction R M
      inst✝² : MulOneClass M
      inst✝¹ : IsScalarTower R M M
      inst✝ : SMulCommClass R M M
      a : SubMulAction R M
      x : M
      ⊢ Iff (Membership.mem (HMul.hMul a 1) x) (Membership.mem a x)
    -/
    simp only [mem_mul, mem_one, mul_smul_comm, exists_exists_eq_and, mul_one]
    /-
      case h
      R : Type u_1
      M : Type u_2
      inst✝⁴ : Monoid R
      inst✝³ : MulAction R M
      inst✝² : MulOneClass M
      inst✝¹ : IsScalarTower R M M
      inst✝ : SMulCommClass R M M
      a : SubMulAction R M
      x : M
      ⊢ Iff (Exists fun y => And (Membership.mem a y) (Exists fun a => Eq (HSMul.hSM …
    -/
    constructor
      /-
        case h.mp
        R : Type u_1
        M : Type u_2
        inst✝⁴ : Monoid R
        inst✝³ : MulAction R M
        inst✝² : MulOneClass M
        inst✝¹ : IsScalarTower R M M
        inst✝ : SMulCommClass R M M
        a : SubMulAction R M
        x : M
        ⊢ (Exists fun y => And (Membership.mem a y) (Exists fun a => Eq (HSMul.hSMul a …
      -/
    · rintro ⟨y, hy, r, rfl⟩
    /-
      R : Type u_1
      M : Type u_2
      inst✝⁴ : Monoid R
      inst✝³ : MulAction R M
      inst✝² : MulOneClass M
      inst✝¹ : IsScalarTower R M M
      inst✝ : SMulCommClass R M M
      a : SubMulAction R M
      ⊢ Eq (HMul.hMul 1 a) a
    -/
      /-
        case h.mp.intro.intro.intro
        R : Type u_1
        M : Type u_2
        inst✝⁴ : Monoid R
        inst✝³ : MulAction R M
        inst✝² : MulOneClass M
        inst✝¹ : IsScalarTower R M M
        inst✝ : SMulCommClass R M M
        a : SubMulAction R M
        y : M
        hy : Membership.mem a y
        r : R
        ⊢ Membership.mem a (HSMul.hSMul r y)
      -/
    /-
      case h
      R : Type u_1
      M : Type u_2
      inst✝⁴ : Monoid R
      inst✝³ : MulAction R M
      inst✝² : MulOneClass M
      inst✝¹ : IsScalarTower R M M
      inst✝ : SMulCommClass R M M
      a : SubMulAction R M
      x : M
      ⊢ Iff (Membership.mem (HMul.hMul 1 a) x) (Membership.mem a x)
    -/
      exact smul_mem _ _ hy
    /-
      case h
      R : Type u_1
      M : Type u_2
      inst✝⁴ : Monoid R
      inst✝³ : MulAction R M
      inst✝² : MulOneClass M
      inst✝¹ : IsScalarTower R M M
      inst✝ : SMulCommClass R M M
      a : SubMulAction R M
      x : M
      ⊢ Iff (Exists fun a_1 => Exists fun z => And (Membership.mem a z) (Eq (HSMul.h …
    -/
      /-
        🎉 no goals
      -/
    /-
      case h
      R : Type u_1
      M : Type u_2
      inst✝⁴ : Monoid R
      inst✝³ : MulAction R M
      inst✝² : MulOneClass M
      inst✝¹ : IsScalarTower R M M
      inst✝ : SMulCommClass R M M
      a : SubMulAction R M
      x : M
      ⊢ (Exists fun a_1 => Exists fun z => And (Membership.mem a z) (Eq (HSMul.hSMul …
    -/
      /-
        case h.mpr
        R : Type u_1
        M : Type u_2
        inst✝⁴ : Monoid R
        inst✝³ : MulAction R M
        inst✝² : MulOneClass M
        inst✝¹ : IsScalarTower R M M
        inst✝ : SMulCommClass R M M
        a : SubMulAction R M
        x : M
        ⊢ Membership.mem a x → Exists fun y => And (Membership.mem a y) (Exists fun a  …
      -/
    /-
      case h.intro.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝⁴ : Monoid R
      inst✝³ : MulAction R M
      inst✝² : MulOneClass M
      inst✝¹ : IsScalarTower R M M
      inst✝ : SMulCommClass R M M
      a : SubMulAction R M
      r : R
      y : M
      hy : Membership.mem a y
      ⊢ Membership.mem a (HSMul.hSMul r y)
    -/
    · intro hx
    /-
      🎉 no goals
    -/
      /-
        case h.mpr
        R : Type u_1
        M : Type u_2
        inst✝⁴ : Monoid R
        inst✝³ : MulAction R M
        inst✝² : MulOneClass M
        inst✝¹ : IsScalarTower R M M
        inst✝ : SMulCommClass R M M
        a : SubMulAction R M
        x : M
        hx : Membership.mem a x
        ⊢ Exists fun y => And (Membership.mem a y) (Exists fun a => Eq (HSMul.hSMul a  …
      -/
      exact ⟨x, hx, 1, one_smul _ _⟩
      /-
        🎉 no goals
      -/
  one_mul a := by
    ext x
    simp only [mem_mul, mem_one, smul_mul_assoc, exists_exists_eq_and, one_mul]
    refine ⟨?_, fun hx => ⟨1, x, hx, one_smul _ _⟩⟩
    rintro ⟨r, y, hy, rfl⟩
    exact smul_mem _ _ hy


instance semiGroup : Semigroup (SubMulAction R M) where
  mul := (· * ·)
  mul_assoc _ _ _ := SetLike.coe_injective (mul_assoc (_ : Set _) _ _)


instance : Monoid (SubMulAction R M) :=
  { SubMulAction.semiGroup,
    SubMulAction.mulOneClass with }


theorem coe_pow (p : SubMulAction R M) : ∀ {n : ℕ} (_ : n ≠ 0), ↑(p ^ n) = (p : Set M) ^ n
  | 0, hn => (hn rfl).elim
               /-
                 R : Type u_1
                 M : Type u_2
                 inst✝⁴ : Monoid R
                 inst✝³ : MulAction R M
                 inst✝² : Monoid M
                 inst✝¹ : IsScalarTower R M M
                 inst✝ : SMulCommClass R M M
                 p : SubMulAction R M
                 x✝ : Ne 1 0
                 ⊢ Eq (↑(HPow.hPow p 1)) (HPow.hPow (↑p) 1)
               -/
  | 1, _ => by rw [pow_one, pow_one]
               /-
                 🎉 no goals
               -/
  | n + 2, _ => by
    /-
      R : Type u_1
      M : Type u_2
      inst✝⁴ : Monoid R
      inst✝³ : MulAction R M
      inst✝² : Monoid M
      inst✝¹ : IsScalarTower R M M
      inst✝ : SMulCommClass R M M
      p : SubMulAction R M
      n : Nat
      x✝ : Ne (HAdd.hAdd n 2) 0
      ⊢ Eq (↑(HPow.hPow p (HAdd.hAdd n 2))) (HPow.hPow (↑p) (HAdd.hAdd n 2))
    -/
    rw [pow_succ _ (n + 1), pow_succ _ (n + 1), coe_mul, coe_pow _ n.succ_ne_zero]
    /-
      🎉 no goals
    -/


theorem subset_coe_pow (p : SubMulAction R M) : ∀ {n : ℕ}, (p : Set M) ^ n ⊆ ↑(p ^ n)
  | 0 => by
    /-
      R : Type u_1
      M : Type u_2
      inst✝⁴ : Monoid R
      inst✝³ : MulAction R M
      inst✝² : Monoid M
      inst✝¹ : IsScalarTower R M M
      inst✝ : SMulCommClass R M M
      p : SubMulAction R M
      ⊢ HasSubset.Subset (HPow.hPow (↑p) 0) ↑(HPow.hPow p 0)
    -/
    rw [pow_zero, pow_zero]
    /-
      R : Type u_1
      M : Type u_2
      inst✝⁴ : Monoid R
      inst✝³ : MulAction R M
      inst✝² : Monoid M
      inst✝¹ : IsScalarTower R M M
      inst✝ : SMulCommClass R M M
      p : SubMulAction R M
      ⊢ HasSubset.Subset 1 ↑1
    -/
    exact subset_coe_one
    /-
      🎉 no goals
    -/
                /-
                  R : Type u_1
                  M : Type u_2
                  inst✝⁴ : Monoid R
                  inst✝³ : MulAction R M
                  inst✝² : Monoid M
                  inst✝¹ : IsScalarTower R M M
                  inst✝ : SMulCommClass R M M
                  p : SubMulAction R M
                  n : Nat
                  ⊢ HasSubset.Subset (HPow.hPow (↑p) (HAdd.hAdd n 1)) ↑(HPow.hPow p (HAdd.hAdd n …
                -/
  | n + 1 => by rw [← Nat.succ_eq_add_one, coe_pow _ n.succ_ne_zero]
                /-
                  🎉 no goals
                -/


