/-- A module is a generalization of vector spaces to a scalar semiring.
  It consists of a scalar semiring `R` and an additive monoid of "vectors" `M`,
  connected by a "scalar multiplication" operation `r • x : M`
  (where `r : R` and `x : M`) with some natural associativity and
  distributivity axioms similar to those on a ring. -/
@[ext]
class Module (R : Type u) (M : Type v) [Semiring R] [AddCommMonoid M] extends
  DistribMulAction R M where
  /-- Scalar multiplication distributes over addition from the right. -/
  protected add_smul : ∀ (r s : R) (x : M), (r + s) • x = r • x + s • x
  /-- Scalar multiplication by zero gives zero. -/
  protected zero_smul : ∀ x : M, (0 : R) • x = 0


/-- A module over a semiring automatically inherits a `MulActionWithZero` structure. -/
instance (priority := 100) Module.toMulActionWithZero
  {R M} {_ : Semiring R} {_ : AddCommMonoid M} [Module R M] : MulActionWithZero R M :=
  { (inferInstance : MulAction R M) with
    smul_zero := smul_zero
    zero_smul := Module.zero_smul }


theorem add_smul : (r + s) • x = r • x + s • x :=
  Module.add_smul r s x


theorem Convex.combo_self {a b : R} (h : a + b = 1) (x : M) : a • x + b • x = x := by
  /-
    R : Type u_1
    M : Type u_3
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    a b : R
    h : Eq (HAdd.hAdd a b) 1
    x : M
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b x)) x
  -/
  rw [← add_smul, h, one_smul]
  /-
    🎉 no goals
  -/


                                             /-
                                               R : Type u_1
                                               M : Type u_3
                                               inst✝² : Semiring R
                                               inst✝¹ : AddCommMonoid M
                                               inst✝ : Module R M
                                               x : M
                                               ⊢ Eq (HSMul.hSMul 2 x) (HAdd.hAdd x x)
                                             -/
theorem two_smul : (2 : R) • x = x + x := by rw [← one_add_one_eq_two, add_smul, one_smul]
                                             /-
                                               🎉 no goals
                                             -/


/-- Pullback a `Module` structure along an injective additive monoid homomorphism.
See note [reducible non-instances]. -/
protected abbrev Function.Injective.module [AddCommMonoid M₂] [SMul R M₂] (f : M₂ →+ M)
    (hf : Injective f) (smul : ∀ (c : R) (x), f (c • x) = c • f x) : Module R M₂ :=
  { hf.distribMulAction f smul with
                                        /-
                                          R : Type u_1
                                          S : Type u_2
                                          M : Type u_3
                                          M₂ : Type u_4
                                          inst✝⁴ : Semiring R
                                          inst✝³ : AddCommMonoid M
                                          inst✝² : Module R M
                                          r s : R
                                          x✝ : M
                                          inst✝¹ : AddCommMonoid M₂
                                          inst✝ : SMul R M₂
                                          f : AddMonoidHom M₂ M
                                          hf : Function.Injective ⇑f
                                          smul : ∀ (c : R) (x : M₂), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
                                          c₁ c₂ : R
                                          x : M₂
                                          ⊢ Eq (f (HSMul.hSMul (HAdd.hAdd c₁ c₂) x)) (f (HAdd.hAdd (HSMul.hSMul c₁ x) (H …
                                        -/
    add_smul := fun c₁ c₂ x => hf <| by simp only [smul, f.map_add, add_smul]
                                        /-
                                          🎉 no goals
                                        -/
                                   /-
                                     R : Type u_1
                                     S : Type u_2
                                     M : Type u_3
                                     M₂ : Type u_4
                                     inst✝⁴ : Semiring R
                                     inst✝³ : AddCommMonoid M
                                     inst✝² : Module R M
                                     r s : R
                                     x✝ : M
                                     inst✝¹ : AddCommMonoid M₂
                                     inst✝ : SMul R M₂
                                     f : AddMonoidHom M₂ M
                                     hf : Function.Injective ⇑f
                                     smul : ∀ (c : R) (x : M₂), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
                                     x : M₂
                                     ⊢ Eq (f (HSMul.hSMul 0 x)) (f 0)
                                   -/
    zero_smul := fun x => hf <| by simp only [smul, zero_smul, f.map_zero] }
                                   /-
                                     🎉 no goals
                                   -/


/-- Pushforward a `Module` structure along a surjective additive monoid homomorphism.
See note [reducible non-instances]. -/
protected abbrev Function.Surjective.module [AddCommMonoid M₂] [SMul R M₂] (f : M →+ M₂)
    (hf : Surjective f) (smul : ∀ (c : R) (x), f (c • x) = c • f x) : Module R M₂ :=
  { toDistribMulAction := hf.distribMulAction f smul
    add_smul := fun c₁ c₂ x => by
      /-
        R : Type u_1
        S : Type u_2
        M : Type u_3
        M₂ : Type u_4
        inst✝⁴ : Semiring R
        inst✝³ : AddCommMonoid M
        inst✝² : Module R M
        r s : R
        x✝ : M
        inst✝¹ : AddCommMonoid M₂
        inst✝ : SMul R M₂
        f : AddMonoidHom M M₂
        hf : Function.Surjective ⇑f
        smul : ∀ (c : R) (x : M), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
        c₁ c₂ : R
        x : M₂
        ⊢ Eq (HSMul.hSMul (HAdd.hAdd c₁ c₂) x) (HAdd.hAdd (HSMul.hSMul c₁ x) (HSMul.hS …
      -/
      rcases hf x with ⟨x, rfl⟩
      /-
        case intro
        R : Type u_1
        S : Type u_2
        M : Type u_3
        M₂ : Type u_4
        inst✝⁴ : Semiring R
        inst✝³ : AddCommMonoid M
        inst✝² : Module R M
        r s : R
        x✝ : M
        inst✝¹ : AddCommMonoid M₂
        inst✝ : SMul R M₂
        f : AddMonoidHom M M₂
        hf : Function.Surjective ⇑f
        smul : ∀ (c : R) (x : M), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
        c₁ c₂ : R
        x : M
        ⊢ Eq (HSMul.hSMul (HAdd.hAdd c₁ c₂) (f x)) (HAdd.hAdd (HSMul.hSMul c₁ (f x)) ( …
      -/
      simp only [add_smul, ← smul, ← f.map_add]
      /-
        🎉 no goals
      -/
    zero_smul := fun x => by
      /-
        R : Type u_1
        S : Type u_2
        M : Type u_3
        M₂ : Type u_4
        inst✝⁴ : Semiring R
        inst✝³ : AddCommMonoid M
        inst✝² : Module R M
        r s : R
        x✝ : M
        inst✝¹ : AddCommMonoid M₂
        inst✝ : SMul R M₂
        f : AddMonoidHom M M₂
        hf : Function.Surjective ⇑f
        smul : ∀ (c : R) (x : M), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
        x : M₂
        ⊢ Eq (HSMul.hSMul 0 x) 0
      -/
      rcases hf x with ⟨x, rfl⟩
      /-
        case intro
        R : Type u_1
        S : Type u_2
        M : Type u_3
        M₂ : Type u_4
        inst✝⁴ : Semiring R
        inst✝³ : AddCommMonoid M
        inst✝² : Module R M
        r s : R
        x✝ : M
        inst✝¹ : AddCommMonoid M₂
        inst✝ : SMul R M₂
        f : AddMonoidHom M M₂
        hf : Function.Surjective ⇑f
        smul : ∀ (c : R) (x : M), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
        x : M
        ⊢ Eq (HSMul.hSMul 0 (f x)) 0
      -/
      rw [← f.map_zero, ← smul, zero_smul] }
      /-
        🎉 no goals
      -/


theorem Module.eq_zero_of_zero_eq_one (zero_eq_one : (0 : R) = 1) : x = 0 := by
  /-
    R : Type u_1
    M : Type u_3
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x : M
    zero_eq_one : Eq 0 1
    ⊢ Eq x 0
  -/
  rw [← one_smul R x, ← zero_eq_one, zero_smul]
  /-
    🎉 no goals
  -/


@[simp]
theorem smul_add_one_sub_smul {R : Type*} [Ring R] [Module R M] {r : R} {m : M} :
                                  /-
                                    M : Type u_3
                                    inst✝² : AddCommMonoid M
                                    R : Type u_5
                                    inst✝¹ : Ring R
                                    inst✝ : Module R M
                                    r : R
                                    m : M
                                    ⊢ Eq (HAdd.hAdd (HSMul.hSMul r m) (HSMul.hSMul (HSub.hSub 1 r) m)) m
                                  -/
    r • m + (1 - r) • m = m := by rw [← add_smul, add_sub_cancel, one_smul]
                                  /-
                                    🎉 no goals
                                  -/


theorem Convex.combo_eq_smul_sub_add [Module R M] {x y : M} {a b : R} (h : a + b = 1) :
    a • x + b • y = b • (y - x) + x :=
  calc
                                                          /-
                                                            R : Type u_1
                                                            M : Type u_3
                                                            inst✝² : Semiring R
                                                            inst✝¹ : AddCommGroup M
                                                            inst✝ : Module R M
                                                            x y : M
                                                            a b : R
                                                            h : Eq (HAdd.hAdd a b) 1
                                                            ⊢ Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y)) (HAdd.hAdd (HSub.hSub (HS …
                                                          -/
    a • x + b • y = b • y - b • x + (a • x + b • x) := by rw [sub_add_add_cancel, add_comm]
                                                          /-
                                                            🎉 no goals
                                                          -/
                              /-
                                R : Type u_1
                                M : Type u_3
                                inst✝² : Semiring R
                                inst✝¹ : AddCommGroup M
                                inst✝ : Module R M
                                x y : M
                                a b : R
                                h : Eq (HAdd.hAdd a b) 1
                                ⊢ Eq (HAdd.hAdd (HSub.hSub (HSMul.hSMul b y) (HSMul.hSMul b x)) (HAdd.hAdd (HS …
                              -/
    _ = b • (y - x) + x := by rw [smul_sub, Convex.combo_self h]
                              /-
                                🎉 no goals
                              -/


/-- A variant of `Module.ext` that's convenient for term-mode. -/
theorem Module.ext' {R : Type*} [Semiring R] {M : Type*} [AddCommMonoid M] (P Q : Module R M)
    (w : ∀ (r : R) (m : M), (haveI := P; r • m) = (haveI := Q; r • m)) :
    P = Q := by
  /-
    R : Type u_5
    inst✝¹ : Semiring R
    M : Type u_6
    inst✝ : AddCommMonoid M
    P Q : Module R M
    w : ∀ (r : R) (m : M), Eq (HSMul.hSMul r m) (HSMul.hSMul r m)
    ⊢ Eq P Q
  -/
  ext
  /-
    case smul.h.h
    R : Type u_5
    inst✝¹ : Semiring R
    M : Type u_6
    inst✝ : AddCommMonoid M
    P Q : Module R M
    w : ∀ (r : R) (m : M), Eq (HSMul.hSMul r m) (HSMul.hSMul r m)
    x✝¹ : R
    x✝ : M
    ⊢ Eq (SMul.smul x✝¹ x✝) (SMul.smul x✝¹ x✝)
  -/
  exact w _ _
  /-
    🎉 no goals
  -/


@[simp]
theorem neg_smul : -r • x = -(r • x) :=
                                   /-
                                     R : Type u_1
                                     M : Type u_3
                                     inst✝² : Ring R
                                     inst✝¹ : AddCommGroup M
                                     inst✝ : Module R M
                                     r : R
                                     x : M
                                     ⊢ Eq (HAdd.hAdd (HSMul.hSMul (Neg.neg r) x) (HSMul.hSMul r x)) 0
                                   -/
  eq_neg_of_add_eq_zero_left <| by rw [← add_smul, neg_add_cancel, zero_smul]
                                   /-
                                     🎉 no goals
                                   -/


                                             /-
                                               R : Type u_1
                                               M : Type u_3
                                               inst✝² : Ring R
                                               inst✝¹ : AddCommGroup M
                                               inst✝ : Module R M
                                               r : R
                                               x : M
                                               ⊢ Eq (HSMul.hSMul (Neg.neg r) (Neg.neg x)) (HSMul.hSMul r x)
                                             -/
theorem neg_smul_neg : -r • -x = r • x := by rw [neg_smul, smul_neg, neg_neg]
                                             /-
                                               🎉 no goals
                                             -/


                                                       /-
                                                         R : Type u_1
                                                         M : Type u_3
                                                         inst✝² : Ring R
                                                         inst✝¹ : AddCommGroup M
                                                         inst✝ : Module R M
                                                         x : M
                                                         ⊢ Eq (HSMul.hSMul (-1) x) (Neg.neg x)
                                                       -/
theorem neg_one_smul (x : M) : (-1 : R) • x = -x := by simp
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem sub_smul (r s : R) (y : M) : (r - s) • y = r • y - s • y := by
  /-
    R : Type u_1
    M : Type u_3
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    r s : R
    y : M
    ⊢ Eq (HSMul.hSMul (HSub.hSub r s) y) (HSub.hSub (HSMul.hSMul r y) (HSMul.hSMul …
  -/
  simp [add_smul, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


/-- A module over a `Subsingleton` semiring is a `Subsingleton`. We cannot register this
as an instance because Lean has no way to guess `R`. -/
protected theorem Module.subsingleton (R M : Type*) [Semiring R] [Subsingleton R] [AddCommMonoid M]
    [Module R M] : Subsingleton M :=
  MulActionWithZero.subsingleton R M


/-- A semiring is `Nontrivial` provided that there exists a nontrivial module over this semiring. -/
protected theorem Module.nontrivial (R M : Type*) [Semiring R] [Nontrivial M] [AddCommMonoid M]
    [Module R M] : Nontrivial R :=
  MulActionWithZero.nontrivial R M

-- see Note [lower instance priority]

instance (priority := 910) Semiring.toModule [Semiring R] : Module R R where
  smul_add := mul_add
  add_smul := add_mul
  zero_smul := zero_mul
  smul_zero := mul_zero


instance [NonUnitalNonAssocSemiring R] : DistribSMul R R where
  smul_add := left_distrib

