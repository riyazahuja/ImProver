/-- Typeclass for scalar multiplication that preserves `0` on the right. -/
class SMulZeroClass (M A : Type*) [Zero A] extends SMul M A where
  /-- Multiplying `0` by a scalar gives `0` -/
  smul_zero : ∀ a : M, a • (0 : A) = 0


@[simp]
theorem smul_zero (a : M) : a • (0 : A) = 0 :=
  SMulZeroClass.smul_zero _


lemma smul_ite_zero (p : Prop) [Decidable p] (a : M) (b : A) :
                                                            /-
                                                              M : Type u_1
                                                              A : Type u_3
                                                              inst✝² : Zero A
                                                              inst✝¹ : SMulZeroClass M A
                                                              p : Prop
                                                              inst✝ : Decidable p
                                                              a : M
                                                              b : A
                                                              ⊢ Eq (HSMul.hSMul a (ite p b 0)) (ite p (HSMul.hSMul a b) 0)
                                                            -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
    (a • if p then b else 0) = if p then a • b else 0 := by split_ifs <;> simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


lemma smul_eq_zero_of_right (a : M) {b : A} (h : b = 0) : a • b = 0 := h.symm ▸ smul_zero a

lemma right_ne_zero_of_smul {a : M} {b : A} : a • b ≠ 0 → b ≠ 0 := mt <| smul_eq_zero_of_right a


/-- Pullback a zero-preserving scalar multiplication along an injective zero-preserving map.
See note [reducible non-instances]. -/
protected abbrev Function.Injective.smulZeroClass [Zero B] [SMul M B] (f : ZeroHom B A)
    (hf : Injective f) (smul : ∀ (c : M) (x), f (c • x) = c • f x) :
    SMulZeroClass M B where
  smul := (· • ·)
                          /-
                            M : Type u_1
                            N : Type u_2
                            A : Type u_3
                            B : Type u_4
                            α : Type u_5
                            β : Type u_6
                            inst✝³ : Zero A
                            inst✝² : SMulZeroClass M A
                            inst✝¹ : Zero B
                            inst✝ : SMul M B
                            f : ZeroHom B A
                            hf : Function.Injective ⇑f
                            smul : ∀ (c : M) (x : B), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
                            c : M
                            ⊢ Eq (f (HSMul.hSMul c 0)) (f 0)
                          -/
  smul_zero c := hf <| by simp only [smul, map_zero, smul_zero]
                          /-
                            🎉 no goals
                          -/


/-- Pushforward a zero-preserving scalar multiplication along a zero-preserving map.
See note [reducible non-instances]. -/
protected abbrev ZeroHom.smulZeroClass [Zero B] [SMul M B] (f : ZeroHom A B)
    (smul : ∀ (c : M) (x), f (c • x) = c • f x) :
    SMulZeroClass M B where
  -- Porting note: `simp` no longer works here.
                    /-
                      M : Type u_1
                      N : Type u_2
                      A : Type u_3
                      B : Type u_4
                      α : Type u_5
                      β : Type u_6
                      inst✝³ : Zero A
                      inst✝² : SMulZeroClass M A
                      inst✝¹ : Zero B
                      inst✝ : SMul M B
                      f : ZeroHom A B
                      smul : ∀ (c : M) (x : A), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
                      c : M
                      ⊢ Eq (HSMul.hSMul c 0) 0
                    -/
  smul_zero c := by rw [← map_zero f, ← smul, smul_zero]
                    /-
                      🎉 no goals
                    -/


/-- Push forward the multiplication of `R` on `M` along a compatible surjective map `f : R → S`.

See also `Function.Surjective.distribMulActionLeft`.
-/
abbrev Function.Surjective.smulZeroClassLeft {R S M : Type*} [Zero M] [SMulZeroClass R M]
    [SMul S M] (f : R → S) (hf : Function.Surjective f)
    (hsmul : ∀ (c) (x : M), f c • x = c • x) :
    SMulZeroClass S M where
  smul := (· • ·)
                                         /-
                                           M✝ : Type u_1
                                           N : Type u_2
                                           A : Type u_3
                                           B : Type u_4
                                           α : Type u_5
                                           β : Type u_6
                                           inst✝⁴ : Zero A
                                           inst✝³ : SMulZeroClass M✝ A
                                           R : Type u_7
                                           S : Type u_8
                                           M : Type u_9
                                           inst✝² : Zero M
                                           inst✝¹ : SMulZeroClass R M
                                           inst✝ : SMul S M
                                           f : R → S
                                           hf : Function.Surjective f
                                           hsmul : ∀ (c : R) (x : M), Eq (HSMul.hSMul (f c) x) (HSMul.hSMul c x)
                                           c : R
                                           ⊢ Eq (HSMul.hSMul (f c) 0) 0
                                         -/
  smul_zero := hf.forall.mpr fun c => by rw [hsmul, smul_zero]
                                         /-
                                           🎉 no goals
                                         -/


/-- Compose a `SMulZeroClass` with a function, with scalar multiplication `f r' • m`.
See note [reducible non-instances]. -/
abbrev SMulZeroClass.compFun (f : N → M) :
    SMulZeroClass N A where
  smul := SMul.comp.smul f
  smul_zero x := smul_zero (f x)


/-- Each element of the scalars defines a zero-preserving map. -/
@[simps]
def SMulZeroClass.toZeroHom (x : M) :
    ZeroHom A A where
  toFun := (x • ·)
  map_zero' := smul_zero x


/-- Typeclass for scalar multiplication that preserves `0` and `+` on the right.

This is exactly `DistribMulAction` without the `MulAction` part.
-/
@[ext]
class DistribSMul (M A : Type*) [AddZeroClass A] extends SMulZeroClass M A where
  /-- Scalar multiplication distributes across addition -/
  smul_add : ∀ (a : M) (x y : A), a • (x + y) = a • x + a • y


theorem smul_add (a : M) (b₁ b₂ : A) : a • (b₁ + b₂) = a • b₁ + a • b₂ :=
  DistribSMul.smul_add _ _ _


instance AddMonoidHom.smulZeroClass [AddZeroClass B] : SMulZeroClass M (B →+ A) where
  smul r f :=
    { toFun := fun a => r • (f a)
                      /-
                        M : Type u_1
                        N : Type u_2
                        A : Type u_3
                        B : Type u_4
                        α : Type u_5
                        β : Type u_6
                        inst✝² : AddZeroClass A
                        inst✝¹ : DistribSMul M A
                        inst✝ : AddZeroClass B
                        r : M
                        f : AddMonoidHom B A
                        ⊢ Eq ((fun a => HSMul.hSMul r (f a)) 0) 0
                      -/
      map_zero' := by simp only [map_zero, smul_zero]
                      /-
                        🎉 no goals
                      -/
                                /-
                                  M : Type u_1
                                  N : Type u_2
                                  A : Type u_3
                                  B : Type u_4
                                  α : Type u_5
                                  β : Type u_6
                                  inst✝² : AddZeroClass A
                                  inst✝¹ : DistribSMul M A
                                  inst✝ : AddZeroClass B
                                  r : M
                                  f : AddMonoidHom B A
                                  x y : B
                                  ⊢ Eq ({ toFun := fun a => HSMul.hSMul r (f a), map_zero' := ⋯ }.toFun (HAdd.hA …
                                -/
      map_add' := fun x y => by simp only [map_add, smul_add] }
                                /-
                                  🎉 no goals
                                -/
  smul_zero _ := ext fun _ => smul_zero _


/-- Pullback a distributive scalar multiplication along an injective additive monoid
homomorphism.
See note [reducible non-instances]. -/
protected abbrev Function.Injective.distribSMul [AddZeroClass B] [SMul M B] (f : B →+ A)
    (hf : Injective f) (smul : ∀ (c : M) (x), f (c • x) = c • f x) : DistribSMul M B :=
  { hf.smulZeroClass f.toZeroHom smul with
                                      /-
                                        M : Type u_1
                                        N : Type u_2
                                        A : Type u_3
                                        B : Type u_4
                                        α : Type u_5
                                        β : Type u_6
                                        inst✝³ : AddZeroClass A
                                        inst✝² : DistribSMul M A
                                        inst✝¹ : AddZeroClass B
                                        inst✝ : SMul M B
                                        f : AddMonoidHom B A
                                        hf : Function.Injective ⇑f
                                        smul : ∀ (c : M) (x : B), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
                                        c : M
                                        x y : B
                                        ⊢ Eq (f (HSMul.hSMul c (HAdd.hAdd x y))) (f (HAdd.hAdd (HSMul.hSMul c x) (HSMu …
                                      -/
    smul_add := fun c x y => hf <| by simp only [smul, map_add, smul_add] }
                                      /-
                                        🎉 no goals
                                      -/


/-- Pushforward a distributive scalar multiplication along a surjective additive monoid
homomorphism.
See note [reducible non-instances]. -/
protected abbrev Function.Surjective.distribSMul [AddZeroClass B] [SMul M B] (f : A →+ B)
    (hf : Surjective f) (smul : ∀ (c : M) (x), f (c • x) = c • f x) : DistribSMul M B :=
  { f.toZeroHom.smulZeroClass smul with
    smul_add := fun c x y => by
      /-
        M : Type u_1
        N : Type u_2
        A : Type u_3
        B : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝³ : AddZeroClass A
        inst✝² : DistribSMul M A
        inst✝¹ : AddZeroClass B
        inst✝ : SMul M B
        f : AddMonoidHom A B
        hf : Function.Surjective ⇑f
        smul : ∀ (c : M) (x : A), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
        c : M
        x y : B
        ⊢ Eq (HSMul.hSMul c (HAdd.hAdd x y)) (HAdd.hAdd (HSMul.hSMul c x) (HSMul.hSMul …
      -/
      rcases hf x with ⟨x, rfl⟩
      /-
        case intro
        M : Type u_1
        N : Type u_2
        A : Type u_3
        B : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝³ : AddZeroClass A
        inst✝² : DistribSMul M A
        inst✝¹ : AddZeroClass B
        inst✝ : SMul M B
        f : AddMonoidHom A B
        hf : Function.Surjective ⇑f
        smul : ∀ (c : M) (x : A), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
        c : M
        y : B
        x : A
        ⊢ Eq (HSMul.hSMul c (HAdd.hAdd (f x) y)) (HAdd.hAdd (HSMul.hSMul c (f x)) (HSM …
      -/
      rcases hf y with ⟨y, rfl⟩
      /-
        case intro.intro
        M : Type u_1
        N : Type u_2
        A : Type u_3
        B : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝³ : AddZeroClass A
        inst✝² : DistribSMul M A
        inst✝¹ : AddZeroClass B
        inst✝ : SMul M B
        f : AddMonoidHom A B
        hf : Function.Surjective ⇑f
        smul : ∀ (c : M) (x : A), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
        c : M
        x y : A
        ⊢ Eq (HSMul.hSMul c (HAdd.hAdd (f x) (f y))) (HAdd.hAdd (HSMul.hSMul c (f x))  …
      -/
      simp only [smul_add, ← smul, ← map_add] }
      /-
        🎉 no goals
      -/


/-- Push forward the multiplication of `R` on `M` along a compatible surjective map `f : R → S`.

See also `Function.Surjective.distribMulActionLeft`.
-/
abbrev Function.Surjective.distribSMulLeft {R S M : Type*} [AddZeroClass M] [DistribSMul R M]
    [SMul S M] (f : R → S) (hf : Function.Surjective f)
    (hsmul : ∀ (c) (x : M), f c • x = c • x) : DistribSMul S M :=
  { hf.smulZeroClassLeft f hsmul with
                                              /-
                                                M✝ : Type u_1
                                                N : Type u_2
                                                A : Type u_3
                                                B : Type u_4
                                                α : Type u_5
                                                β : Type u_6
                                                inst✝⁴ : AddZeroClass A
                                                inst✝³ : DistribSMul M✝ A
                                                R : Type u_7
                                                S : Type u_8
                                                M : Type u_9
                                                inst✝² : AddZeroClass M
                                                inst✝¹ : DistribSMul R M
                                                inst✝ : SMul S M
                                                f : R → S
                                                hf : Function.Surjective f
                                                hsmul : ∀ (c : R) (x : M), Eq (HSMul.hSMul (f c) x) (HSMul.hSMul c x)
                                                c : R
                                                x y : M
                                                ⊢ Eq (HSMul.hSMul (f c) (HAdd.hAdd x y)) (HAdd.hAdd (HSMul.hSMul (f c) x) (HSM …
                                              -/
    smul_add := hf.forall.mpr fun c x y => by simp only [hsmul, smul_add] }
                                              /-
                                                🎉 no goals
                                              -/


/-- Compose a `DistribSMul` with a function, with scalar multiplication `f r' • m`.
See note [reducible non-instances]. -/
abbrev DistribSMul.compFun (f : N → M) : DistribSMul N A :=
  { SMulZeroClass.compFun A f with
    smul_add := fun x => smul_add (f x) }


/-- Each element of the scalars defines an additive monoid homomorphism. -/
@[simps]
def DistribSMul.toAddMonoidHom (x : M) : A →+ A :=
  { SMulZeroClass.toZeroHom A x with toFun := (x • ·), map_add' := smul_add x }


/-- Typeclass for multiplicative actions on additive structures. This generalizes group modules. -/
@[ext]
class DistribMulAction (M A : Type*) [Monoid M] [AddMonoid A] extends MulAction M A where
  /-- Multiplying `0` by a scalar gives `0` -/
  smul_zero : ∀ a : M, a • (0 : A) = 0
  /-- Scalar multiplication distributes across addition -/
  smul_add : ∀ (a : M) (x y : A), a • (x + y) = a • x + a • y


instance (priority := 100) DistribMulAction.toDistribSMul : DistribSMul M A :=
  { ‹DistribMulAction M A› with }

-- Porting note: this probably is no longer relevant.

/-- Pullback a distributive multiplicative action along an injective additive monoid
homomorphism.
See note [reducible non-instances]. -/
protected abbrev Function.Injective.distribMulAction [AddMonoid B] [SMul M B] (f : B →+ A)
    (hf : Injective f) (smul : ∀ (c : M) (x), f (c • x) = c • f x) : DistribMulAction M B :=
  { hf.distribSMul f smul, hf.mulAction f smul with }


/-- Pushforward a distributive multiplicative action along a surjective additive monoid
homomorphism.
See note [reducible non-instances]. -/
protected abbrev Function.Surjective.distribMulAction [AddMonoid B] [SMul M B] (f : A →+ B)
    (hf : Surjective f) (smul : ∀ (c : M) (x), f (c • x) = c • f x) : DistribMulAction M B :=
  { hf.distribSMul f smul, hf.mulAction f smul with }


/-- Each element of the monoid defines an additive monoid homomorphism. -/
@[simps!]
def DistribMulAction.toAddMonoidHom (x : M) : A →+ A :=
  DistribSMul.toAddMonoidHom A x


/-- Each element of the monoid defines an additive monoid homomorphism. -/
@[simps]
def DistribMulAction.toAddMonoidEnd :
    M →* AddMonoid.End A where
  toFun := DistribMulAction.toAddMonoidHom A
  map_one' := AddMonoidHom.ext <| one_smul M
  map_mul' x y := AddMonoidHom.ext <| mul_smul x y


instance AddMonoid.nat_smulCommClass :
    SMulCommClass ℕ M
      A where smul_comm n x y := ((DistribMulAction.toAddMonoidHom A x).map_nsmul y n).symm

-- `SMulCommClass.symm` is not registered as an instance, as it would cause a loop

instance AddMonoid.nat_smulCommClass' : SMulCommClass M ℕ A :=
  SMulCommClass.symm _ _ _


instance AddGroup.int_smulCommClass : SMulCommClass ℤ M A where
  smul_comm n x y := ((DistribMulAction.toAddMonoidHom A x).map_zsmul y n).symm

-- `SMulCommClass.symm` is not registered as an instance, as it would cause a loop

instance AddGroup.int_smulCommClass' : SMulCommClass M ℤ A :=
  SMulCommClass.symm _ _ _


@[simp]
theorem smul_neg (r : M) (x : A) : r • -x = -(r • x) :=
                                   /-
                                     M : Type u_1
                                     A : Type u_3
                                     inst✝² : Monoid M
                                     inst✝¹ : AddGroup A
                                     inst✝ : DistribMulAction M A
                                     r : M
                                     x : A
                                     ⊢ Eq (HAdd.hAdd (HSMul.hSMul r (Neg.neg x)) (HSMul.hSMul r x)) 0
                                   -/
  eq_neg_of_add_eq_zero_left <| by rw [← smul_add, neg_add_cancel, smul_zero]
                                   /-
                                     🎉 no goals
                                   -/


theorem smul_sub (r : M) (x y : A) : r • (x - y) = r • x - r • y := by
  /-
    M : Type u_1
    A : Type u_3
    inst✝² : Monoid M
    inst✝¹ : AddGroup A
    inst✝ : DistribMulAction M A
    r : M
    x y : A
    ⊢ Eq (HSMul.hSMul r (HSub.hSub x y)) (HSub.hSub (HSMul.hSMul r x) (HSMul.hSMul …
  -/
  rw [sub_eq_add_neg, sub_eq_add_neg, smul_add, smul_neg]
  /-
    🎉 no goals
  -/


/-- Typeclass for multiplicative actions on multiplicative structures. This generalizes
conjugation actions. -/
@[ext]
class MulDistribMulAction (M : Type*) (A : Type*) [Monoid M] [Monoid A] extends
  MulAction M A where
  /-- Distributivity of `•` across `*` -/
  smul_mul : ∀ (r : M) (x y : A), r • (x * y) = r • x * r • y
  /-- Multiplying `1` by a scalar gives `1` -/
  smul_one : ∀ r : M, r • (1 : A) = 1


theorem smul_mul' (a : M) (b₁ b₂ : A) : a • (b₁ * b₂) = a • b₁ * a • b₂ :=
  MulDistribMulAction.smul_mul _ _ _


/-- Pullback a multiplicative distributive multiplicative action along an injective monoid
homomorphism.
See note [reducible non-instances]. -/
protected abbrev Function.Injective.mulDistribMulAction [Monoid B] [SMul M B] (f : B →* A)
    (hf : Injective f) (smul : ∀ (c : M) (x), f (c • x) = c • f x) : MulDistribMulAction M B :=
  { hf.mulAction f smul with
                                      /-
                                        M : Type u_1
                                        N : Type u_2
                                        A : Type u_3
                                        B : Type u_4
                                        α : Type u_5
                                        β : Type u_6
                                        inst✝⁴ : Monoid M
                                        inst✝³ : Monoid A
                                        inst✝² : MulDistribMulAction M A
                                        inst✝¹ : Monoid B
                                        inst✝ : SMul M B
                                        f : MonoidHom B A
                                        hf : Function.Injective ⇑f
                                        smul : ∀ (c : M) (x : B), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
                                        c : M
                                        x y : B
                                        ⊢ Eq (f (HSMul.hSMul c (HMul.hMul x y))) (f (HMul.hMul (HSMul.hSMul c x) (HSMu …
                                      -/
    smul_mul := fun c x y => hf <| by simp only [smul, f.map_mul, smul_mul'],
                                      /-
                                        🎉 no goals
                                      -/
                                  /-
                                    M : Type u_1
                                    N : Type u_2
                                    A : Type u_3
                                    B : Type u_4
                                    α : Type u_5
                                    β : Type u_6
                                    inst✝⁴ : Monoid M
                                    inst✝³ : Monoid A
                                    inst✝² : MulDistribMulAction M A
                                    inst✝¹ : Monoid B
                                    inst✝ : SMul M B
                                    f : MonoidHom B A
                                    hf : Function.Injective ⇑f
                                    smul : ∀ (c : M) (x : B), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
                                    c : M
                                    ⊢ Eq (f (HSMul.hSMul c 1)) (f 1)
                                  -/
    smul_one := fun c => hf <| by simp only [smul, f.map_one, smul_one] }
                                  /-
                                    🎉 no goals
                                  -/


/-- Pushforward a multiplicative distributive multiplicative action along a surjective monoid
homomorphism.
See note [reducible non-instances]. -/
protected abbrev Function.Surjective.mulDistribMulAction [Monoid B] [SMul M B] (f : A →* B)
    (hf : Surjective f) (smul : ∀ (c : M) (x), f (c • x) = c • f x) : MulDistribMulAction M B :=
  { hf.mulAction f smul with
    smul_mul := fun c x y => by
      /-
        M : Type u_1
        N : Type u_2
        A : Type u_3
        B : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝⁴ : Monoid M
        inst✝³ : Monoid A
        inst✝² : MulDistribMulAction M A
        inst✝¹ : Monoid B
        inst✝ : SMul M B
        f : MonoidHom A B
        hf : Function.Surjective ⇑f
        smul : ∀ (c : M) (x : A), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
        c : M
        x y : B
        ⊢ Eq (HSMul.hSMul c (HMul.hMul x y)) (HMul.hMul (HSMul.hSMul c x) (HSMul.hSMul …
      -/
      rcases hf x with ⟨x, rfl⟩
      /-
        case intro
        M : Type u_1
        N : Type u_2
        A : Type u_3
        B : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝⁴ : Monoid M
        inst✝³ : Monoid A
        inst✝² : MulDistribMulAction M A
        inst✝¹ : Monoid B
        inst✝ : SMul M B
        f : MonoidHom A B
        hf : Function.Surjective ⇑f
        smul : ∀ (c : M) (x : A), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
        c : M
        y : B
        x : A
        ⊢ Eq (HSMul.hSMul c (HMul.hMul (f x) y)) (HMul.hMul (HSMul.hSMul c (f x)) (HSM …
      -/
      rcases hf y with ⟨y, rfl⟩
      /-
        case intro.intro
        M : Type u_1
        N : Type u_2
        A : Type u_3
        B : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝⁴ : Monoid M
        inst✝³ : Monoid A
        inst✝² : MulDistribMulAction M A
        inst✝¹ : Monoid B
        inst✝ : SMul M B
        f : MonoidHom A B
        hf : Function.Surjective ⇑f
        smul : ∀ (c : M) (x : A), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
        c : M
        x y : A
        ⊢ Eq (HSMul.hSMul c (HMul.hMul (f x) (f y))) (HMul.hMul (HSMul.hSMul c (f x))  …
      -/
      simp only [smul_mul', ← smul, ← f.map_mul],
      /-
        🎉 no goals
      -/
                            /-
                              M : Type u_1
                              N : Type u_2
                              A : Type u_3
                              B : Type u_4
                              α : Type u_5
                              β : Type u_6
                              inst✝⁴ : Monoid M
                              inst✝³ : Monoid A
                              inst✝² : MulDistribMulAction M A
                              inst✝¹ : Monoid B
                              inst✝ : SMul M B
                              f : MonoidHom A B
                              hf : Function.Surjective ⇑f
                              smul : ∀ (c : M) (x : A), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
                              c : M
                              ⊢ Eq (HSMul.hSMul c 1) 1
                            -/
    smul_one := fun c => by rw [← f.map_one, ← smul, smul_one] }
                            /-
                              🎉 no goals
                            -/


/-- Scalar multiplication by `r` as a `MonoidHom`. -/
def MulDistribMulAction.toMonoidHom (r : M) :
    A →* A where
  toFun := (r • ·)
  map_one' := smul_one r
  map_mul' := smul_mul' r


@[simp]
theorem MulDistribMulAction.toMonoidHom_apply (r : M) (x : A) :
    MulDistribMulAction.toMonoidHom A r x = r • x :=
  rfl


@[simp] lemma smul_pow' (r : M) (x : A) (n : ℕ) : r • x ^ n = (r • x) ^ n :=
  (MulDistribMulAction.toMonoidHom _ _).map_pow _ _


@[simp]
theorem smul_inv' (r : M) (x : A) : r • x⁻¹ = (r • x)⁻¹ :=
  (MulDistribMulAction.toMonoidHom A r).map_inv x


theorem smul_div' (r : M) (x y : A) : r • (x / y) = r • x / r • y :=
  map_div (MulDistribMulAction.toMonoidHom A r) x y


lemma smul_eq_zero_iff_eq (a : α) {x : β} : a • x = 0 ↔ x = 0 :=
               /-
                 α : Type u_5
                 β : Type u_6
                 inst✝² : Group α
                 inst✝¹ : AddMonoid β
                 inst✝ : DistribMulAction α β
                 a : α
                 x : β
                 h : Eq (HSMul.hSMul a x) 0
                 ⊢ Eq x 0
               -/
  ⟨fun h => by rw [← inv_smul_smul a x, h, smul_zero], fun h => h.symm ▸ smul_zero _⟩
               /-
                 🎉 no goals
               -/


lemma smul_ne_zero_iff_ne (a : α) {x : β} : a • x ≠ 0 ↔ x ≠ 0 :=
  not_congr <| smul_eq_zero_iff_eq a


