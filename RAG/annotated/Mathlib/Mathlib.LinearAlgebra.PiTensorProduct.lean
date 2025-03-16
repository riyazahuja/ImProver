/-- The relation on `FreeAddMonoid (R × Π i, s i)` that generates a congruence whose quotient is
the tensor product. -/
inductive Eqv : FreeAddMonoid (R × Π i, s i) → FreeAddMonoid (R × Π i, s i) → Prop
  | of_zero : ∀ (r : R) (f : Π i, s i) (i : ι) (_ : f i = 0), Eqv (FreeAddMonoid.of (r, f)) 0
  | of_zero_scalar : ∀ f : Π i, s i, Eqv (FreeAddMonoid.of (0, f)) 0
  | of_add : ∀ (_ : DecidableEq ι) (r : R) (f : Π i, s i) (i : ι) (m₁ m₂ : s i),
      Eqv (FreeAddMonoid.of (r, update f i m₁) + FreeAddMonoid.of (r, update f i m₂))
        (FreeAddMonoid.of (r, update f i (m₁ + m₂)))
  | of_add_scalar : ∀ (r r' : R) (f : Π i, s i),
      Eqv (FreeAddMonoid.of (r, f) + FreeAddMonoid.of (r', f)) (FreeAddMonoid.of (r + r', f))
  | of_smul : ∀ (_ : DecidableEq ι) (r : R) (f : Π i, s i) (i : ι) (r' : R),
      Eqv (FreeAddMonoid.of (r, update f i (r' • f i))) (FreeAddMonoid.of (r' * r, f))
  | add_comm : ∀ x y, Eqv (x + y) (y + x)


/-- `PiTensorProduct R s` with `R` a commutative semiring and `s : ι → Type*` is the tensor
  product of all the `s i`'s. This is denoted by `⨂[R] i, s i`. -/
def PiTensorProduct : Type _ :=
  (addConGen (PiTensorProduct.Eqv R s)).Quotient


unsuppress_compilation in
/-- This enables the notation `⨂[R] i : ι, s i` for the pi tensor product `PiTensorProduct`,
given an indexed family of types `s : ι → Type*`. -/
scoped[TensorProduct] notation3:100"⨂["R"] "(...)", "r:(scoped f => PiTensorProduct R f) => r


instance : AddCommMonoid (⨂[R] i, s i) :=
  { (addConGen (PiTensorProduct.Eqv R s)).addMonoid with
    add_comm := fun x y ↦
      AddCon.induction_on₂ x y fun _ _ ↦
        Quotient.sound' <| AddConGen.Rel.of _ _ <| Eqv.add_comm _ _ }


instance : Inhabited (⨂[R] i, s i) := ⟨0⟩


/-- `tprodCoeff R r f` with `r : R` and `f : Π i, s i` is the tensor product of the vectors `f i`
over all `i : ι`, multiplied by the coefficient `r`. Note that this is meant as an auxiliary
definition for this file alone, and that one should use `tprod` defined below for most purposes. -/
def tprodCoeff (r : R) (f : Π i, s i) : ⨂[R] i, s i :=
  AddCon.mk' _ <| FreeAddMonoid.of (r, f)


theorem zero_tprodCoeff (f : Π i, s i) : tprodCoeff R 0 f = 0 :=
  Quotient.sound' <| AddConGen.Rel.of _ _ <| Eqv.of_zero_scalar _


theorem zero_tprodCoeff' (z : R) (f : Π i, s i) (i : ι) (hf : f i = 0) : tprodCoeff R z f = 0 :=
  Quotient.sound' <| AddConGen.Rel.of _ _ <| Eqv.of_zero _ _ i hf


theorem add_tprodCoeff [DecidableEq ι] (z : R) (f : Π i, s i) (i : ι) (m₁ m₂ : s i) :
    tprodCoeff R z (update f i m₁) + tprodCoeff R z (update f i m₂) =
      tprodCoeff R z (update f i (m₁ + m₂)) :=
  Quotient.sound' <| AddConGen.Rel.of _ _ (Eqv.of_add _ z f i m₁ m₂)


theorem add_tprodCoeff' (z₁ z₂ : R) (f : Π i, s i) :
    tprodCoeff R z₁ f + tprodCoeff R z₂ f = tprodCoeff R (z₁ + z₂) f :=
  Quotient.sound' <| AddConGen.Rel.of _ _ (Eqv.of_add_scalar z₁ z₂ f)


theorem smul_tprodCoeff_aux [DecidableEq ι] (z : R) (f : Π i, s i) (i : ι) (r : R) :
    tprodCoeff R z (update f i (r • f i)) = tprodCoeff R (r * z) f :=
  Quotient.sound' <| AddConGen.Rel.of _ _ <| Eqv.of_smul _ _ _ _ _


theorem smul_tprodCoeff [DecidableEq ι] (z : R) (f : Π i, s i) (i : ι) (r : R₁) [SMul R₁ R]
    [IsScalarTower R₁ R R] [SMul R₁ (s i)] [IsScalarTower R₁ R (s i)] :
    tprodCoeff R z (update f i (r • f i)) = tprodCoeff R (r • z) f := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁷ : CommSemiring R
    R₁ : Type u_5
    s : ι → Type u_7
    inst✝⁶ : (i : ι) → AddCommMonoid (s i)
    inst✝⁵ : (i : ι) → Module R (s i)
    inst✝⁴ : DecidableEq ι
    z : R
    f : (i : ι) → s i
    i : ι
    r : R₁
    inst✝³ : SMul R₁ R
    inst✝² : IsScalarTower R₁ R R
    inst✝¹ : SMul R₁ (s i)
    inst✝ : IsScalarTower R₁ R (s i)
    ⊢ Eq (PiTensorProduct.tprodCoeff R z (Function.update f i (HSMul.hSMul r (f i) …
  -/
  have h₁ : r • z = r • (1 : R) * z := by rw [smul_mul_assoc, one_mul]
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁷ : CommSemiring R
    R₁ : Type u_5
    s : ι → Type u_7
    inst✝⁶ : (i : ι) → AddCommMonoid (s i)
    inst✝⁵ : (i : ι) → Module R (s i)
    inst✝⁴ : DecidableEq ι
    z : R
    f : (i : ι) → s i
    i : ι
    r : R₁
    inst✝³ : SMul R₁ R
    inst✝² : IsScalarTower R₁ R R
    inst✝¹ : SMul R₁ (s i)
    inst✝ : IsScalarTower R₁ R (s i)
    h₁ : Eq (HSMul.hSMul r z) (HMul.hMul (HSMul.hSMul r 1) z)
    ⊢ Eq (PiTensorProduct.tprodCoeff R z (Function.update f i (HSMul.hSMul r (f i) …
  -/
  have h₂ : r • f i = (r • (1 : R)) • f i := (smul_one_smul _ _ _).symm
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁷ : CommSemiring R
    R₁ : Type u_5
    s : ι → Type u_7
    inst✝⁶ : (i : ι) → AddCommMonoid (s i)
    inst✝⁵ : (i : ι) → Module R (s i)
    inst✝⁴ : DecidableEq ι
    z : R
    f : (i : ι) → s i
    i : ι
    r : R₁
    inst✝³ : SMul R₁ R
    inst✝² : IsScalarTower R₁ R R
    inst✝¹ : SMul R₁ (s i)
    inst✝ : IsScalarTower R₁ R (s i)
    h₁ : Eq (HSMul.hSMul r z) (HMul.hMul (HSMul.hSMul r 1) z)
    h₂ : Eq (HSMul.hSMul r (f i)) (HSMul.hSMul (HSMul.hSMul r 1) (f i))
    ⊢ Eq (PiTensorProduct.tprodCoeff R z (Function.update f i (HSMul.hSMul r (f i) …
  -/
  rw [h₁, h₂]
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁷ : CommSemiring R
    R₁ : Type u_5
    s : ι → Type u_7
    inst✝⁶ : (i : ι) → AddCommMonoid (s i)
    inst✝⁵ : (i : ι) → Module R (s i)
    inst✝⁴ : DecidableEq ι
    z : R
    f : (i : ι) → s i
    i : ι
    r : R₁
    inst✝³ : SMul R₁ R
    inst✝² : IsScalarTower R₁ R R
    inst✝¹ : SMul R₁ (s i)
    inst✝ : IsScalarTower R₁ R (s i)
    h₁ : Eq (HSMul.hSMul r z) (HMul.hMul (HSMul.hSMul r 1) z)
    h₂ : Eq (HSMul.hSMul r (f i)) (HSMul.hSMul (HSMul.hSMul r 1) (f i))
    ⊢ Eq (PiTensorProduct.tprodCoeff R z (Function.update f i (HSMul.hSMul (HSMul. …
  -/
  exact smul_tprodCoeff_aux z f i _
  /-
    🎉 no goals
  -/


/-- Construct an `AddMonoidHom` from `(⨂[R] i, s i)` to some space `F` from a function
`φ : (R × Π i, s i) → F` with the appropriate properties. -/
def liftAddHom (φ : (R × Π i, s i) → F)
    (C0 : ∀ (r : R) (f : Π i, s i) (i : ι) (_ : f i = 0), φ (r, f) = 0)
    (C0' : ∀ f : Π i, s i, φ (0, f) = 0)
    (C_add : ∀ [DecidableEq ι] (r : R) (f : Π i, s i) (i : ι) (m₁ m₂ : s i),
      φ (r, update f i m₁) + φ (r, update f i m₂) = φ (r, update f i (m₁ + m₂)))
    (C_add_scalar : ∀ (r r' : R) (f : Π i, s i), φ (r, f) + φ (r', f) = φ (r + r', f))
    (C_smul : ∀ [DecidableEq ι] (r : R) (f : Π i, s i) (i : ι) (r' : R),
      φ (r, update f i (r' • f i)) = φ (r' * r, f)) :
    (⨂[R] i, s i) →+ F :=
  (addConGen (PiTensorProduct.Eqv R s)).lift (FreeAddMonoid.lift φ) <|
    AddCon.addConGen_le fun x y hxy ↦
      match hxy with
      | Eqv.of_zero r' f i hf =>
                                   /-
                                     ι : Type u_1
                                     ι₂ : Type u_2
                                     ι₃ : Type u_3
                                     R : Type u_4
                                     inst✝⁷ : CommSemiring R
                                     R₁ : Type u_5
                                     R₂ : Type u_6
                                     s : ι → Type u_7
                                     inst✝⁶ : (i : ι) → AddCommMonoid (s i)
                                     inst✝⁵ : (i : ι) → Module R (s i)
                                     M : Type u_8
                                     inst✝⁴ : AddCommMonoid M
                                     inst✝³ : Module R M
                                     E : Type u_9
                                     inst✝² : AddCommMonoid E
                                     inst✝¹ : Module R E
                                     F : Type u_10
                                     inst✝ : AddCommMonoid F
                                     φ : Prod R ((i : ι) → s i) → F
                                     C0 : ∀ (r : R) (f : (i : ι) → s i) (i : ι), Eq (f i) 0 → Eq (φ { fst := r, snd …
                                     C0' : ∀ (f : (i : ι) → s i), Eq (φ { fst := 0, snd := f }) 0
                                     C_add : ∀ [inst : DecidableEq ι] (r : R) (f : (i : ι) → s i) (i : ι) (m₁ m₂ :  …
                                     C_add_scalar : ∀ (r r' : R) (f : (i : ι) → s i), Eq (HAdd.hAdd (φ { fst := r,  …
                                     C_smul : ∀ [inst : DecidableEq ι] (r : R) (f : (i : ι) → s i) (i : ι) (r' : R) …
                                     x y : FreeAddMonoid (Prod R ((i : ι) → s i))
                                     hxy : PiTensorProduct.Eqv R s x y
                                     r' : R
                                     f : (i : ι) → s i
                                     i : ι
                                     hf : Eq (f i) 0
                                     ⊢ Eq ((FreeAddMonoid.lift φ) (FreeAddMonoid.of { fst := r', snd := f })) ((Fre …
                                   -/
        (AddCon.ker_rel _).2 <| by simp [FreeAddMonoid.lift_eval_of, C0 r' f i hf]
                                   /-
                                     🎉 no goals
                                   -/
      | Eqv.of_zero_scalar f =>
                                   /-
                                     ι : Type u_1
                                     ι₂ : Type u_2
                                     ι₃ : Type u_3
                                     R : Type u_4
                                     inst✝⁷ : CommSemiring R
                                     R₁ : Type u_5
                                     R₂ : Type u_6
                                     s : ι → Type u_7
                                     inst✝⁶ : (i : ι) → AddCommMonoid (s i)
                                     inst✝⁵ : (i : ι) → Module R (s i)
                                     M : Type u_8
                                     inst✝⁴ : AddCommMonoid M
                                     inst✝³ : Module R M
                                     E : Type u_9
                                     inst✝² : AddCommMonoid E
                                     inst✝¹ : Module R E
                                     F : Type u_10
                                     inst✝ : AddCommMonoid F
                                     φ : Prod R ((i : ι) → s i) → F
                                     C0 : ∀ (r : R) (f : (i : ι) → s i) (i : ι), Eq (f i) 0 → Eq (φ { fst := r, snd …
                                     C0' : ∀ (f : (i : ι) → s i), Eq (φ { fst := 0, snd := f }) 0
                                     C_add : ∀ [inst : DecidableEq ι] (r : R) (f : (i : ι) → s i) (i : ι) (m₁ m₂ :  …
                                     C_add_scalar : ∀ (r r' : R) (f : (i : ι) → s i), Eq (HAdd.hAdd (φ { fst := r,  …
                                     C_smul : ∀ [inst : DecidableEq ι] (r : R) (f : (i : ι) → s i) (i : ι) (r' : R) …
                                     x y : FreeAddMonoid (Prod R ((i : ι) → s i))
                                     hxy : PiTensorProduct.Eqv R s x y
                                     f : (i : ι) → s i
                                     ⊢ Eq ((FreeAddMonoid.lift φ) (FreeAddMonoid.of { fst := 0, snd := f })) ((Free …
                                   -/
        (AddCon.ker_rel _).2 <| by simp [FreeAddMonoid.lift_eval_of, C0']
                                   /-
                                     🎉 no goals
                                   -/
      | Eqv.of_add inst z f i m₁ m₂ =>
                                   /-
                                     ι : Type u_1
                                     ι₂ : Type u_2
                                     ι₃ : Type u_3
                                     R : Type u_4
                                     inst✝⁷ : CommSemiring R
                                     R₁ : Type u_5
                                     R₂ : Type u_6
                                     s : ι → Type u_7
                                     inst✝⁶ : (i : ι) → AddCommMonoid (s i)
                                     inst✝⁵ : (i : ι) → Module R (s i)
                                     M : Type u_8
                                     inst✝⁴ : AddCommMonoid M
                                     inst✝³ : Module R M
                                     E : Type u_9
                                     inst✝² : AddCommMonoid E
                                     inst✝¹ : Module R E
                                     F : Type u_10
                                     inst✝ : AddCommMonoid F
                                     φ : Prod R ((i : ι) → s i) → F
                                     C0 : ∀ (r : R) (f : (i : ι) → s i) (i : ι), Eq (f i) 0 → Eq (φ { fst := r, snd …
                                     C0' : ∀ (f : (i : ι) → s i), Eq (φ { fst := 0, snd := f }) 0
                                     C_add : ∀ [inst : DecidableEq ι] (r : R) (f : (i : ι) → s i) (i : ι) (m₁ m₂ :  …
                                     C_add_scalar : ∀ (r r' : R) (f : (i : ι) → s i), Eq (HAdd.hAdd (φ { fst := r,  …
                                     C_smul : ∀ [inst : DecidableEq ι] (r : R) (f : (i : ι) → s i) (i : ι) (r' : R) …
                                     x y : FreeAddMonoid (Prod R ((i : ι) → s i))
                                     hxy : PiTensorProduct.Eqv R s x y
                                     inst : DecidableEq ι
                                     z : R
                                     f : (i : ι) → s i
                                     i : ι
                                     m₁ m₂ : s i
                                     ⊢ Eq ((FreeAddMonoid.lift φ) (HAdd.hAdd (FreeAddMonoid.of { fst := z, snd := F …
                                   -/
        (AddCon.ker_rel _).2 <| by simp [FreeAddMonoid.lift_eval_of, @C_add inst]
                                   /-
                                     🎉 no goals
                                   -/
      | Eqv.of_add_scalar z₁ z₂ f =>
                                   /-
                                     ι : Type u_1
                                     ι₂ : Type u_2
                                     ι₃ : Type u_3
                                     R : Type u_4
                                     inst✝⁷ : CommSemiring R
                                     R₁ : Type u_5
                                     R₂ : Type u_6
                                     s : ι → Type u_7
                                     inst✝⁶ : (i : ι) → AddCommMonoid (s i)
                                     inst✝⁵ : (i : ι) → Module R (s i)
                                     M : Type u_8
                                     inst✝⁴ : AddCommMonoid M
                                     inst✝³ : Module R M
                                     E : Type u_9
                                     inst✝² : AddCommMonoid E
                                     inst✝¹ : Module R E
                                     F : Type u_10
                                     inst✝ : AddCommMonoid F
                                     φ : Prod R ((i : ι) → s i) → F
                                     C0 : ∀ (r : R) (f : (i : ι) → s i) (i : ι), Eq (f i) 0 → Eq (φ { fst := r, snd …
                                     C0' : ∀ (f : (i : ι) → s i), Eq (φ { fst := 0, snd := f }) 0
                                     C_add : ∀ [inst : DecidableEq ι] (r : R) (f : (i : ι) → s i) (i : ι) (m₁ m₂ :  …
                                     C_add_scalar : ∀ (r r' : R) (f : (i : ι) → s i), Eq (HAdd.hAdd (φ { fst := r,  …
                                     C_smul : ∀ [inst : DecidableEq ι] (r : R) (f : (i : ι) → s i) (i : ι) (r' : R) …
                                     x y : FreeAddMonoid (Prod R ((i : ι) → s i))
                                     hxy : PiTensorProduct.Eqv R s x y
                                     z₁ z₂ : R
                                     f : (i : ι) → s i
                                     ⊢ Eq ((FreeAddMonoid.lift φ) (HAdd.hAdd (FreeAddMonoid.of { fst := z₁, snd :=  …
                                   -/
        (AddCon.ker_rel _).2 <| by simp [FreeAddMonoid.lift_eval_of, C_add_scalar]
                                   /-
                                     🎉 no goals
                                   -/
      | Eqv.of_smul inst z f i r' =>
                                   /-
                                     ι : Type u_1
                                     ι₂ : Type u_2
                                     ι₃ : Type u_3
                                     R : Type u_4
                                     inst✝⁷ : CommSemiring R
                                     R₁ : Type u_5
                                     R₂ : Type u_6
                                     s : ι → Type u_7
                                     inst✝⁶ : (i : ι) → AddCommMonoid (s i)
                                     inst✝⁵ : (i : ι) → Module R (s i)
                                     M : Type u_8
                                     inst✝⁴ : AddCommMonoid M
                                     inst✝³ : Module R M
                                     E : Type u_9
                                     inst✝² : AddCommMonoid E
                                     inst✝¹ : Module R E
                                     F : Type u_10
                                     inst✝ : AddCommMonoid F
                                     φ : Prod R ((i : ι) → s i) → F
                                     C0 : ∀ (r : R) (f : (i : ι) → s i) (i : ι), Eq (f i) 0 → Eq (φ { fst := r, snd …
                                     C0' : ∀ (f : (i : ι) → s i), Eq (φ { fst := 0, snd := f }) 0
                                     C_add : ∀ [inst : DecidableEq ι] (r : R) (f : (i : ι) → s i) (i : ι) (m₁ m₂ :  …
                                     C_add_scalar : ∀ (r r' : R) (f : (i : ι) → s i), Eq (HAdd.hAdd (φ { fst := r,  …
                                     C_smul : ∀ [inst : DecidableEq ι] (r : R) (f : (i : ι) → s i) (i : ι) (r' : R) …
                                     x y : FreeAddMonoid (Prod R ((i : ι) → s i))
                                     hxy : PiTensorProduct.Eqv R s x y
                                     inst : DecidableEq ι
                                     z : R
                                     f : (i : ι) → s i
                                     i : ι
                                     r' : R
                                     ⊢ Eq ((FreeAddMonoid.lift φ) (FreeAddMonoid.of { fst := z, snd := Function.upd …
                                   -/
        (AddCon.ker_rel _).2 <| by simp [FreeAddMonoid.lift_eval_of, @C_smul inst]
                                   /-
                                     🎉 no goals
                                   -/
      | Eqv.add_comm x y =>
                                   /-
                                     ι : Type u_1
                                     ι₂ : Type u_2
                                     ι₃ : Type u_3
                                     R : Type u_4
                                     inst✝⁷ : CommSemiring R
                                     R₁ : Type u_5
                                     R₂ : Type u_6
                                     s : ι → Type u_7
                                     inst✝⁶ : (i : ι) → AddCommMonoid (s i)
                                     inst✝⁵ : (i : ι) → Module R (s i)
                                     M : Type u_8
                                     inst✝⁴ : AddCommMonoid M
                                     inst✝³ : Module R M
                                     E : Type u_9
                                     inst✝² : AddCommMonoid E
                                     inst✝¹ : Module R E
                                     F : Type u_10
                                     inst✝ : AddCommMonoid F
                                     φ : Prod R ((i : ι) → s i) → F
                                     C0 : ∀ (r : R) (f : (i : ι) → s i) (i : ι), Eq (f i) 0 → Eq (φ { fst := r, snd …
                                     C0' : ∀ (f : (i : ι) → s i), Eq (φ { fst := 0, snd := f }) 0
                                     C_add : ∀ [inst : DecidableEq ι] (r : R) (f : (i : ι) → s i) (i : ι) (m₁ m₂ :  …
                                     C_add_scalar : ∀ (r r' : R) (f : (i : ι) → s i), Eq (HAdd.hAdd (φ { fst := r,  …
                                     C_smul : ∀ [inst : DecidableEq ι] (r : R) (f : (i : ι) → s i) (i : ι) (r' : R) …
                                     x✝ y✝ : FreeAddMonoid (Prod R ((i : ι) → s i))
                                     hxy : PiTensorProduct.Eqv R s x✝ y✝
                                     x y : FreeAddMonoid (Prod R ((i : ι) → s i))
                                     ⊢ Eq ((FreeAddMonoid.lift φ) (HAdd.hAdd x y)) ((FreeAddMonoid.lift φ) (HAdd.hA …
                                   -/
        (AddCon.ker_rel _).2 <| by simp_rw [AddMonoidHom.map_add, add_comm]
                                   /-
                                     🎉 no goals
                                   -/


/-- Induct using `tprodCoeff` -/
@[elab_as_elim]
protected theorem induction_on' {motive : (⨂[R] i, s i) → Prop} (z : ⨂[R] i, s i)
    (tprodCoeff : ∀ (r : R) (f : Π i, s i), motive (tprodCoeff R r f))
    (add : ∀ x y, motive x → motive y → motive (x + y)) :
    motive z := by
  have C0 : motive 0 := by
    have h₁ := tprodCoeff 0 0
    rwa [zero_tprodCoeff] at h₁
  /-
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    motive : (PiTensorProduct R fun i => s i) → Prop
    z : PiTensorProduct R fun i => s i
    tprodCoeff : ∀ (r : R) (f : (i : ι) → s i), motive (PiTensorProduct.tprodCoeff …
    add : ∀ (x y : PiTensorProduct R fun i => s i), motive x → motive y → motive ( …
    C0 : motive 0
    ⊢ motive z
  -/
  refine AddCon.induction_on z fun x ↦ FreeAddMonoid.recOn x C0 ?_
  /-
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    motive : (PiTensorProduct R fun i => s i) → Prop
    z : PiTensorProduct R fun i => s i
    tprodCoeff : ∀ (r : R) (f : (i : ι) → s i), motive (PiTensorProduct.tprodCoeff …
    add : ∀ (x y : PiTensorProduct R fun i => s i), motive x → motive y → motive ( …
    C0 : motive 0
    x : FreeAddMonoid (Prod R ((i : ι) → (fun i => s i) i))
    ⊢ ∀ (x : Prod R ((i : ι) → (fun i => s i) i)) (xs : FreeAddMonoid (Prod R ((i  …
  -/
  simp_rw [AddCon.coe_add]
  /-
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    motive : (PiTensorProduct R fun i => s i) → Prop
    z : PiTensorProduct R fun i => s i
    tprodCoeff : ∀ (r : R) (f : (i : ι) → s i), motive (PiTensorProduct.tprodCoeff …
    add : ∀ (x y : PiTensorProduct R fun i => s i), motive x → motive y → motive ( …
    C0 : motive 0
    x : FreeAddMonoid (Prod R ((i : ι) → (fun i => s i) i))
    ⊢ ∀ (x : Prod R ((i : ι) → s i)) (xs : FreeAddMonoid (Prod R ((i : ι) → s i))) …
  -/
  refine fun f y ih ↦ add _ _ ?_ ih
  /-
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    motive : (PiTensorProduct R fun i => s i) → Prop
    z : PiTensorProduct R fun i => s i
    tprodCoeff : ∀ (r : R) (f : (i : ι) → s i), motive (PiTensorProduct.tprodCoeff …
    add : ∀ (x y : PiTensorProduct R fun i => s i), motive x → motive y → motive ( …
    C0 : motive 0
    x : FreeAddMonoid (Prod R ((i : ι) → (fun i => s i) i))
    f : Prod R ((i : ι) → s i)
    y : FreeAddMonoid (Prod R ((i : ι) → s i))
    ih : motive ↑y
    ⊢ motive ↑(FreeAddMonoid.of f)
  -/
  convert tprodCoeff f.1 f.2
  /-
    🎉 no goals
  -/


instance hasSMul' : SMul R₁ (⨂[R] i, s i) :=
  ⟨fun r ↦
    liftAddHom (fun f : R × Π i, s i ↦ tprodCoeff R (r • f.1) f.2)
                          /-
                            ι : Type u_1
                            ι₂ : Type u_2
                            ι₃ : Type u_3
                            R : Type u_4
                            inst✝¹³ : CommSemiring R
                            R₁ : Type u_5
                            R₂ : Type u_6
                            s : ι → Type u_7
                            inst✝¹² : (i : ι) → AddCommMonoid (s i)
                            inst✝¹¹ : (i : ι) → Module R (s i)
                            M : Type u_8
                            inst✝¹⁰ : AddCommMonoid M
                            inst✝⁹ : Module R M
                            E : Type u_9
                            inst✝⁸ : AddCommMonoid E
                            inst✝⁷ : Module R E
                            F : Type u_10
                            inst✝⁶ : AddCommMonoid F
                            inst✝⁵ : Monoid R₁
                            inst✝⁴ : DistribMulAction R₁ R
                            inst✝³ : SMulCommClass R₁ R R
                            inst✝² : Monoid R₂
                            inst✝¹ : DistribMulAction R₂ R
                            inst✝ : SMulCommClass R₂ R R
                            r : R₁
                            r' : R
                            f : (i : ι) → s i
                            i : ι
                            hf : Eq (f i) 0
                            ⊢ Eq ((fun f => PiTensorProduct.tprodCoeff R (HSMul.hSMul r f.1) f.2) { fst := …
                          -/
      (fun r' f i hf ↦ by simp_rw [zero_tprodCoeff' _ f i hf])
                          /-
                            🎉 no goals
                          -/
                  /-
                    ι : Type u_1
                    ι₂ : Type u_2
                    ι₃ : Type u_3
                    R : Type u_4
                    inst✝¹³ : CommSemiring R
                    R₁ : Type u_5
                    R₂ : Type u_6
                    s : ι → Type u_7
                    inst✝¹² : (i : ι) → AddCommMonoid (s i)
                    inst✝¹¹ : (i : ι) → Module R (s i)
                    M : Type u_8
                    inst✝¹⁰ : AddCommMonoid M
                    inst✝⁹ : Module R M
                    E : Type u_9
                    inst✝⁸ : AddCommMonoid E
                    inst✝⁷ : Module R E
                    F : Type u_10
                    inst✝⁶ : AddCommMonoid F
                    inst✝⁵ : Monoid R₁
                    inst✝⁴ : DistribMulAction R₁ R
                    inst✝³ : SMulCommClass R₁ R R
                    inst✝² : Monoid R₂
                    inst✝¹ : DistribMulAction R₂ R
                    inst✝ : SMulCommClass R₂ R R
                    r : R₁
                    f : (i : ι) → s i
                    ⊢ Eq ((fun f => PiTensorProduct.tprodCoeff R (HSMul.hSMul r f.1) f.2) { fst := …
                  -/
                  /-
                    🎉 no goals
                  -/
      (fun f ↦ by simp [zero_tprodCoeff]) (fun r' f i m₁ m₂ ↦ by simp [add_tprodCoeff])
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                         /-
                           ι : Type u_1
                           ι₂ : Type u_2
                           ι₃ : Type u_3
                           R : Type u_4
                           inst✝¹³ : CommSemiring R
                           R₁ : Type u_5
                           R₂ : Type u_6
                           s : ι → Type u_7
                           inst✝¹² : (i : ι) → AddCommMonoid (s i)
                           inst✝¹¹ : (i : ι) → Module R (s i)
                           M : Type u_8
                           inst✝¹⁰ : AddCommMonoid M
                           inst✝⁹ : Module R M
                           E : Type u_9
                           inst✝⁸ : AddCommMonoid E
                           inst✝⁷ : Module R E
                           F : Type u_10
                           inst✝⁶ : AddCommMonoid F
                           inst✝⁵ : Monoid R₁
                           inst✝⁴ : DistribMulAction R₁ R
                           inst✝³ : SMulCommClass R₁ R R
                           inst✝² : Monoid R₂
                           inst✝¹ : DistribMulAction R₂ R
                           inst✝ : SMulCommClass R₂ R R
                           r : R₁
                           r' r'' : R
                           f : (i : ι) → s i
                           ⊢ Eq (HAdd.hAdd ((fun f => PiTensorProduct.tprodCoeff R (HSMul.hSMul r f.1) f. …
                         -/
      (fun r' r'' f ↦ by simp [add_tprodCoeff', mul_add]) fun z f i r' ↦ by
                         /-
                           🎉 no goals
                         -/
      /-
        ι : Type u_1
        ι₂ : Type u_2
        ι₃ : Type u_3
        R : Type u_4
        inst✝¹⁴ : CommSemiring R
        R₁ : Type u_5
        R₂ : Type u_6
        s : ι → Type u_7
        inst✝¹³ : (i : ι) → AddCommMonoid (s i)
        inst✝¹² : (i : ι) → Module R (s i)
        M : Type u_8
        inst✝¹¹ : AddCommMonoid M
        inst✝¹⁰ : Module R M
        E : Type u_9
        inst✝⁹ : AddCommMonoid E
        inst✝⁸ : Module R E
        F : Type u_10
        inst✝⁷ : AddCommMonoid F
        inst✝⁶ : Monoid R₁
        inst✝⁵ : DistribMulAction R₁ R
        inst✝⁴ : SMulCommClass R₁ R R
        inst✝³ : Monoid R₂
        inst✝² : DistribMulAction R₂ R
        inst✝¹ : SMulCommClass R₂ R R
        r : R₁
        inst✝ : DecidableEq ι
        z : R
        f : (i : ι) → s i
        i : ι
        r' : R
        ⊢ Eq ((fun f => PiTensorProduct.tprodCoeff R (HSMul.hSMul r f.1) f.2) { fst := …
      -/
      simp [smul_tprodCoeff, mul_smul_comm]⟩
      /-
        🎉 no goals
      -/


instance : SMul R (⨂[R] i, s i) :=
  PiTensorProduct.hasSMul'


theorem smul_tprodCoeff' (r : R₁) (z : R) (f : Π i, s i) :
    r • tprodCoeff R z f = tprodCoeff R (r • z) f := rfl


protected theorem smul_add (r : R₁) (x y : ⨂[R] i, s i) : r • (x + y) = r • x + r • y :=
  AddMonoidHom.map_add _ _ _


instance distribMulAction' : DistribMulAction R₁ (⨂[R] i, s i) where
  smul := (· • ·)
  smul_add _ _ _ := AddMonoidHom.map_add _ _ _
  mul_smul r r' x :=
                                                      /-
                                                        ι : Type u_1
                                                        ι₂ : Type u_2
                                                        ι₃ : Type u_3
                                                        R : Type u_4
                                                        inst✝¹³ : CommSemiring R
                                                        R₁ : Type u_5
                                                        R₂ : Type u_6
                                                        s : ι → Type u_7
                                                        inst✝¹² : (i : ι) → AddCommMonoid (s i)
                                                        inst✝¹¹ : (i : ι) → Module R (s i)
                                                        M : Type u_8
                                                        inst✝¹⁰ : AddCommMonoid M
                                                        inst✝⁹ : Module R M
                                                        E : Type u_9
                                                        inst✝⁸ : AddCommMonoid E
                                                        inst✝⁷ : Module R E
                                                        F : Type u_10
                                                        inst✝⁶ : AddCommMonoid F
                                                        inst✝⁵ : Monoid R₁
                                                        inst✝⁴ : DistribMulAction R₁ R
                                                        inst✝³ : SMulCommClass R₁ R R
                                                        inst✝² : Monoid R₂
                                                        inst✝¹ : DistribMulAction R₂ R
                                                        inst✝ : SMulCommClass R₂ R R
                                                        r r' : R₁
                                                        x : PiTensorProduct R fun i => s i
                                                        r'' : R
                                                        f : (i : ι) → s i
                                                        ⊢ Eq (HSMul.hSMul (HMul.hMul r r') (PiTensorProduct.tprodCoeff R r'' f)) (HSMu …
                                                      -/
    PiTensorProduct.induction_on' x (fun {r'' f} ↦ by simp [smul_tprodCoeff', smul_smul])
                                                      /-
                                                        🎉 no goals
                                                      -/
                                                    /-
                                                      ι : Type u_1
                                                      ι₂ : Type u_2
                                                      ι₃ : Type u_3
                                                      R : Type u_4
                                                      inst✝¹³ : CommSemiring R
                                                      R₁ : Type u_5
                                                      R₂ : Type u_6
                                                      s : ι → Type u_7
                                                      inst✝¹² : (i : ι) → AddCommMonoid (s i)
                                                      inst✝¹¹ : (i : ι) → Module R (s i)
                                                      M : Type u_8
                                                      inst✝¹⁰ : AddCommMonoid M
                                                      inst✝⁹ : Module R M
                                                      E : Type u_9
                                                      inst✝⁸ : AddCommMonoid E
                                                      inst✝⁷ : Module R E
                                                      F : Type u_10
                                                      inst✝⁶ : AddCommMonoid F
                                                      inst✝⁵ : Monoid R₁
                                                      inst✝⁴ : DistribMulAction R₁ R
                                                      inst✝³ : SMulCommClass R₁ R R
                                                      inst✝² : Monoid R₂
                                                      inst✝¹ : DistribMulAction R₂ R
                                                      inst✝ : SMulCommClass R₂ R R
                                                      x : PiTensorProduct R fun i => s i
                                                      r : R
                                                      f : (i : ι) → s i
                                                      ⊢ Eq (HSMul.hSMul 1 (PiTensorProduct.tprodCoeff R r f)) (PiTensorProduct.tprod …
                                                    -/
                             /-
                               ι : Type u_1
                               ι₂ : Type u_2
                               ι₃ : Type u_3
                               R : Type u_4
                               inst✝¹³ : CommSemiring R
                               R₁ : Type u_5
                               R₂ : Type u_6
                               s : ι → Type u_7
                               inst✝¹² : (i : ι) → AddCommMonoid (s i)
                               inst✝¹¹ : (i : ι) → Module R (s i)
                               M : Type u_8
                               inst✝¹⁰ : AddCommMonoid M
                               inst✝⁹ : Module R M
                               E : Type u_9
                               inst✝⁸ : AddCommMonoid E
                               inst✝⁷ : Module R E
                               F : Type u_10
                               inst✝⁶ : AddCommMonoid F
                               inst✝⁵ : Monoid R₁
                               inst✝⁴ : DistribMulAction R₁ R
                               inst✝³ : SMulCommClass R₁ R R
                               inst✝² : Monoid R₂
                               inst✝¹ : DistribMulAction R₂ R
                               inst✝ : SMulCommClass R₂ R R
                               r r' : R₁
                               x✝ x y : PiTensorProduct R fun i => s i
                               ihx : Eq (HSMul.hSMul (HMul.hMul r r') x) (HSMul.hSMul r (HSMul.hSMul r' x))
                               ihy : Eq (HSMul.hSMul (HMul.hMul r r') y) (HSMul.hSMul r (HSMul.hSMul r' y))
                               ⊢ Eq (HSMul.hSMul (HMul.hMul r r') (HAdd.hAdd x y)) (HSMul.hSMul r (HSMul.hSMu …
                             -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                             /-
                               ι : Type u_1
                               ι₂ : Type u_2
                               ι₃ : Type u_3
                               R : Type u_4
                               inst✝¹³ : CommSemiring R
                               R₁ : Type u_5
                               R₂ : Type u_6
                               s : ι → Type u_7
                               inst✝¹² : (i : ι) → AddCommMonoid (s i)
                               inst✝¹¹ : (i : ι) → Module R (s i)
                               M : Type u_8
                               inst✝¹⁰ : AddCommMonoid M
                               inst✝⁹ : Module R M
                               E : Type u_9
                               inst✝⁸ : AddCommMonoid E
                               inst✝⁷ : Module R E
                               F : Type u_10
                               inst✝⁶ : AddCommMonoid F
                               inst✝⁵ : Monoid R₁
                               inst✝⁴ : DistribMulAction R₁ R
                               inst✝³ : SMulCommClass R₁ R R
                               inst✝² : Monoid R₂
                               inst✝¹ : DistribMulAction R₂ R
                               inst✝ : SMulCommClass R₂ R R
                               x z y : PiTensorProduct R fun i => s i
                               ihz : Eq (HSMul.hSMul 1 z) z
                               ihy : Eq (HSMul.hSMul 1 y) y
                               ⊢ Eq (HSMul.hSMul 1 (HAdd.hAdd z y)) (HAdd.hAdd z y)
                             -/
      fun {x y} ihx ihy ↦ by simp_rw [PiTensorProduct.smul_add, ihx, ihy]
                             /-
                               🎉 no goals
                             -/
                             /-
                               🎉 no goals
                             -/
  one_smul x :=
    PiTensorProduct.induction_on' x (fun {r f} ↦ by rw [smul_tprodCoeff', one_smul])
      fun {z y} ihz ihy ↦ by simp_rw [PiTensorProduct.smul_add, ihz, ihy]
  smul_zero _ := AddMonoidHom.map_zero _


instance smulCommClass' [SMulCommClass R₁ R₂ R] : SMulCommClass R₁ R₂ (⨂[R] i, s i) :=
  ⟨fun {r' r''} x ↦
                                                      /-
                                                        ι : Type u_1
                                                        ι₂ : Type u_2
                                                        ι₃ : Type u_3
                                                        R : Type u_4
                                                        inst✝¹⁴ : CommSemiring R
                                                        R₁ : Type u_5
                                                        R₂ : Type u_6
                                                        s : ι → Type u_7
                                                        inst✝¹³ : (i : ι) → AddCommMonoid (s i)
                                                        inst✝¹² : (i : ι) → Module R (s i)
                                                        M : Type u_8
                                                        inst✝¹¹ : AddCommMonoid M
                                                        inst✝¹⁰ : Module R M
                                                        E : Type u_9
                                                        inst✝⁹ : AddCommMonoid E
                                                        inst✝⁸ : Module R E
                                                        F : Type u_10
                                                        inst✝⁷ : AddCommMonoid F
                                                        inst✝⁶ : Monoid R₁
                                                        inst✝⁵ : DistribMulAction R₁ R
                                                        inst✝⁴ : SMulCommClass R₁ R R
                                                        inst✝³ : Monoid R₂
                                                        inst✝² : DistribMulAction R₂ R
                                                        inst✝¹ : SMulCommClass R₂ R R
                                                        inst✝ : SMulCommClass R₁ R₂ R
                                                        r' : R₁
                                                        r'' : R₂
                                                        x : PiTensorProduct R fun i => s i
                                                        xr : R
                                                        xf : (i : ι) → s i
                                                        ⊢ Eq (HSMul.hSMul r' (HSMul.hSMul r'' (PiTensorProduct.tprodCoeff R xr xf))) ( …
                                                      -/
    PiTensorProduct.induction_on' x (fun {xr xf} ↦ by simp only [smul_tprodCoeff', smul_comm])
                                                      /-
                                                        🎉 no goals
                                                      -/
                             /-
                               ι : Type u_1
                               ι₂ : Type u_2
                               ι₃ : Type u_3
                               R : Type u_4
                               inst✝¹⁴ : CommSemiring R
                               R₁ : Type u_5
                               R₂ : Type u_6
                               s : ι → Type u_7
                               inst✝¹³ : (i : ι) → AddCommMonoid (s i)
                               inst✝¹² : (i : ι) → Module R (s i)
                               M : Type u_8
                               inst✝¹¹ : AddCommMonoid M
                               inst✝¹⁰ : Module R M
                               E : Type u_9
                               inst✝⁹ : AddCommMonoid E
                               inst✝⁸ : Module R E
                               F : Type u_10
                               inst✝⁷ : AddCommMonoid F
                               inst✝⁶ : Monoid R₁
                               inst✝⁵ : DistribMulAction R₁ R
                               inst✝⁴ : SMulCommClass R₁ R R
                               inst✝³ : Monoid R₂
                               inst✝² : DistribMulAction R₂ R
                               inst✝¹ : SMulCommClass R₂ R R
                               inst✝ : SMulCommClass R₁ R₂ R
                               r' : R₁
                               r'' : R₂
                               x z y : PiTensorProduct R fun i => s i
                               ihz : Eq (HSMul.hSMul r' (HSMul.hSMul r'' z)) (HSMul.hSMul r'' (HSMul.hSMul r' …
                               ihy : Eq (HSMul.hSMul r' (HSMul.hSMul r'' y)) (HSMul.hSMul r'' (HSMul.hSMul r' …
                               ⊢ Eq (HSMul.hSMul r' (HSMul.hSMul r'' (HAdd.hAdd z y))) (HSMul.hSMul r'' (HSMu …
                             -/
      fun {z y} ihz ihy ↦ by simp_rw [PiTensorProduct.smul_add, ihz, ihy]⟩
                             /-
                               🎉 no goals
                             -/


instance isScalarTower' [SMul R₁ R₂] [IsScalarTower R₁ R₂ R] :
    IsScalarTower R₁ R₂ (⨂[R] i, s i) :=
  ⟨fun {r' r''} x ↦
                                                      /-
                                                        ι : Type u_1
                                                        ι₂ : Type u_2
                                                        ι₃ : Type u_3
                                                        R : Type u_4
                                                        inst✝¹⁵ : CommSemiring R
                                                        R₁ : Type u_5
                                                        R₂ : Type u_6
                                                        s : ι → Type u_7
                                                        inst✝¹⁴ : (i : ι) → AddCommMonoid (s i)
                                                        inst✝¹³ : (i : ι) → Module R (s i)
                                                        M : Type u_8
                                                        inst✝¹² : AddCommMonoid M
                                                        inst✝¹¹ : Module R M
                                                        E : Type u_9
                                                        inst✝¹⁰ : AddCommMonoid E
                                                        inst✝⁹ : Module R E
                                                        F : Type u_10
                                                        inst✝⁸ : AddCommMonoid F
                                                        inst✝⁷ : Monoid R₁
                                                        inst✝⁶ : DistribMulAction R₁ R
                                                        inst✝⁵ : SMulCommClass R₁ R R
                                                        inst✝⁴ : Monoid R₂
                                                        inst✝³ : DistribMulAction R₂ R
                                                        inst✝² : SMulCommClass R₂ R R
                                                        inst✝¹ : SMul R₁ R₂
                                                        inst✝ : IsScalarTower R₁ R₂ R
                                                        r' : R₁
                                                        r'' : R₂
                                                        x : PiTensorProduct R fun i => s i
                                                        xr : R
                                                        xf : (i : ι) → s i
                                                        ⊢ Eq (HSMul.hSMul (HSMul.hSMul r' r'') (PiTensorProduct.tprodCoeff R xr xf)) ( …
                                                      -/
    PiTensorProduct.induction_on' x (fun {xr xf} ↦ by simp only [smul_tprodCoeff', smul_assoc])
                                                      /-
                                                        🎉 no goals
                                                      -/
                             /-
                               ι : Type u_1
                               ι₂ : Type u_2
                               ι₃ : Type u_3
                               R : Type u_4
                               inst✝¹⁵ : CommSemiring R
                               R₁ : Type u_5
                               R₂ : Type u_6
                               s : ι → Type u_7
                               inst✝¹⁴ : (i : ι) → AddCommMonoid (s i)
                               inst✝¹³ : (i : ι) → Module R (s i)
                               M : Type u_8
                               inst✝¹² : AddCommMonoid M
                               inst✝¹¹ : Module R M
                               E : Type u_9
                               inst✝¹⁰ : AddCommMonoid E
                               inst✝⁹ : Module R E
                               F : Type u_10
                               inst✝⁸ : AddCommMonoid F
                               inst✝⁷ : Monoid R₁
                               inst✝⁶ : DistribMulAction R₁ R
                               inst✝⁵ : SMulCommClass R₁ R R
                               inst✝⁴ : Monoid R₂
                               inst✝³ : DistribMulAction R₂ R
                               inst✝² : SMulCommClass R₂ R R
                               inst✝¹ : SMul R₁ R₂
                               inst✝ : IsScalarTower R₁ R₂ R
                               r' : R₁
                               r'' : R₂
                               x z y : PiTensorProduct R fun i => s i
                               ihz : Eq (HSMul.hSMul (HSMul.hSMul r' r'') z) (HSMul.hSMul r' (HSMul.hSMul r'' …
                               ihy : Eq (HSMul.hSMul (HSMul.hSMul r' r'') y) (HSMul.hSMul r' (HSMul.hSMul r'' …
                               ⊢ Eq (HSMul.hSMul (HSMul.hSMul r' r'') (HAdd.hAdd z y)) (HSMul.hSMul r' (HSMul …
                             -/
      fun {z y} ihz ihy ↦ by simp_rw [PiTensorProduct.smul_add, ihz, ihy]⟩
                             /-
                               🎉 no goals
                             -/


instance module' [Semiring R₁] [Module R₁ R] [SMulCommClass R₁ R R] : Module R₁ (⨂[R] i, s i) :=
  { PiTensorProduct.distribMulAction' with
    add_smul := fun r r' x ↦
      PiTensorProduct.induction_on' x
                        /-
                          ι : Type u_1
                          ι₂ : Type u_2
                          ι₃ : Type u_3
                          R : Type u_4
                          inst✝¹⁰ : CommSemiring R
                          R₁ : Type u_5
                          R₂ : Type u_6
                          s : ι → Type u_7
                          inst✝⁹ : (i : ι) → AddCommMonoid (s i)
                          inst✝⁸ : (i : ι) → Module R (s i)
                          M : Type u_8
                          inst✝⁷ : AddCommMonoid M
                          inst✝⁶ : Module R M
                          E : Type u_9
                          inst✝⁵ : AddCommMonoid E
                          inst✝⁴ : Module R E
                          F : Type u_10
                          inst✝³ : AddCommMonoid F
                          inst✝² : Semiring R₁
                          inst✝¹ : Module R₁ R
                          inst✝ : SMulCommClass R₁ R R
                          r✝ r' : R₁
                          x : PiTensorProduct R fun i => s i
                          r : R
                          f : (i : ι) → s i
                          ⊢ Eq (HSMul.hSMul (HAdd.hAdd r✝ r') (PiTensorProduct.tprodCoeff R r f)) (HAdd. …
                        -/
        (fun {r f} ↦ by simp_rw [smul_tprodCoeff', add_smul, add_tprodCoeff'])
                        /-
                          🎉 no goals
                        -/
                               /-
                                 ι : Type u_1
                                 ι₂ : Type u_2
                                 ι₃ : Type u_3
                                 R : Type u_4
                                 inst✝¹⁰ : CommSemiring R
                                 R₁ : Type u_5
                                 R₂ : Type u_6
                                 s : ι → Type u_7
                                 inst✝⁹ : (i : ι) → AddCommMonoid (s i)
                                 inst✝⁸ : (i : ι) → Module R (s i)
                                 M : Type u_8
                                 inst✝⁷ : AddCommMonoid M
                                 inst✝⁶ : Module R M
                                 E : Type u_9
                                 inst✝⁵ : AddCommMonoid E
                                 inst✝⁴ : Module R E
                                 F : Type u_10
                                 inst✝³ : AddCommMonoid F
                                 inst✝² : Semiring R₁
                                 inst✝¹ : Module R₁ R
                                 inst✝ : SMulCommClass R₁ R R
                                 r r' : R₁
                                 x✝ x y : PiTensorProduct R fun i => s i
                                 ihx : Eq (HSMul.hSMul (HAdd.hAdd r r') x) (HAdd.hAdd (HSMul.hSMul r x) (HSMul. …
                                 ihy : Eq (HSMul.hSMul (HAdd.hAdd r r') y) (HAdd.hAdd (HSMul.hSMul r y) (HSMul. …
                                 ⊢ Eq (HSMul.hSMul (HAdd.hAdd r r') (HAdd.hAdd x y)) (HAdd.hAdd (HSMul.hSMul r  …
                               -/
        fun {x y} ihx ihy ↦ by simp_rw [PiTensorProduct.smul_add, ihx, ihy, add_add_add_comm]
                               /-
                                 🎉 no goals
                               -/
    zero_smul := fun x ↦
      PiTensorProduct.induction_on' x
                        /-
                          ι : Type u_1
                          ι₂ : Type u_2
                          ι₃ : Type u_3
                          R : Type u_4
                          inst✝¹⁰ : CommSemiring R
                          R₁ : Type u_5
                          R₂ : Type u_6
                          s : ι → Type u_7
                          inst✝⁹ : (i : ι) → AddCommMonoid (s i)
                          inst✝⁸ : (i : ι) → Module R (s i)
                          M : Type u_8
                          inst✝⁷ : AddCommMonoid M
                          inst✝⁶ : Module R M
                          E : Type u_9
                          inst✝⁵ : AddCommMonoid E
                          inst✝⁴ : Module R E
                          F : Type u_10
                          inst✝³ : AddCommMonoid F
                          inst✝² : Semiring R₁
                          inst✝¹ : Module R₁ R
                          inst✝ : SMulCommClass R₁ R R
                          x : PiTensorProduct R fun i => s i
                          r : R
                          f : (i : ι) → s i
                          ⊢ Eq (HSMul.hSMul 0 (PiTensorProduct.tprodCoeff R r f)) 0
                        -/
        (fun {r f} ↦ by simp_rw [smul_tprodCoeff', zero_smul, zero_tprodCoeff])
                        /-
                          🎉 no goals
                        -/
                               /-
                                 ι : Type u_1
                                 ι₂ : Type u_2
                                 ι₃ : Type u_3
                                 R : Type u_4
                                 inst✝¹⁰ : CommSemiring R
                                 R₁ : Type u_5
                                 R₂ : Type u_6
                                 s : ι → Type u_7
                                 inst✝⁹ : (i : ι) → AddCommMonoid (s i)
                                 inst✝⁸ : (i : ι) → Module R (s i)
                                 M : Type u_8
                                 inst✝⁷ : AddCommMonoid M
                                 inst✝⁶ : Module R M
                                 E : Type u_9
                                 inst✝⁵ : AddCommMonoid E
                                 inst✝⁴ : Module R E
                                 F : Type u_10
                                 inst✝³ : AddCommMonoid F
                                 inst✝² : Semiring R₁
                                 inst✝¹ : Module R₁ R
                                 inst✝ : SMulCommClass R₁ R R
                                 x✝ x y : PiTensorProduct R fun i => s i
                                 ihx : Eq (HSMul.hSMul 0 x) 0
                                 ihy : Eq (HSMul.hSMul 0 y) 0
                                 ⊢ Eq (HSMul.hSMul 0 (HAdd.hAdd x y)) 0
                               -/
        fun {x y} ihx ihy ↦ by simp_rw [PiTensorProduct.smul_add, ihx, ihy, add_zero] }
                               /-
                                 🎉 no goals
                               -/

-- shortcut instances

instance : Module R (⨂[R] i, s i) :=
  PiTensorProduct.module'


instance : SMulCommClass R R (⨂[R] i, s i) :=
  PiTensorProduct.smulCommClass'


instance : IsScalarTower R R (⨂[R] i, s i) :=
  PiTensorProduct.isScalarTower'


/-- The canonical `MultilinearMap R s (⨂[R] i, s i)`.

`tprod R fun i => f i` has notation `⨂ₜ[R] i, f i`. -/
def tprod : MultilinearMap R s (⨂[R] i, s i) where
  toFun := tprodCoeff R 1
  map_update_add' {_ f} i x y := (add_tprodCoeff (1 : R) f i x y).symm
  map_update_smul' {_ f} i r x := by
    /-
      ι : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      R : Type u_4
      inst✝⁷ : CommSemiring R
      R₁ : Type u_5
      R₂ : Type u_6
      s : ι → Type u_7
      inst✝⁶ : (i : ι) → AddCommMonoid (s i)
      inst✝⁵ : (i : ι) → Module R (s i)
      M : Type u_8
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      E : Type u_9
      inst✝² : AddCommMonoid E
      inst✝¹ : Module R E
      F : Type u_10
      inst✝ : AddCommMonoid F
      x✝ : DecidableEq ι
      f : (i : ι) → s i
      i : ι
      r : R
      x : s i
      ⊢ Eq (PiTensorProduct.tprodCoeff R 1 (Function.update f i (HSMul.hSMul r x)))  …
    -/
    rw [smul_tprodCoeff', ← smul_tprodCoeff (1 : R) _ i, update_idem, update_self]
    /-
      🎉 no goals
    -/


unsuppress_compilation in
@[inherit_doc tprod]
notation3:100 "⨂ₜ["R"] "(...)", "r:(scoped f => tprod R f) => r


theorem tprod_eq_tprodCoeff_one :
    ⇑(tprod R : MultilinearMap R s (⨂[R] i, s i)) = tprodCoeff R 1 := rfl


@[simp]
theorem tprodCoeff_eq_smul_tprod (z : R) (f : Π i, s i) : tprodCoeff R z f = z • tprod R f := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    z : R
    f : (i : ι) → s i
    ⊢ Eq (PiTensorProduct.tprodCoeff R z f) (HSMul.hSMul z ((PiTensorProduct.tprod …
  -/
  have : z = z • (1 : R) := by simp only [mul_one, Algebra.id.smul_eq_mul]
  /-
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    z : R
    f : (i : ι) → s i
    this : Eq z (HSMul.hSMul z 1)
    ⊢ Eq (PiTensorProduct.tprodCoeff R z f) (HSMul.hSMul z ((PiTensorProduct.tprod …
  -/
  conv_lhs => rw [this]
  /-
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    z : R
    f : (i : ι) → s i
    this : Eq z (HSMul.hSMul z 1)
    ⊢ Eq (PiTensorProduct.tprodCoeff R (HSMul.hSMul z 1) f) (HSMul.hSMul z ((PiTen …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The image of an element `p` of `FreeAddMonoid (R × Π i, s i)` in the `PiTensorProduct` is
equal to the sum of `a • ⨂ₜ[R] i, m i` over all the entries `(a, m)` of `p`.
-/
lemma _root_.FreeAddMonoid.toPiTensorProduct (p : FreeAddMonoid (R × Π i, s i)) :
    AddCon.toQuotient (c := addConGen (PiTensorProduct.Eqv R s)) p =
    List.sum (List.map (fun x ↦ x.1 • ⨂ₜ[R] i, x.2 i) p) := by
  match p with
  | [] => rw [List.map_nil, List.sum_nil]; rfl
  | x :: ps => rw [List.map_cons, List.sum_cons, ← List.singleton_append, ← toPiTensorProduct ps,
                 ← tprodCoeff_eq_smul_tprod]; rfl


/-- The set of lifts of an element `x` of `⨂[R] i, s i` in `FreeAddMonoid (R × Π i, s i)`. -/
def lifts (x : ⨂[R] i, s i) : Set (FreeAddMonoid (R × Π i, s i)) :=
  {p | AddCon.toQuotient (c := addConGen (PiTensorProduct.Eqv R s)) p = x}


/-- An element `p` of `FreeAddMonoid (R × Π i, s i)` lifts an element `x` of `⨂[R] i, s i`
if and only if `x` is equal to the sum of `a • ⨂ₜ[R] i, m i` over all the entries
`(a, m)` of `p`.
-/
lemma mem_lifts_iff (x : ⨂[R] i, s i) (p : FreeAddMonoid (R × Π i, s i)) :
    p ∈ lifts x ↔ List.sum (List.map (fun x ↦ x.1 • ⨂ₜ[R] i, x.2 i) p) = x := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    x : PiTensorProduct R fun i => s i
    p : FreeAddMonoid (Prod R ((i : ι) → s i))
    ⊢ Iff (Membership.mem x.lifts p) (Eq (List.map (fun x => HSMul.hSMul x.1 ((PiT …
  -/
  simp only [lifts, Set.mem_setOf_eq, FreeAddMonoid.toPiTensorProduct]
  /-
    🎉 no goals
  -/


/-- Every element of `⨂[R] i, s i` has a lift in `FreeAddMonoid (R × Π i, s i)`.
-/
lemma nonempty_lifts (x : ⨂[R] i, s i) : Set.Nonempty (lifts x) := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    x : PiTensorProduct R fun i => s i
    ⊢ x.lifts.Nonempty
  -/
  existsi @Quotient.out _ (addConGen (PiTensorProduct.Eqv R s)).toSetoid x
  /-
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    x : PiTensorProduct R fun i => s i
    ⊢ Membership.mem x.lifts (Quotient.out x)
  -/
  simp only [lifts, Set.mem_setOf_eq]
  /-
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    x : PiTensorProduct R fun i => s i
    ⊢ Eq (↑(Quotient.out x)) x
  -/
  rw [← AddCon.quot_mk_eq_coe]
  /-
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    x : PiTensorProduct R fun i => s i
    ⊢ Eq (Quot.mk (⇑(addConGen (PiTensorProduct.Eqv R s))) (Quotient.out x)) x
  -/
  erw [Quot.out_eq]
  /-
    🎉 no goals
  -/


/-- The empty list lifts the element `0` of `⨂[R] i, s i`.
-/
lemma lifts_zero : 0 ∈ lifts (0 : ⨂[R] i, s i) := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    ⊢ Membership.mem (PiTensorProduct.lifts 0) 0
  -/
  rw [mem_lifts_iff]; erw [List.map_nil]; rw [List.sum_nil]
                                          /-
                                            🎉 no goals
                                          -/


/-- If elements `p,q` of `FreeAddMonoid (R × Π i, s i)` lift elements `x,y` of `⨂[R] i, s i`
respectively, then `p + q` lifts `x + y`.
-/
lemma lifts_add {x y : ⨂[R] i, s i} {p q : FreeAddMonoid (R × Π i, s i)}
    (hp : p ∈ lifts x) (hq : q ∈ lifts y) : p + q ∈ lifts (x + y) := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    x y : PiTensorProduct R fun i => s i
    p q : FreeAddMonoid (Prod R ((i : ι) → s i))
    hp : Membership.mem x.lifts p
    hq : Membership.mem y.lifts q
    ⊢ Membership.mem (HAdd.hAdd x y).lifts (HAdd.hAdd p q)
  -/
  simp only [lifts, Set.mem_setOf_eq, AddCon.coe_add]
  /-
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    x y : PiTensorProduct R fun i => s i
    p q : FreeAddMonoid (Prod R ((i : ι) → s i))
    hp : Membership.mem x.lifts p
    hq : Membership.mem y.lifts q
    ⊢ Eq (HAdd.hAdd ↑p ↑q) (HAdd.hAdd x y)
  -/
  rw [hp, hq]
  /-
    🎉 no goals
  -/


/-- If an element `p` of `FreeAddMonoid (R × Π i, s i)` lifts an element `x` of `⨂[R] i, s i`,
and if `a` is an element of `R`, then the list obtained by multiplying the first entry of each
element of `p` by `a` lifts `a • x`.
-/
lemma lifts_smul {x : ⨂[R] i, s i} {p : FreeAddMonoid (R × Π i, s i)} (h : p ∈ lifts x) (a : R) :
    List.map (fun (y : R × Π i, s i) ↦ (a * y.1, y.2)) p ∈ lifts (a • x) := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    x : PiTensorProduct R fun i => s i
    p : FreeAddMonoid (Prod R ((i : ι) → s i))
    h : Membership.mem x.lifts p
    a : R
    ⊢ Membership.mem (HSMul.hSMul a x).lifts (List.map (fun y => { fst := HMul.hMu …
  -/
  rw [mem_lifts_iff] at h ⊢
  /-
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    x : PiTensorProduct R fun i => s i
    p : FreeAddMonoid (Prod R ((i : ι) → s i))
    h : Eq (List.map (fun x => HSMul.hSMul x.1 ((PiTensorProduct.tprod R) fun i => …
    a : R
    ⊢ Eq (List.map (fun x => HSMul.hSMul x.1 ((PiTensorProduct.tprod R) fun i => x …
  -/
  rw [← List.comp_map, ← h, List.smul_sum, ← List.comp_map]
  /-
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    x : PiTensorProduct R fun i => s i
    p : FreeAddMonoid (Prod R ((i : ι) → s i))
    h : Eq (List.map (fun x => HSMul.hSMul x.1 ((PiTensorProduct.tprod R) fun i => …
    a : R
    ⊢ Eq (List.map (Function.comp (fun x => HSMul.hSMul x.1 ((PiTensorProduct.tpro …
  -/
  congr 2
  /-
    case e_a.e_f
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    x : PiTensorProduct R fun i => s i
    p : FreeAddMonoid (Prod R ((i : ι) → s i))
    h : Eq (List.map (fun x => HSMul.hSMul x.1 ((PiTensorProduct.tprod R) fun i => …
    a : R
    ⊢ Eq (Function.comp (fun x => HSMul.hSMul x.1 ((PiTensorProduct.tprod R) fun i …
  -/
  ext _
  /-
    case e_a.e_f.h
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    x : PiTensorProduct R fun i => s i
    p : FreeAddMonoid (Prod R ((i : ι) → s i))
    h : Eq (List.map (fun x => HSMul.hSMul x.1 ((PiTensorProduct.tprod R) fun i => …
    a : R
    x✝ : Prod R ((i : ι) → s i)
    ⊢ Eq (Function.comp (fun x => HSMul.hSMul x.1 ((PiTensorProduct.tprod R) fun i …
  -/
  simp only [comp_apply, smul_smul]
  /-
    🎉 no goals
  -/


/-- Induct using scaled versions of `PiTensorProduct.tprod`. -/
@[elab_as_elim]
protected theorem induction_on {motive : (⨂[R] i, s i) → Prop} (z : ⨂[R] i, s i)
    (smul_tprod : ∀ (r : R) (f : Π i, s i), motive (r • tprod R f))
    (add : ∀ x y, motive x → motive y → motive (x + y)) :
    motive z := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    motive : (PiTensorProduct R fun i => s i) → Prop
    z : PiTensorProduct R fun i => s i
    smul_tprod : ∀ (r : R) (f : (i : ι) → s i), motive (HSMul.hSMul r ((PiTensorPr …
    add : ∀ (x y : PiTensorProduct R fun i => s i), motive x → motive y → motive ( …
    ⊢ motive z
  -/
  simp_rw [← tprodCoeff_eq_smul_tprod] at smul_tprod
  /-
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    motive : (PiTensorProduct R fun i => s i) → Prop
    z : PiTensorProduct R fun i => s i
    add : ∀ (x y : PiTensorProduct R fun i => s i), motive x → motive y → motive ( …
    smul_tprod : ∀ (r : R) (f : (i : ι) → s i), motive (PiTensorProduct.tprodCoeff …
    ⊢ motive z
  -/
  exact PiTensorProduct.induction_on' z smul_tprod add
  /-
    🎉 no goals
  -/


@[ext]
theorem ext {φ₁ φ₂ : (⨂[R] i, s i) →ₗ[R] E}
    (H : φ₁.compMultilinearMap (tprod R) = φ₂.compMultilinearMap (tprod R)) : φ₁ = φ₂ := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁴ : CommSemiring R
    s : ι → Type u_7
    inst✝³ : (i : ι) → AddCommMonoid (s i)
    inst✝² : (i : ι) → Module R (s i)
    E : Type u_9
    inst✝¹ : AddCommMonoid E
    inst✝ : Module R E
    φ₁ φ₂ : LinearMap (RingHom.id R) (PiTensorProduct R fun i => s i) E
    H : Eq (φ₁.compMultilinearMap (PiTensorProduct.tprod R)) (φ₂.compMultilinearMa …
    ⊢ Eq φ₁ φ₂
  -/
  refine LinearMap.ext ?_
  refine fun z ↦
    PiTensorProduct.induction_on' z ?_ fun {x y} hx hy ↦ by rw [φ₁.map_add, φ₂.map_add, hx, hy]
    /-
      ι : Type u_1
      R : Type u_4
      inst✝⁴ : CommSemiring R
      s : ι → Type u_7
      inst✝³ : (i : ι) → AddCommMonoid (s i)
      inst✝² : (i : ι) → Module R (s i)
      E : Type u_9
      inst✝¹ : AddCommMonoid E
      inst✝ : Module R E
      φ₁ φ₂ : LinearMap (RingHom.id R) (PiTensorProduct R fun i => s i) E
      H : Eq (φ₁.compMultilinearMap (PiTensorProduct.tprod R)) (φ₂.compMultilinearMa …
      z : PiTensorProduct R fun i => s i
      ⊢ ∀ (r : R) (f : (i : ι) → s i), Eq (φ₁ (PiTensorProduct.tprodCoeff R r f)) (φ …
    -/
  · intro r f
    /-
      ι : Type u_1
      R : Type u_4
      inst✝⁴ : CommSemiring R
      s : ι → Type u_7
      inst✝³ : (i : ι) → AddCommMonoid (s i)
      inst✝² : (i : ι) → Module R (s i)
      E : Type u_9
      inst✝¹ : AddCommMonoid E
      inst✝ : Module R E
      φ₁ φ₂ : LinearMap (RingHom.id R) (PiTensorProduct R fun i => s i) E
      H : Eq (φ₁.compMultilinearMap (PiTensorProduct.tprod R)) (φ₂.compMultilinearMa …
      z : PiTensorProduct R fun i => s i
      r : R
      f : (i : ι) → s i
      ⊢ Eq (φ₁ (PiTensorProduct.tprodCoeff R r f)) (φ₂ (PiTensorProduct.tprodCoeff R …
    -/
    rw [tprodCoeff_eq_smul_tprod, φ₁.map_smul, φ₂.map_smul]
    /-
      ι : Type u_1
      R : Type u_4
      inst✝⁴ : CommSemiring R
      s : ι → Type u_7
      inst✝³ : (i : ι) → AddCommMonoid (s i)
      inst✝² : (i : ι) → Module R (s i)
      E : Type u_9
      inst✝¹ : AddCommMonoid E
      inst✝ : Module R E
      φ₁ φ₂ : LinearMap (RingHom.id R) (PiTensorProduct R fun i => s i) E
      H : Eq (φ₁.compMultilinearMap (PiTensorProduct.tprod R)) (φ₂.compMultilinearMa …
      z : PiTensorProduct R fun i => s i
      r : R
      f : (i : ι) → s i
      ⊢ Eq (HSMul.hSMul r (φ₁ ((PiTensorProduct.tprod R) f))) (HSMul.hSMul r (φ₂ ((P …
    -/
    apply congr_arg
    /-
      case h
      ι : Type u_1
      R : Type u_4
      inst✝⁴ : CommSemiring R
      s : ι → Type u_7
      inst✝³ : (i : ι) → AddCommMonoid (s i)
      inst✝² : (i : ι) → Module R (s i)
      E : Type u_9
      inst✝¹ : AddCommMonoid E
      inst✝ : Module R E
      φ₁ φ₂ : LinearMap (RingHom.id R) (PiTensorProduct R fun i => s i) E
      H : Eq (φ₁.compMultilinearMap (PiTensorProduct.tprod R)) (φ₂.compMultilinearMa …
      z : PiTensorProduct R fun i => s i
      r : R
      f : (i : ι) → s i
      ⊢ Eq (φ₁ ((PiTensorProduct.tprod R) f)) (φ₂ ((PiTensorProduct.tprod R) f))
    -/
    exact MultilinearMap.congr_fun H f
    /-
      🎉 no goals
    -/


/-- The pure tensors (i.e. the elements of the image of `PiTensorProduct.tprod`) span
the tensor product. -/
theorem span_tprod_eq_top :
    Submodule.span R (Set.range (tprod R)) = (⊤ : Submodule R (⨂[R] i, s i)) :=
  Submodule.eq_top_iff'.mpr fun t ↦ t.induction_on
    (fun _ _ ↦ Submodule.smul_mem _ _
                                 /-
                                   ι : Type u_1
                                   R : Type u_4
                                   inst✝² : CommSemiring R
                                   s : ι → Type u_7
                                   inst✝¹ : (i : ι) → AddCommMonoid (s i)
                                   inst✝ : (i : ι) → Module R (s i)
                                   t : PiTensorProduct R fun i => s i
                                   x✝¹ : R
                                   x✝ : (i : ι) → s i
                                   ⊢ Membership.mem (Set.range ⇑(PiTensorProduct.tprod R)) ((PiTensorProduct.tpro …
                                 -/
      (Submodule.subset_span (by simp only [Set.mem_range, exists_apply_eq_apply])))
                                 /-
                                   🎉 no goals
                                 -/
    (fun _ _ hx hy ↦ Submodule.add_mem _ hx hy)


/-- Auxiliary function to constructing a linear map `(⨂[R] i, s i) → E` given a
`MultilinearMap R s E` with the property that its composition with the canonical
`MultilinearMap R s (⨂[R] i, s i)` is the given multilinear map. -/
def liftAux (φ : MultilinearMap R s E) : (⨂[R] i, s i) →+ E :=
  liftAddHom (fun p : R × Π i, s i ↦ p.1 • φ p.2)
                       /-
                         ι : Type u_1
                         ι₂ : Type u_2
                         ι₃ : Type u_3
                         R : Type u_4
                         inst✝⁷ : CommSemiring R
                         R₁ : Type u_5
                         R₂ : Type u_6
                         s : ι → Type u_7
                         inst✝⁶ : (i : ι) → AddCommMonoid (s i)
                         inst✝⁵ : (i : ι) → Module R (s i)
                         M : Type u_8
                         inst✝⁴ : AddCommMonoid M
                         inst✝³ : Module R M
                         E : Type u_9
                         inst✝² : AddCommMonoid E
                         inst✝¹ : Module R E
                         F : Type u_10
                         inst✝ : AddCommMonoid F
                         φ : MultilinearMap R s E
                         z : R
                         f : (i : ι) → s i
                         i : ι
                         hf : Eq (f i) 0
                         ⊢ Eq ((fun p => HSMul.hSMul p.1 (φ p.2)) { fst := z, snd := f }) 0
                       -/
    (fun z f i hf ↦ by simp_rw [map_coord_zero φ i hf, smul_zero])
                       /-
                         🎉 no goals
                       -/
                /-
                  ι : Type u_1
                  ι₂ : Type u_2
                  ι₃ : Type u_3
                  R : Type u_4
                  inst✝⁷ : CommSemiring R
                  R₁ : Type u_5
                  R₂ : Type u_6
                  s : ι → Type u_7
                  inst✝⁶ : (i : ι) → AddCommMonoid (s i)
                  inst✝⁵ : (i : ι) → Module R (s i)
                  M : Type u_8
                  inst✝⁴ : AddCommMonoid M
                  inst✝³ : Module R M
                  E : Type u_9
                  inst✝² : AddCommMonoid E
                  inst✝¹ : Module R E
                  F : Type u_10
                  inst✝ : AddCommMonoid F
                  φ : MultilinearMap R s E
                  f : (i : ι) → s i
                  ⊢ Eq ((fun p => HSMul.hSMul p.1 (φ p.2)) { fst := 0, snd := f }) 0
                -/
    (fun f ↦ by simp_rw [zero_smul])
                /-
                  🎉 no goals
                -/
                          /-
                            ι : Type u_1
                            ι₂ : Type u_2
                            ι₃ : Type u_3
                            R : Type u_4
                            inst✝⁸ : CommSemiring R
                            R₁ : Type u_5
                            R₂ : Type u_6
                            s : ι → Type u_7
                            inst✝⁷ : (i : ι) → AddCommMonoid (s i)
                            inst✝⁶ : (i : ι) → Module R (s i)
                            M : Type u_8
                            inst✝⁵ : AddCommMonoid M
                            inst✝⁴ : Module R M
                            E : Type u_9
                            inst✝³ : AddCommMonoid E
                            inst✝² : Module R E
                            F : Type u_10
                            inst✝¹ : AddCommMonoid F
                            φ : MultilinearMap R s E
                            inst✝ : DecidableEq ι
                            z : R
                            f : (i : ι) → s i
                            i : ι
                            m₁ m₂ : s i
                            ⊢ Eq (HAdd.hAdd ((fun p => HSMul.hSMul p.1 (φ p.2)) { fst := z, snd := Functio …
                          -/
    (fun z f i m₁ m₂ ↦ by simp_rw [← smul_add, φ.map_update_add])
                          /-
                            🎉 no goals
                          -/
                      /-
                        ι : Type u_1
                        ι₂ : Type u_2
                        ι₃ : Type u_3
                        R : Type u_4
                        inst✝⁷ : CommSemiring R
                        R₁ : Type u_5
                        R₂ : Type u_6
                        s : ι → Type u_7
                        inst✝⁶ : (i : ι) → AddCommMonoid (s i)
                        inst✝⁵ : (i : ι) → Module R (s i)
                        M : Type u_8
                        inst✝⁴ : AddCommMonoid M
                        inst✝³ : Module R M
                        E : Type u_9
                        inst✝² : AddCommMonoid E
                        inst✝¹ : Module R E
                        F : Type u_10
                        inst✝ : AddCommMonoid F
                        φ : MultilinearMap R s E
                        z₁ z₂ : R
                        f : (i : ι) → s i
                        ⊢ Eq (HAdd.hAdd ((fun p => HSMul.hSMul p.1 (φ p.2)) { fst := z₁, snd := f }) ( …
                      -/
    (fun z₁ z₂ f ↦ by rw [← add_smul])
                      /-
                        🎉 no goals
                      -/
                     /-
                       ι : Type u_1
                       ι₂ : Type u_2
                       ι₃ : Type u_3
                       R : Type u_4
                       inst✝⁸ : CommSemiring R
                       R₁ : Type u_5
                       R₂ : Type u_6
                       s : ι → Type u_7
                       inst✝⁷ : (i : ι) → AddCommMonoid (s i)
                       inst✝⁶ : (i : ι) → Module R (s i)
                       M : Type u_8
                       inst✝⁵ : AddCommMonoid M
                       inst✝⁴ : Module R M
                       E : Type u_9
                       inst✝³ : AddCommMonoid E
                       inst✝² : Module R E
                       F : Type u_10
                       inst✝¹ : AddCommMonoid F
                       φ : MultilinearMap R s E
                       inst✝ : DecidableEq ι
                       z : R
                       f : (i : ι) → s i
                       i : ι
                       r : R
                       ⊢ Eq ((fun p => HSMul.hSMul p.1 (φ p.2)) { fst := z, snd := Function.update f  …
                     -/
    fun z f i r ↦ by simp [φ.map_update_smul, smul_smul, mul_comm]
                     /-
                       🎉 no goals
                     -/


theorem liftAux_tprod (φ : MultilinearMap R s E) (f : Π i, s i) : liftAux φ (tprod R f) = φ f := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁴ : CommSemiring R
    s : ι → Type u_7
    inst✝³ : (i : ι) → AddCommMonoid (s i)
    inst✝² : (i : ι) → Module R (s i)
    E : Type u_9
    inst✝¹ : AddCommMonoid E
    inst✝ : Module R E
    φ : MultilinearMap R s E
    f : (i : ι) → s i
    ⊢ Eq ((PiTensorProduct.liftAux φ) ((PiTensorProduct.tprod R) f)) (φ f)
  -/
  simp only [liftAux, liftAddHom, tprod_eq_tprodCoeff_one, tprodCoeff, AddCon.coe_mk']
  -- The end of this proof was very different before https://github.com/leanprover/lean4/pull/2644:
  -- rw [FreeAddMonoid.of, FreeAddMonoid.ofList, Equiv.refl_apply, AddCon.lift_coe]
  -- dsimp [FreeAddMonoid.lift, FreeAddMonoid.sumAux]
  -- show _ • _ = _
  -- rw [one_smul]
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁴ : CommSemiring R
    s : ι → Type u_7
    inst✝³ : (i : ι) → AddCommMonoid (s i)
    inst✝² : (i : ι) → Module R (s i)
    E : Type u_9
    inst✝¹ : AddCommMonoid E
    inst✝ : Module R E
    φ : MultilinearMap R s E
    f : (i : ι) → s i
    ⊢ Eq (((addConGen (PiTensorProduct.Eqv R s)).lift (FreeAddMonoid.lift fun p => …
  -/
  erw [AddCon.lift_coe]
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁴ : CommSemiring R
    s : ι → Type u_7
    inst✝³ : (i : ι) → AddCommMonoid (s i)
    inst✝² : (i : ι) → Module R (s i)
    E : Type u_9
    inst✝¹ : AddCommMonoid E
    inst✝ : Module R E
    φ : MultilinearMap R s E
    f : (i : ι) → s i
    ⊢ Eq ((FreeAddMonoid.lift fun p => HSMul.hSMul p.1 (φ p.2)) (FreeAddMonoid.of  …
  -/
  rw [FreeAddMonoid.of]
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁴ : CommSemiring R
    s : ι → Type u_7
    inst✝³ : (i : ι) → AddCommMonoid (s i)
    inst✝² : (i : ι) → Module R (s i)
    E : Type u_9
    inst✝¹ : AddCommMonoid E
    inst✝ : Module R E
    φ : MultilinearMap R s E
    f : (i : ι) → s i
    ⊢ Eq ((FreeAddMonoid.lift fun p => HSMul.hSMul p.1 (φ p.2)) (FreeAddMonoid.ofL …
  -/
  dsimp [FreeAddMonoid.ofList]
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁴ : CommSemiring R
    s : ι → Type u_7
    inst✝³ : (i : ι) → AddCommMonoid (s i)
    inst✝² : (i : ι) → Module R (s i)
    E : Type u_9
    inst✝¹ : AddCommMonoid E
    inst✝ : Module R E
    φ : MultilinearMap R s E
    f : (i : ι) → s i
    ⊢ Eq ((FreeAddMonoid.lift fun p => HSMul.hSMul p.1 (φ p.2)) ((Equiv.refl (List …
  -/
  rw [← one_smul R (φ f)]
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁴ : CommSemiring R
    s : ι → Type u_7
    inst✝³ : (i : ι) → AddCommMonoid (s i)
    inst✝² : (i : ι) → Module R (s i)
    E : Type u_9
    inst✝¹ : AddCommMonoid E
    inst✝ : Module R E
    φ : MultilinearMap R s E
    f : (i : ι) → s i
    ⊢ Eq ((FreeAddMonoid.lift fun p => HSMul.hSMul p.1 (φ p.2)) ((Equiv.refl (List …
  -/
  erw [Equiv.refl_apply]
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁴ : CommSemiring R
    s : ι → Type u_7
    inst✝³ : (i : ι) → AddCommMonoid (s i)
    inst✝² : (i : ι) → Module R (s i)
    E : Type u_9
    inst✝¹ : AddCommMonoid E
    inst✝ : Module R E
    φ : MultilinearMap R s E
    f : (i : ι) → s i
    ⊢ Eq (FreeAddMonoid.sumAux.match_1 (fun x => E) (List.map (fun p => HSMul.hSMu …
  -/
  convert one_smul R (φ f)
  /-
    case h.e'_3
    ι : Type u_1
    R : Type u_4
    inst✝⁴ : CommSemiring R
    s : ι → Type u_7
    inst✝³ : (i : ι) → AddCommMonoid (s i)
    inst✝² : (i : ι) → Module R (s i)
    E : Type u_9
    inst✝¹ : AddCommMonoid E
    inst✝ : Module R E
    φ : MultilinearMap R s E
    f : (i : ι) → s i
    ⊢ Eq (HSMul.hSMul 1 (φ f)) (φ f)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem liftAux_tprodCoeff (φ : MultilinearMap R s E) (z : R) (f : Π i, s i) :
    liftAux φ (tprodCoeff R z f) = z • φ f := rfl


theorem liftAux.smul {φ : MultilinearMap R s E} (r : R) (x : ⨂[R] i, s i) :
    liftAux φ (r • x) = r • liftAux φ x := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁴ : CommSemiring R
    s : ι → Type u_7
    inst✝³ : (i : ι) → AddCommMonoid (s i)
    inst✝² : (i : ι) → Module R (s i)
    E : Type u_9
    inst✝¹ : AddCommMonoid E
    inst✝ : Module R E
    φ : MultilinearMap R s E
    r : R
    x : PiTensorProduct R fun i => s i
    ⊢ Eq ((PiTensorProduct.liftAux φ) (HSMul.hSMul r x)) (HSMul.hSMul r ((PiTensor …
  -/
  refine PiTensorProduct.induction_on' x ?_ ?_
    /-
      case refine_1
      ι : Type u_1
      R : Type u_4
      inst✝⁴ : CommSemiring R
      s : ι → Type u_7
      inst✝³ : (i : ι) → AddCommMonoid (s i)
      inst✝² : (i : ι) → Module R (s i)
      E : Type u_9
      inst✝¹ : AddCommMonoid E
      inst✝ : Module R E
      φ : MultilinearMap R s E
      r : R
      x : PiTensorProduct R fun i => s i
      ⊢ ∀ (r_1 : R) (f : (i : ι) → s i), Eq ((PiTensorProduct.liftAux φ) (HSMul.hSMu …
    -/
  · intro z f
    /-
      case refine_1
      ι : Type u_1
      R : Type u_4
      inst✝⁴ : CommSemiring R
      s : ι → Type u_7
      inst✝³ : (i : ι) → AddCommMonoid (s i)
      inst✝² : (i : ι) → Module R (s i)
      E : Type u_9
      inst✝¹ : AddCommMonoid E
      inst✝ : Module R E
      φ : MultilinearMap R s E
      r : R
      x : PiTensorProduct R fun i => s i
      z : R
      f : (i : ι) → s i
      ⊢ Eq ((PiTensorProduct.liftAux φ) (HSMul.hSMul r (PiTensorProduct.tprodCoeff R …
    -/
    rw [smul_tprodCoeff' r z f, liftAux_tprodCoeff, liftAux_tprodCoeff, smul_assoc]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      R : Type u_4
      inst✝⁴ : CommSemiring R
      s : ι → Type u_7
      inst✝³ : (i : ι) → AddCommMonoid (s i)
      inst✝² : (i : ι) → Module R (s i)
      E : Type u_9
      inst✝¹ : AddCommMonoid E
      inst✝ : Module R E
      φ : MultilinearMap R s E
      r : R
      x : PiTensorProduct R fun i => s i
      ⊢ ∀ (x y : PiTensorProduct R fun i => s i), Eq ((PiTensorProduct.liftAux φ) (H …
    -/
  · intro z y ihz ihy
    /-
      case refine_2
      ι : Type u_1
      R : Type u_4
      inst✝⁴ : CommSemiring R
      s : ι → Type u_7
      inst✝³ : (i : ι) → AddCommMonoid (s i)
      inst✝² : (i : ι) → Module R (s i)
      E : Type u_9
      inst✝¹ : AddCommMonoid E
      inst✝ : Module R E
      φ : MultilinearMap R s E
      r : R
      x z y : PiTensorProduct R fun i => s i
      ihz : Eq ((PiTensorProduct.liftAux φ) (HSMul.hSMul r z)) (HSMul.hSMul r ((PiTe …
      ihy : Eq ((PiTensorProduct.liftAux φ) (HSMul.hSMul r y)) (HSMul.hSMul r ((PiTe …
      ⊢ Eq ((PiTensorProduct.liftAux φ) (HSMul.hSMul r (HAdd.hAdd z y))) (HSMul.hSMu …
    -/
    rw [smul_add, (liftAux φ).map_add, ihz, ihy, (liftAux φ).map_add, smul_add]
    /-
      🎉 no goals
    -/


/-- Constructing a linear map `(⨂[R] i, s i) → E` given a `MultilinearMap R s E` with the
property that its composition with the canonical `MultilinearMap R s E` is
the given multilinear map `φ`. -/
def lift : MultilinearMap R s E ≃ₗ[R] (⨂[R] i, s i) →ₗ[R] E where
  toFun φ := { liftAux φ with map_smul' := liftAux.smul }
  invFun φ' := φ'.compMultilinearMap (tprod R)
  left_inv φ := by
    /-
      ι : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      R : Type u_4
      inst✝⁷ : CommSemiring R
      R₁ : Type u_5
      R₂ : Type u_6
      s : ι → Type u_7
      inst✝⁶ : (i : ι) → AddCommMonoid (s i)
      inst✝⁵ : (i : ι) → Module R (s i)
      M : Type u_8
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      E : Type u_9
      inst✝² : AddCommMonoid E
      inst✝¹ : Module R E
      F : Type u_10
      inst✝ : AddCommMonoid F
      φ : MultilinearMap R s E
      ⊢ Eq
          ((fun φ' => φ'.compMultilinearMap (PiTensorProduct.tprod R))
            ({
                  toFun := fun φ =>
                    let __src := PiTensorProduct.liftAux φ;
                    { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ },
                  map_add' := ⋯, map_smul' := ⋯ }.toFun
              φ))
          φ
    -/
    ext
    /-
      case H
      ι : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      R : Type u_4
      inst✝⁷ : CommSemiring R
      R₁ : Type u_5
      R₂ : Type u_6
      s : ι → Type u_7
      inst✝⁶ : (i : ι) → AddCommMonoid (s i)
      inst✝⁵ : (i : ι) → Module R (s i)
      M : Type u_8
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      E : Type u_9
      inst✝² : AddCommMonoid E
      inst✝¹ : Module R E
      F : Type u_10
      inst✝ : AddCommMonoid F
      φ : MultilinearMap R s E
      x✝ : (i : ι) → s i
      ⊢ Eq
          (((fun φ' => φ'.compMultilinearMap (PiTensorProduct.tprod R))
              ({
                    toFun := fun φ =>
                      let __src := PiTensorProduct.liftAux φ;
                      { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ },
                    map_add' := ⋯, map_smul' := ⋯ }.toFun
                φ))
            x✝)
          (φ x✝)
    -/
    simp [liftAux_tprod, LinearMap.compMultilinearMap]
    /-
      🎉 no goals
    -/
  right_inv φ := by
    /-
      ι : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      R : Type u_4
      inst✝⁷ : CommSemiring R
      R₁ : Type u_5
      R₂ : Type u_6
      s : ι → Type u_7
      inst✝⁶ : (i : ι) → AddCommMonoid (s i)
      inst✝⁵ : (i : ι) → Module R (s i)
      M : Type u_8
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      E : Type u_9
      inst✝² : AddCommMonoid E
      inst✝¹ : Module R E
      F : Type u_10
      inst✝ : AddCommMonoid F
      φ₁ φ₂ : MultilinearMap R s E
      ⊢ Eq
          ((fun φ =>
              let __src := PiTensorProduct.liftAux φ;
              { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ })
            (HAdd.hAdd φ₁ φ₂))
          (HAdd.hAdd
            ((fun φ =>
                let __src := PiTensorProduct.liftAux φ;
                { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ })
              φ₁)
            ((fun φ =>
                let __src := PiTensorProduct.liftAux φ;
                { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ })
              φ₂))
    -/
    /-
      ι : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      R : Type u_4
      inst✝⁷ : CommSemiring R
      R₁ : Type u_5
      R₂ : Type u_6
      s : ι → Type u_7
      inst✝⁶ : (i : ι) → AddCommMonoid (s i)
      inst✝⁵ : (i : ι) → Module R (s i)
      M : Type u_8
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      E : Type u_9
      inst✝² : AddCommMonoid E
      inst✝¹ : Module R E
      F : Type u_10
      inst✝ : AddCommMonoid F
      φ : LinearMap (RingHom.id R) (PiTensorProduct R fun i => s i) E
      ⊢ Eq
          ({
                toFun := fun φ =>
                  let __src := PiTensorProduct.liftAux φ;
                  { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ },
                map_add' := ⋯, map_smul' := ⋯ }.toFun
            ((fun φ' => φ'.compMultilinearMap (PiTensorProduct.tprod R)) φ))
          φ
    -/
    /-
      case H.H
      ι : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      R : Type u_4
      inst✝⁷ : CommSemiring R
      R₁ : Type u_5
      R₂ : Type u_6
      s : ι → Type u_7
      inst✝⁶ : (i : ι) → AddCommMonoid (s i)
      inst✝⁵ : (i : ι) → Module R (s i)
      M : Type u_8
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      E : Type u_9
      inst✝² : AddCommMonoid E
      inst✝¹ : Module R E
      F : Type u_10
      inst✝ : AddCommMonoid F
      φ₁ φ₂ : MultilinearMap R s E
      x✝ : (i : ι) → s i
      ⊢ Eq
          ((((fun φ =>
                    let __src := PiTensorProduct.liftAux φ;
                    { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ })
                  (HAdd.hAdd φ₁ φ₂)).compMultilinearMap
              (PiTensorProduct.tprod R))
            x✝)
          (((HAdd.hAdd
                  ((fun φ =>
                      let __src := PiTensorProduct.liftAux φ;
                      { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ })
                    φ₁)
                  ((fun φ =>
                      let __src := PiTensorProduct.liftAux φ;
                      { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ })
                    φ₂)).compMultilinearMap
              (PiTensorProduct.tprod R))
            x✝)
    -/
    ext
    /-
      🎉 no goals
    -/
    /-
      case H.H
      ι : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      R : Type u_4
      inst✝⁷ : CommSemiring R
      R₁ : Type u_5
      R₂ : Type u_6
      s : ι → Type u_7
      inst✝⁶ : (i : ι) → AddCommMonoid (s i)
      inst✝⁵ : (i : ι) → Module R (s i)
      M : Type u_8
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      E : Type u_9
      inst✝² : AddCommMonoid E
      inst✝¹ : Module R E
      F : Type u_10
      inst✝ : AddCommMonoid F
      φ : LinearMap (RingHom.id R) (PiTensorProduct R fun i => s i) E
      x✝ : (i : ι) → s i
      ⊢ Eq
          ((({
                      toFun := fun φ =>
                        let __src := PiTensorProduct.liftAux φ;
                        { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ },
                      map_add' := ⋯, map_smul' := ⋯ }.toFun
                  ((fun φ' => φ'.compMultilinearMap (PiTensorProduct.tprod R)) φ)).c …
              (PiTensorProduct.tprod R))
            x✝)
          ((φ.compMultilinearMap (PiTensorProduct.tprod R)) x✝)
    -/
    /-
      ι : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      R : Type u_4
      inst✝⁷ : CommSemiring R
      R₁ : Type u_5
      R₂ : Type u_6
      s : ι → Type u_7
      inst✝⁶ : (i : ι) → AddCommMonoid (s i)
      inst✝⁵ : (i : ι) → Module R (s i)
      M : Type u_8
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      E : Type u_9
      inst✝² : AddCommMonoid E
      inst✝¹ : Module R E
      F : Type u_10
      inst✝ : AddCommMonoid F
      r : R
      φ₂ : MultilinearMap R s E
      ⊢ Eq
          ({
                toFun := fun φ =>
                  let __src := PiTensorProduct.liftAux φ;
                  { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ },
                map_add' := ⋯ }.toFun
            (HSMul.hSMul r φ₂))
          (HSMul.hSMul ((RingHom.id R) r)
            ({
                  toFun := fun φ =>
                    let __src := PiTensorProduct.liftAux φ;
                    { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ },
                  map_add' := ⋯ }.toFun
              φ₂))
    -/
    simp [liftAux_tprod]
    /-
      case H.H
      ι : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      R : Type u_4
      inst✝⁷ : CommSemiring R
      R₁ : Type u_5
      R₂ : Type u_6
      s : ι → Type u_7
      inst✝⁶ : (i : ι) → AddCommMonoid (s i)
      inst✝⁵ : (i : ι) → Module R (s i)
      M : Type u_8
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      E : Type u_9
      inst✝² : AddCommMonoid E
      inst✝¹ : Module R E
      F : Type u_10
      inst✝ : AddCommMonoid F
      r : R
      φ₂ : MultilinearMap R s E
      x✝ : (i : ι) → s i
      ⊢ Eq
          ((({
                      toFun := fun φ =>
                        let __src := PiTensorProduct.liftAux φ;
                        { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ },
                      map_add' := ⋯ }.toFun
                  (HSMul.hSMul r φ₂)).compMultilinearMap
              (PiTensorProduct.tprod R))
            x✝)
          (((HSMul.hSMul ((RingHom.id R) r)
                  ({
                        toFun := fun φ =>
                          let __src := PiTensorProduct.liftAux φ;
                          { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ },
                        map_add' := ⋯ }.toFun
                    φ₂)).compMultilinearMap
              (PiTensorProduct.tprod R))
            x✝)
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  map_add' φ₁ φ₂ := by
    ext
    simp [liftAux_tprod]
  map_smul' r φ₂ := by
    ext
    simp [liftAux_tprod]


@[simp]
theorem lift.tprod (f : Π i, s i) : lift φ (tprod R f) = φ f :=
  liftAux_tprod φ f


theorem lift.unique' {φ' : (⨂[R] i, s i) →ₗ[R] E}
    (H : φ'.compMultilinearMap (PiTensorProduct.tprod R) = φ) : φ' = lift φ :=
  ext <| H.symm ▸ (lift.symm_apply_apply φ).symm


theorem lift.unique {φ' : (⨂[R] i, s i) →ₗ[R] E} (H : ∀ f, φ' (PiTensorProduct.tprod R f) = φ f) :
    φ' = lift φ :=
  lift.unique' (MultilinearMap.ext H)


@[simp]
theorem lift_symm (φ' : (⨂[R] i, s i) →ₗ[R] E) : lift.symm φ' = φ'.compMultilinearMap (tprod R) :=
  rfl


@[simp]
theorem lift_tprod : lift (tprod R : MultilinearMap R s _) = LinearMap.id :=
  Eq.symm <| lift.unique' rfl


/--
Let `sᵢ` and `tᵢ` be two families of `R`-modules.
Let `f` be a family of `R`-linear maps between `sᵢ` and `tᵢ`, i.e. `f : Πᵢ sᵢ → tᵢ`,
then there is an induced map `⨂ᵢ sᵢ → ⨂ᵢ tᵢ` by `⨂ aᵢ ↦ ⨂ fᵢ aᵢ`.

This is `TensorProduct.map` for an arbitrary family of modules.
-/
def map : (⨂[R] i, s i) →ₗ[R] ⨂[R] i, t i :=
  lift <| (tprod R).compLinearMap f


@[simp] lemma map_tprod (x : Π i, s i) :
    map f (tprod R x) = tprod R fun i ↦ f i (x i) :=
  lift.tprod _

-- No lemmas about associativity, because we don't have associativity of `PiTensorProduct` yet.


theorem map_range_eq_span_tprod :
    LinearMap.range (map f) =
      Submodule.span R {t | ∃ (m : Π i, s i), tprod R (fun i ↦ f i (m i)) = t} := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁴ : CommSemiring R
    s : ι → Type u_7
    inst✝³ : (i : ι) → AddCommMonoid (s i)
    inst✝² : (i : ι) → Module R (s i)
    t : ι → Type u_11
    inst✝¹ : (i : ι) → AddCommMonoid (t i)
    inst✝ : (i : ι) → Module R (t i)
    f : (i : ι) → LinearMap (RingHom.id R) (s i) (t i)
    ⊢ Eq (LinearMap.range (PiTensorProduct.map f)) (Submodule.span R (setOf fun t_ …
  -/
  rw [← Submodule.map_top, ← span_tprod_eq_top, Submodule.map_span, ← Set.range_comp]
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁴ : CommSemiring R
    s : ι → Type u_7
    inst✝³ : (i : ι) → AddCommMonoid (s i)
    inst✝² : (i : ι) → Module R (s i)
    t : ι → Type u_11
    inst✝¹ : (i : ι) → AddCommMonoid (t i)
    inst✝ : (i : ι) → Module R (t i)
    f : (i : ι) → LinearMap (RingHom.id R) (s i) (t i)
    ⊢ Eq (Submodule.span R (Set.range (Function.comp ⇑(PiTensorProduct.map f) ⇑(Pi …
  -/
  apply congrArg; ext x
  /-
    case h.h
    ι : Type u_1
    R : Type u_4
    inst✝⁴ : CommSemiring R
    s : ι → Type u_7
    inst✝³ : (i : ι) → AddCommMonoid (s i)
    inst✝² : (i : ι) → Module R (s i)
    t : ι → Type u_11
    inst✝¹ : (i : ι) → AddCommMonoid (t i)
    inst✝ : (i : ι) → Module R (t i)
    f : (i : ι) → LinearMap (RingHom.id R) (s i) (t i)
    x : PiTensorProduct R fun i => t i
    ⊢ Iff (Membership.mem (Set.range (Function.comp ⇑(PiTensorProduct.map f) ⇑(PiT …
  -/
  simp only [Set.mem_range, comp_apply, map_tprod, Set.mem_setOf_eq]
  /-
    🎉 no goals
  -/


/-- Given submodules `p i ⊆ s i`, this is the natural map: `⨂[R] i, p i → ⨂[R] i, s i`.
This is `TensorProduct.mapIncl` for an arbitrary family of modules.
-/
@[simp]
def mapIncl (p : Π i, Submodule R (s i)) : (⨂[R] i, p i) →ₗ[R] ⨂[R] i, s i :=
  map fun (i : ι) ↦ (p i).subtype


theorem map_comp : map (fun (i : ι) ↦ g i ∘ₗ f i) = map g ∘ₗ map f := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁶ : CommSemiring R
    s : ι → Type u_7
    inst✝⁵ : (i : ι) → AddCommMonoid (s i)
    inst✝⁴ : (i : ι) → Module R (s i)
    t : ι → Type u_11
    t' : ι → Type u_12
    inst✝³ : (i : ι) → AddCommMonoid (t i)
    inst✝² : (i : ι) → Module R (t i)
    inst✝¹ : (i : ι) → AddCommMonoid (t' i)
    inst✝ : (i : ι) → Module R (t' i)
    g : (i : ι) → LinearMap (RingHom.id R) (t i) (t' i)
    f : (i : ι) → LinearMap (RingHom.id R) (s i) (t i)
    ⊢ Eq (PiTensorProduct.map fun i => (g i).comp (f i)) ((PiTensorProduct.map g). …
  -/
  ext
  /-
    case H.H
    ι : Type u_1
    R : Type u_4
    inst✝⁶ : CommSemiring R
    s : ι → Type u_7
    inst✝⁵ : (i : ι) → AddCommMonoid (s i)
    inst✝⁴ : (i : ι) → Module R (s i)
    t : ι → Type u_11
    t' : ι → Type u_12
    inst✝³ : (i : ι) → AddCommMonoid (t i)
    inst✝² : (i : ι) → Module R (t i)
    inst✝¹ : (i : ι) → AddCommMonoid (t' i)
    inst✝ : (i : ι) → Module R (t' i)
    g : (i : ι) → LinearMap (RingHom.id R) (t i) (t' i)
    f : (i : ι) → LinearMap (RingHom.id R) (s i) (t i)
    x✝ : (i : ι) → s i
    ⊢ Eq (((PiTensorProduct.map fun i => (g i).comp (f i)).compMultilinearMap (PiT …
  -/
  simp only [LinearMap.compMultilinearMap_apply, map_tprod, LinearMap.coe_comp, Function.comp_apply]
  /-
    🎉 no goals
  -/


theorem lift_comp_map (h : MultilinearMap R t E) :
    lift h ∘ₗ map f = lift (h.compLinearMap f) := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁶ : CommSemiring R
    s : ι → Type u_7
    inst✝⁵ : (i : ι) → AddCommMonoid (s i)
    inst✝⁴ : (i : ι) → Module R (s i)
    E : Type u_9
    inst✝³ : AddCommMonoid E
    inst✝² : Module R E
    t : ι → Type u_11
    inst✝¹ : (i : ι) → AddCommMonoid (t i)
    inst✝ : (i : ι) → Module R (t i)
    f : (i : ι) → LinearMap (RingHom.id R) (s i) (t i)
    h : MultilinearMap R t E
    ⊢ Eq ((PiTensorProduct.lift h).comp (PiTensorProduct.map f)) (PiTensorProduct. …
  -/
  ext
  simp only [LinearMap.compMultilinearMap_apply, LinearMap.coe_comp, Function.comp_apply,
    map_tprod, lift.tprod, MultilinearMap.compLinearMap_apply]


@[simp]
theorem map_id : map (fun i ↦ (LinearMap.id : s i →ₗ[R] s i)) = .id := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    ⊢ Eq (PiTensorProduct.map fun i => LinearMap.id) LinearMap.id
  -/
  ext
  /-
    case H.H
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    x✝ : (i : ι) → s i
    ⊢ Eq (((PiTensorProduct.map fun i => LinearMap.id).compMultilinearMap (PiTenso …
  -/
  simp only [LinearMap.compMultilinearMap_apply, map_tprod, LinearMap.id_coe, id_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_one : map (fun (i : ι) ↦ (1 : s i →ₗ[R] s i)) = 1 :=
  map_id


theorem map_mul (f₁ f₂ : Π i, s i →ₗ[R] s i) :
    map (fun i ↦ f₁ i * f₂ i) = map f₁ * map f₂ :=
  map_comp f₁ f₂


/-- Upgrading `PiTensorProduct.map` to a `MonoidHom` when `s = t`. -/
@[simps]
def mapMonoidHom : (Π i, s i →ₗ[R] s i) →* ((⨂[R] i, s i) →ₗ[R] ⨂[R] i, s i) where
  toFun := map
  map_one' := map_one
  map_mul' := map_mul


@[simp]
protected theorem map_pow (f : Π i, s i →ₗ[R] s i) (n : ℕ) :
    map (f ^ n) = map f ^ n := MonoidHom.map_pow mapMonoidHom _ _


open Function in
private theorem map_add_smul_aux [DecidableEq ι] (i : ι) (x : Π i, s i) (u : s i →ₗ[R] t i) :
    (fun j ↦ update f i u j (x j)) = update (fun j ↦ (f j) (x j)) i (u (x i)) := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁵ : CommSemiring R
    s : ι → Type u_7
    inst✝⁴ : (i : ι) → AddCommMonoid (s i)
    inst✝³ : (i : ι) → Module R (s i)
    t : ι → Type u_11
    inst✝² : (i : ι) → AddCommMonoid (t i)
    inst✝¹ : (i : ι) → Module R (t i)
    f : (i : ι) → LinearMap (RingHom.id R) (s i) (t i)
    inst✝ : DecidableEq ι
    i : ι
    x : (i : ι) → s i
    u : LinearMap (RingHom.id R) (s i) (t i)
    ⊢ Eq (fun j => (Function.update f i u j) (x j)) (Function.update (fun j => (f  …
  -/
  ext j
  /-
    case h
    ι : Type u_1
    R : Type u_4
    inst✝⁵ : CommSemiring R
    s : ι → Type u_7
    inst✝⁴ : (i : ι) → AddCommMonoid (s i)
    inst✝³ : (i : ι) → Module R (s i)
    t : ι → Type u_11
    inst✝² : (i : ι) → AddCommMonoid (t i)
    inst✝¹ : (i : ι) → Module R (t i)
    f : (i : ι) → LinearMap (RingHom.id R) (s i) (t i)
    inst✝ : DecidableEq ι
    i : ι
    x : (i : ι) → s i
    u : LinearMap (RingHom.id R) (s i) (t i)
    j : ι
    ⊢ Eq ((Function.update f i u j) (x j)) (Function.update (fun j => (f j) (x j)) …
  -/
  exact apply_update (fun i F => F (x i)) f i u j
  /-
    🎉 no goals
  -/


open Function in
protected theorem map_update_add [DecidableEq ι] (i : ι) (u v : s i →ₗ[R] t i) :
    map (update f i (u + v)) = map (update f i u) + map (update f i v) := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁵ : CommSemiring R
    s : ι → Type u_7
    inst✝⁴ : (i : ι) → AddCommMonoid (s i)
    inst✝³ : (i : ι) → Module R (s i)
    t : ι → Type u_11
    inst✝² : (i : ι) → AddCommMonoid (t i)
    inst✝¹ : (i : ι) → Module R (t i)
    f : (i : ι) → LinearMap (RingHom.id R) (s i) (t i)
    inst✝ : DecidableEq ι
    i : ι
    u v : LinearMap (RingHom.id R) (s i) (t i)
    ⊢ Eq (PiTensorProduct.map (Function.update f i (HAdd.hAdd u v))) (HAdd.hAdd (P …
  -/
  ext x
  simp only [LinearMap.compMultilinearMap_apply, map_tprod, map_add_smul_aux, LinearMap.add_apply,
    MultilinearMap.map_update_add]


@[deprecated (since := "2024-11-03")] protected alias map_add := PiTensorProduct.map_update_add


open Function in
protected theorem map_update_smul [DecidableEq ι] (i : ι) (c : R) (u : s i →ₗ[R] t i) :
    map (update f i (c • u)) = c • map (update f i u) := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁵ : CommSemiring R
    s : ι → Type u_7
    inst✝⁴ : (i : ι) → AddCommMonoid (s i)
    inst✝³ : (i : ι) → Module R (s i)
    t : ι → Type u_11
    inst✝² : (i : ι) → AddCommMonoid (t i)
    inst✝¹ : (i : ι) → Module R (t i)
    f : (i : ι) → LinearMap (RingHom.id R) (s i) (t i)
    inst✝ : DecidableEq ι
    i : ι
    c : R
    u : LinearMap (RingHom.id R) (s i) (t i)
    ⊢ Eq (PiTensorProduct.map (Function.update f i (HSMul.hSMul c u))) (HSMul.hSMu …
  -/
  ext x
  simp only [LinearMap.compMultilinearMap_apply, map_tprod, map_add_smul_aux, LinearMap.smul_apply,
    MultilinearMap.map_update_smul]


@[deprecated (since := "2024-11-03")] protected alias map_smul := PiTensorProduct.map_update_smul


/-- The tensor of a family of linear maps from `sᵢ` to `tᵢ`, as a multilinear map of
the family.
-/
@[simps]
noncomputable def mapMultilinear :
    MultilinearMap R (fun (i : ι) ↦ s i →ₗ[R] t i) ((⨂[R] i, s i) →ₗ[R] ⨂[R] i, t i) where
  toFun := map
  map_update_smul' _ _ _ _ := PiTensorProduct.map_update_smul _ _ _ _
  map_update_add' _ _ _ _ := PiTensorProduct.map_update_add _ _ _ _


/--
Let `sᵢ` and `tᵢ` be families of `R`-modules.
Then there is an `R`-linear map between `⨂ᵢ Hom(sᵢ, tᵢ)` and `Hom(⨂ᵢ sᵢ, ⨂ tᵢ)` defined by
`⨂ᵢ fᵢ ↦ ⨂ᵢ aᵢ ↦ ⨂ᵢ fᵢ aᵢ`.

This is `TensorProduct.homTensorHomMap` for an arbitrary family of modules.

Note that `PiTensorProduct.piTensorHomMap (tprod R f)` is equal to `PiTensorProduct.map f`.
-/
def piTensorHomMap : (⨂[R] i, s i →ₗ[R] t i) →ₗ[R] (⨂[R] i, s i) →ₗ[R] ⨂[R] i, t i :=
  lift.toLinearMap ∘ₗ lift (MultilinearMap.piLinearMap <| tprod R)


@[simp] lemma piTensorHomMap_tprod_tprod (f : Π i, s i →ₗ[R] t i) (x : Π i, s i) :
    piTensorHomMap (tprod R f) (tprod R x) = tprod R fun i ↦ f i (x i) := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁴ : CommSemiring R
    s : ι → Type u_7
    inst✝³ : (i : ι) → AddCommMonoid (s i)
    inst✝² : (i : ι) → Module R (s i)
    t : ι → Type u_11
    inst✝¹ : (i : ι) → AddCommMonoid (t i)
    inst✝ : (i : ι) → Module R (t i)
    f : (i : ι) → LinearMap (RingHom.id R) (s i) (t i)
    x : (i : ι) → s i
    ⊢ Eq ((PiTensorProduct.piTensorHomMap ((PiTensorProduct.tprod R) f)) ((PiTenso …
  -/
  simp [piTensorHomMap]
  /-
    🎉 no goals
  -/


lemma piTensorHomMap_tprod_eq_map (f : Π i, s i →ₗ[R] t i) :
    piTensorHomMap (tprod R f) = map f := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁴ : CommSemiring R
    s : ι → Type u_7
    inst✝³ : (i : ι) → AddCommMonoid (s i)
    inst✝² : (i : ι) → Module R (s i)
    t : ι → Type u_11
    inst✝¹ : (i : ι) → AddCommMonoid (t i)
    inst✝ : (i : ι) → Module R (t i)
    f : (i : ι) → LinearMap (RingHom.id R) (s i) (t i)
    ⊢ Eq (PiTensorProduct.piTensorHomMap ((PiTensorProduct.tprod R) f)) (PiTensorP …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


/-- If `s i` and `t i` are linearly equivalent for every `i` in `ι`, then `⨂[R] i, s i` and
`⨂[R] i, t i` are linearly equivalent.

This is the n-ary version of `TensorProduct.congr`
-/
noncomputable def congr (f : Π i, s i ≃ₗ[R] t i) :
    (⨂[R] i, s i) ≃ₗ[R] ⨂[R] i, t i :=
  .ofLinear
    (map (fun i ↦ f i))
    (map (fun i ↦ (f i).symm))
        /-
          ι : Type u_1
          ι₂ : Type u_2
          ι₃ : Type u_3
          R : Type u_4
          inst✝¹¹ : CommSemiring R
          R₁ : Type u_5
          R₂ : Type u_6
          s : ι → Type u_7
          inst✝¹⁰ : (i : ι) → AddCommMonoid (s i)
          inst✝⁹ : (i : ι) → Module R (s i)
          M : Type u_8
          inst✝⁸ : AddCommMonoid M
          inst✝⁷ : Module R M
          E : Type u_9
          inst✝⁶ : AddCommMonoid E
          inst✝⁵ : Module R E
          F : Type u_10
          inst✝⁴ : AddCommMonoid F
          t : ι → Type u_11
          t' : ι → Type u_12
          inst✝³ : (i : ι) → AddCommMonoid (t i)
          inst✝² : (i : ι) → Module R (t i)
          inst✝¹ : (i : ι) → AddCommMonoid (t' i)
          inst✝ : (i : ι) → Module R (t' i)
          g : (i : ι) → LinearMap (RingHom.id R) (t i) (t' i)
          f✝ : (i : ι) → LinearMap (RingHom.id R) (s i) (t i)
          f : (i : ι) → LinearEquiv (RingHom.id R) (s i) (t i)
          ⊢ Eq ((PiTensorProduct.map fun i => ↑(f i)).comp (PiTensorProduct.map fun i => …
        -/
    (by ext; simp)
             /-
               🎉 no goals
             -/
        /-
          ι : Type u_1
          ι₂ : Type u_2
          ι₃ : Type u_3
          R : Type u_4
          inst✝¹¹ : CommSemiring R
          R₁ : Type u_5
          R₂ : Type u_6
          s : ι → Type u_7
          inst✝¹⁰ : (i : ι) → AddCommMonoid (s i)
          inst✝⁹ : (i : ι) → Module R (s i)
          M : Type u_8
          inst✝⁸ : AddCommMonoid M
          inst✝⁷ : Module R M
          E : Type u_9
          inst✝⁶ : AddCommMonoid E
          inst✝⁵ : Module R E
          F : Type u_10
          inst✝⁴ : AddCommMonoid F
          t : ι → Type u_11
          t' : ι → Type u_12
          inst✝³ : (i : ι) → AddCommMonoid (t i)
          inst✝² : (i : ι) → Module R (t i)
          inst✝¹ : (i : ι) → AddCommMonoid (t' i)
          inst✝ : (i : ι) → Module R (t' i)
          g : (i : ι) → LinearMap (RingHom.id R) (t i) (t' i)
          f✝ : (i : ι) → LinearMap (RingHom.id R) (s i) (t i)
          f : (i : ι) → LinearEquiv (RingHom.id R) (s i) (t i)
          ⊢ Eq ((PiTensorProduct.map fun i => ↑(f i).symm).comp (PiTensorProduct.map fun …
        -/
    (by ext; simp)
             /-
               🎉 no goals
             -/


@[simp]
theorem congr_tprod (f : Π i, s i ≃ₗ[R] t i) (m : Π i, s i) :
    congr f (tprod R m) = tprod R (fun (i : ι) ↦ (f i) (m i)) := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁴ : CommSemiring R
    s : ι → Type u_7
    inst✝³ : (i : ι) → AddCommMonoid (s i)
    inst✝² : (i : ι) → Module R (s i)
    t : ι → Type u_11
    inst✝¹ : (i : ι) → AddCommMonoid (t i)
    inst✝ : (i : ι) → Module R (t i)
    f : (i : ι) → LinearEquiv (RingHom.id R) (s i) (t i)
    m : (i : ι) → s i
    ⊢ Eq ((PiTensorProduct.congr f) ((PiTensorProduct.tprod R) m)) ((PiTensorProdu …
  -/
  simp only [congr, LinearEquiv.ofLinear_apply, map_tprod, LinearEquiv.coe_coe]
  /-
    🎉 no goals
  -/


@[simp]
theorem congr_symm_tprod (f : Π i, s i ≃ₗ[R] t i) (p : Π i, t i) :
    (congr f).symm (tprod R p) = tprod R (fun (i : ι) ↦ (f i).symm (p i)) := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁴ : CommSemiring R
    s : ι → Type u_7
    inst✝³ : (i : ι) → AddCommMonoid (s i)
    inst✝² : (i : ι) → Module R (s i)
    t : ι → Type u_11
    inst✝¹ : (i : ι) → AddCommMonoid (t i)
    inst✝ : (i : ι) → Module R (t i)
    f : (i : ι) → LinearEquiv (RingHom.id R) (s i) (t i)
    p : (i : ι) → t i
    ⊢ Eq ((PiTensorProduct.congr f).symm ((PiTensorProduct.tprod R) p)) ((PiTensor …
  -/
  simp only [congr, LinearEquiv.ofLinear_symm_apply, map_tprod, LinearEquiv.coe_coe]
  /-
    🎉 no goals
  -/


/--
Let `sᵢ`, `tᵢ` and `t'ᵢ` be families of `R`-modules, then `f : Πᵢ sᵢ → tᵢ → t'ᵢ` induces an
element of `Hom(⨂ᵢ sᵢ, Hom(⨂ tᵢ, ⨂ᵢ t'ᵢ))` defined by `⨂ᵢ aᵢ ↦ ⨂ᵢ bᵢ ↦ ⨂ᵢ fᵢ aᵢ bᵢ`.

This is `PiTensorProduct.map` for two arbitrary families of modules.
This is `TensorProduct.map₂` for families of modules.
-/
def map₂ (f : Π i, s i →ₗ[R] t i →ₗ[R] t' i) :
    (⨂[R] i, s i) →ₗ[R] (⨂[R] i, t i) →ₗ[R] ⨂[R] i, t' i :=
  lift <| LinearMap.compMultilinearMap piTensorHomMap <| (tprod R).compLinearMap f


lemma map₂_tprod_tprod (f : Π i, s i →ₗ[R] t i →ₗ[R] t' i) (x : Π i, s i) (y : Π i, t i) :
    map₂ f (tprod R x) (tprod R y) = tprod R fun i ↦ f i (x i) (y i) := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁶ : CommSemiring R
    s : ι → Type u_7
    inst✝⁵ : (i : ι) → AddCommMonoid (s i)
    inst✝⁴ : (i : ι) → Module R (s i)
    t : ι → Type u_11
    t' : ι → Type u_12
    inst✝³ : (i : ι) → AddCommMonoid (t i)
    inst✝² : (i : ι) → Module R (t i)
    inst✝¹ : (i : ι) → AddCommMonoid (t' i)
    inst✝ : (i : ι) → Module R (t' i)
    f : (i : ι) → LinearMap (RingHom.id R) (s i) (LinearMap (RingHom.id R) (t i) ( …
    x : (i : ι) → s i
    y : (i : ι) → t i
    ⊢ Eq (((PiTensorProduct.map₂ f) ((PiTensorProduct.tprod R) x)) ((PiTensorProdu …
  -/
  simp [map₂]
  /-
    🎉 no goals
  -/


/--
Let `sᵢ`, `tᵢ` and `t'ᵢ` be families of `R`-modules.
Then there is a function from `⨂ᵢ Hom(sᵢ, Hom(tᵢ, t'ᵢ))` to `Hom(⨂ᵢ sᵢ, Hom(⨂ tᵢ, ⨂ᵢ t'ᵢ))`
defined by `⨂ᵢ fᵢ ↦ ⨂ᵢ aᵢ ↦ ⨂ᵢ bᵢ ↦ ⨂ᵢ fᵢ aᵢ bᵢ`. -/
def piTensorHomMapFun₂ : (⨂[R] i, s i →ₗ[R] t i →ₗ[R] t' i) →
    (⨂[R] i, s i) →ₗ[R] (⨂[R] i, t i) →ₗ[R] (⨂[R] i, t' i) :=
  fun φ => lift <| LinearMap.compMultilinearMap piTensorHomMap <|
    (lift <| MultilinearMap.piLinearMap <| tprod R) φ


theorem piTensorHomMapFun₂_add (φ ψ : ⨂[R] i, s i →ₗ[R] t i →ₗ[R] t' i) :
    piTensorHomMapFun₂ (φ + ψ) = piTensorHomMapFun₂ φ + piTensorHomMapFun₂ ψ := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁶ : CommSemiring R
    s : ι → Type u_7
    inst✝⁵ : (i : ι) → AddCommMonoid (s i)
    inst✝⁴ : (i : ι) → Module R (s i)
    t : ι → Type u_11
    t' : ι → Type u_12
    inst✝³ : (i : ι) → AddCommMonoid (t i)
    inst✝² : (i : ι) → Module R (t i)
    inst✝¹ : (i : ι) → AddCommMonoid (t' i)
    inst✝ : (i : ι) → Module R (t' i)
    φ ψ : PiTensorProduct R fun i => LinearMap (RingHom.id R) (s i) (LinearMap (Ri …
    ⊢ Eq (HAdd.hAdd φ ψ).piTensorHomMapFun₂ (HAdd.hAdd φ.piTensorHomMapFun₂ ψ.piTe …
  -/
  dsimp [piTensorHomMapFun₂]; ext; simp only [map_add, LinearMap.compMultilinearMap_apply,
    lift.tprod, add_apply, LinearMap.add_apply]


theorem piTensorHomMapFun₂_smul (r : R) (φ : ⨂[R] i, s i →ₗ[R] t i →ₗ[R] t' i) :
    piTensorHomMapFun₂ (r • φ) = r • piTensorHomMapFun₂ φ := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁶ : CommSemiring R
    s : ι → Type u_7
    inst✝⁵ : (i : ι) → AddCommMonoid (s i)
    inst✝⁴ : (i : ι) → Module R (s i)
    t : ι → Type u_11
    t' : ι → Type u_12
    inst✝³ : (i : ι) → AddCommMonoid (t i)
    inst✝² : (i : ι) → Module R (t i)
    inst✝¹ : (i : ι) → AddCommMonoid (t' i)
    inst✝ : (i : ι) → Module R (t' i)
    r : R
    φ : PiTensorProduct R fun i => LinearMap (RingHom.id R) (s i) (LinearMap (Ring …
    ⊢ Eq (HSMul.hSMul r φ).piTensorHomMapFun₂ (HSMul.hSMul r φ.piTensorHomMapFun₂)
  -/
  dsimp [piTensorHomMapFun₂]; ext; simp only [map_smul, LinearMap.compMultilinearMap_apply,
    lift.tprod, smul_apply, LinearMap.smul_apply]


/--
Let `sᵢ`, `tᵢ` and `t'ᵢ` be families of `R`-modules.
Then there is an linear map from `⨂ᵢ Hom(sᵢ, Hom(tᵢ, t'ᵢ))` to `Hom(⨂ᵢ sᵢ, Hom(⨂ tᵢ, ⨂ᵢ t'ᵢ))`
defined by `⨂ᵢ fᵢ ↦ ⨂ᵢ aᵢ ↦ ⨂ᵢ bᵢ ↦ ⨂ᵢ fᵢ aᵢ bᵢ`.

This is `TensorProduct.homTensorHomMap` for two arbitrary families of modules.
-/
def piTensorHomMap₂ : (⨂[R] i, s i →ₗ[R] t i →ₗ[R] t' i) →ₗ[R]
    (⨂[R] i, s i) →ₗ[R] (⨂[R] i, t i) →ₗ[R] (⨂[R] i, t' i) where
  toFun := piTensorHomMapFun₂
  map_add' x y := piTensorHomMapFun₂_add x y
  map_smul' x y :=  piTensorHomMapFun₂_smul x y


@[simp] lemma piTensorHomMap₂_tprod_tprod_tprod
    (f : ∀ i, s i →ₗ[R] t i →ₗ[R] t' i) (a : ∀ i, s i) (b : ∀ i, t i) :
    piTensorHomMap₂ (tprod R f) (tprod R a) (tprod R b) = tprod R (fun i ↦ f i (a i) (b i)) := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁶ : CommSemiring R
    s : ι → Type u_7
    inst✝⁵ : (i : ι) → AddCommMonoid (s i)
    inst✝⁴ : (i : ι) → Module R (s i)
    t : ι → Type u_11
    t' : ι → Type u_12
    inst✝³ : (i : ι) → AddCommMonoid (t i)
    inst✝² : (i : ι) → Module R (t i)
    inst✝¹ : (i : ι) → AddCommMonoid (t' i)
    inst✝ : (i : ι) → Module R (t' i)
    f : (i : ι) → LinearMap (RingHom.id R) (s i) (LinearMap (RingHom.id R) (t i) ( …
    a : (i : ι) → s i
    b : (i : ι) → t i
    ⊢ Eq (((PiTensorProduct.piTensorHomMap₂ ((PiTensorProduct.tprod R) f)) ((PiTen …
  -/
  simp [piTensorHomMapFun₂, piTensorHomMap₂]
  /-
    🎉 no goals
  -/


variable (s) in
/-- Re-index the components of the tensor power by `e`. -/
def reindex (e : ι ≃ ι₂) : (⨂[R] i : ι, s i) ≃ₗ[R] ⨂[R] i : ι₂, s (e.symm i) :=
  let f := domDomCongrLinearEquiv' R R s (⨂[R] (i : ι₂), s (e.symm i)) e
  let g := domDomCongrLinearEquiv' R R s (⨂[R] (i : ι), s i) e
  #adaptation_note /-- v4.7.0-rc1
  An alternative to the last two proofs would be `aesop (simp_config := {zetaDelta := true})`
  or a wrapper macro to that effect. -/
  LinearEquiv.ofLinear (lift <| f.symm <| tprod R) (lift <| g <| tprod R)
        /-
          ι : Type u_1
          ι₂ : Type u_2
          ι₃ : Type u_3
          R : Type u_4
          inst✝⁷ : CommSemiring R
          R₁ : Type u_5
          R₂ : Type u_6
          s : ι → Type u_7
          inst✝⁶ : (i : ι) → AddCommMonoid (s i)
          inst✝⁵ : (i : ι) → Module R (s i)
          M : Type u_8
          inst✝⁴ : AddCommMonoid M
          inst✝³ : Module R M
          E : Type u_9
          inst✝² : AddCommMonoid E
          inst✝¹ : Module R E
          F : Type u_10
          inst✝ : AddCommMonoid F
          e : Equiv ι ι₂
          f : LinearEquiv (RingHom.id R) (MultilinearMap R s (PiTensorProduct R fun i => …
          g : LinearEquiv (RingHom.id R) (MultilinearMap R s (PiTensorProduct R fun i => …
          ⊢ Eq ((PiTensorProduct.lift (f.symm (PiTensorProduct.tprod R))).comp (PiTensor …
        -/
    (by aesop (add norm simp [f, g]))
        /-
          🎉 no goals
        -/
        /-
          ι : Type u_1
          ι₂ : Type u_2
          ι₃ : Type u_3
          R : Type u_4
          inst✝⁷ : CommSemiring R
          R₁ : Type u_5
          R₂ : Type u_6
          s : ι → Type u_7
          inst✝⁶ : (i : ι) → AddCommMonoid (s i)
          inst✝⁵ : (i : ι) → Module R (s i)
          M : Type u_8
          inst✝⁴ : AddCommMonoid M
          inst✝³ : Module R M
          E : Type u_9
          inst✝² : AddCommMonoid E
          inst✝¹ : Module R E
          F : Type u_10
          inst✝ : AddCommMonoid F
          e : Equiv ι ι₂
          f : LinearEquiv (RingHom.id R) (MultilinearMap R s (PiTensorProduct R fun i => …
          g : LinearEquiv (RingHom.id R) (MultilinearMap R s (PiTensorProduct R fun i => …
          ⊢ Eq ((PiTensorProduct.lift (g (PiTensorProduct.tprod R))).comp (PiTensorProdu …
        -/
    (by aesop (add norm simp [f, g]))
        /-
          🎉 no goals
        -/


@[simp]
theorem reindex_tprod (e : ι ≃ ι₂) (f : Π i, s i) :
    reindex R s e (tprod R f) = tprod R fun i ↦ f (e.symm i) := by
  /-
    ι : Type u_1
    ι₂ : Type u_2
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    e : Equiv ι ι₂
    f : (i : ι) → s i
    ⊢ Eq ((PiTensorProduct.reindex R s e) ((PiTensorProduct.tprod R) f)) ((PiTenso …
  -/
  dsimp [reindex]
  /-
    ι : Type u_1
    ι₂ : Type u_2
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    e : Equiv ι ι₂
    f : (i : ι) → s i
    ⊢ Eq ((PiTensorProduct.lift { toFun := Function.comp ⇑(PiTensorProduct.tprod R …
  -/
  exact liftAux_tprod _ f
  /-
    🎉 no goals
  -/


@[simp]
theorem reindex_comp_tprod (e : ι ≃ ι₂) :
    (reindex R s e).compMultilinearMap (tprod R) =
    (domDomCongrLinearEquiv' R R s _ e).symm (tprod R) :=
  MultilinearMap.ext <| reindex_tprod e


theorem lift_comp_reindex (e : ι ≃ ι₂) (φ : MultilinearMap R (fun i ↦ s (e.symm i)) E) :
    lift φ ∘ₗ (reindex R s e) = lift ((domDomCongrLinearEquiv' R R s _ e).symm φ) := by
  /-
    ι : Type u_1
    ι₂ : Type u_2
    R : Type u_4
    inst✝⁴ : CommSemiring R
    s : ι → Type u_7
    inst✝³ : (i : ι) → AddCommMonoid (s i)
    inst✝² : (i : ι) → Module R (s i)
    E : Type u_9
    inst✝¹ : AddCommMonoid E
    inst✝ : Module R E
    e : Equiv ι ι₂
    φ : MultilinearMap R (fun i => s (e.symm i)) E
    ⊢ Eq ((PiTensorProduct.lift φ).comp ↑(PiTensorProduct.reindex R s e)) (PiTenso …
  -/
  ext; simp [reindex]
       /-
         🎉 no goals
       -/


@[simp]
theorem lift_comp_reindex_symm (e : ι ≃ ι₂) (φ : MultilinearMap R s E) :
    lift φ ∘ₗ (reindex R s e).symm = lift (domDomCongrLinearEquiv' R R s _ e φ) := by
  /-
    ι : Type u_1
    ι₂ : Type u_2
    R : Type u_4
    inst✝⁴ : CommSemiring R
    s : ι → Type u_7
    inst✝³ : (i : ι) → AddCommMonoid (s i)
    inst✝² : (i : ι) → Module R (s i)
    E : Type u_9
    inst✝¹ : AddCommMonoid E
    inst✝ : Module R E
    e : Equiv ι ι₂
    φ : MultilinearMap R s E
    ⊢ Eq ((PiTensorProduct.lift φ).comp ↑(PiTensorProduct.reindex R s e).symm) (Pi …
  -/
  ext; simp [reindex]
       /-
         🎉 no goals
       -/


theorem lift_reindex
    (e : ι ≃ ι₂) (φ : MultilinearMap R (fun i ↦ s (e.symm i)) E) (x : ⨂[R] i, s i) :
    lift φ (reindex R s e x) = lift ((domDomCongrLinearEquiv' R R s _ e).symm φ) x :=
  LinearMap.congr_fun (lift_comp_reindex e φ) x


@[simp]
theorem lift_reindex_symm
    (e : ι ≃ ι₂) (φ : MultilinearMap R s E) (x : ⨂[R] i, s (e.symm i)) :
    lift φ (reindex R s e |>.symm x) = lift (domDomCongrLinearEquiv' R R s _ e φ) x :=
  LinearMap.congr_fun (lift_comp_reindex_symm e φ) x


@[simp]
theorem reindex_trans (e : ι ≃ ι₂) (e' : ι₂ ≃ ι₃) :
    (reindex R s e).trans (reindex R _ e') = reindex R s (e.trans e') := by
  /-
    ι : Type u_1
    ι₂ : Type u_2
    ι₃ : Type u_3
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    e : Equiv ι ι₂
    e' : Equiv ι₂ ι₃
    ⊢ Eq ((PiTensorProduct.reindex R s e).trans (PiTensorProduct.reindex R (fun i  …
  -/
  apply LinearEquiv.toLinearMap_injective
  /-
    case a
    ι : Type u_1
    ι₂ : Type u_2
    ι₃ : Type u_3
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    e : Equiv ι ι₂
    e' : Equiv ι₂ ι₃
    ⊢ Eq ↑((PiTensorProduct.reindex R s e).trans (PiTensorProduct.reindex R (fun i …
  -/
  ext f
  simp only [LinearEquiv.trans_apply, LinearEquiv.coe_coe, reindex_tprod,
    LinearMap.coe_compMultilinearMap, Function.comp_apply, MultilinearMap.domDomCongr_apply,
    reindex_comp_tprod]
  /-
    case a.H.H
    ι : Type u_1
    ι₂ : Type u_2
    ι₃ : Type u_3
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    e : Equiv ι ι₂
    e' : Equiv ι₂ ι₃
    f : (i : ι) → s i
    ⊢ Eq ((PiTensorProduct.tprod R) fun i => f (e.symm (e'.symm i))) (((Multilinea …
  -/
  congr
  /-
    🎉 no goals
  -/


theorem reindex_reindex (e : ι ≃ ι₂) (e' : ι₂ ≃ ι₃) (x : ⨂[R] i, s i) :
    reindex R _ e' (reindex R s e x) = reindex R s (e.trans e') x :=
  LinearEquiv.congr_fun (reindex_trans e e' : _ = reindex R s (e.trans e')) x


/-- This lemma is impractical to state in the dependent case. -/
@[simp]
theorem reindex_symm (e : ι ≃ ι₂) :
    (reindex R (fun _ ↦ M) e).symm = reindex R (fun _ ↦ M) e.symm := by
  /-
    ι : Type u_1
    ι₂ : Type u_2
    R : Type u_4
    inst✝² : CommSemiring R
    M : Type u_8
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    e : Equiv ι ι₂
    ⊢ Eq (PiTensorProduct.reindex R (fun x => M) e).symm (PiTensorProduct.reindex  …
  -/
  ext x
  simp only [reindex, domDomCongrLinearEquiv', LinearEquiv.coe_symm_mk, LinearEquiv.coe_mk,
    LinearEquiv.ofLinear_symm_apply, Equiv.symm_symm_apply, LinearEquiv.ofLinear_apply,
    Equiv.piCongrLeft'_symm]


@[simp]
theorem reindex_refl : reindex R s (Equiv.refl ι) = LinearEquiv.refl R _ := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    ⊢ Eq (PiTensorProduct.reindex R s (Equiv.refl ι)) (LinearEquiv.refl R (PiTenso …
  -/
  apply LinearEquiv.toLinearMap_injective
  /-
    case a
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    ⊢ Eq ↑(PiTensorProduct.reindex R s (Equiv.refl ι)) ↑(LinearEquiv.refl R (PiTen …
  -/
  ext
  simp only [Equiv.refl_symm, Equiv.refl_apply, reindex, domDomCongrLinearEquiv',
    LinearEquiv.coe_symm_mk, LinearMap.compMultilinearMap_apply, LinearEquiv.coe_coe,
    LinearEquiv.refl_toLinearMap, LinearMap.id_coe, id_eq]
  /-
    case a.H.H
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    x✝ : (i : ι) → s i
    ⊢ Eq ((LinearEquiv.ofLinear (PiTensorProduct.lift { toFun := Function.comp ⇑(P …
  -/
  erw [lift.tprod]
  /-
    case a.H.H
    ι : Type u_1
    R : Type u_4
    inst✝² : CommSemiring R
    s : ι → Type u_7
    inst✝¹ : (i : ι) → AddCommMonoid (s i)
    inst✝ : (i : ι) → Module R (s i)
    x✝ : (i : ι) → s i
    ⊢ Eq ({ toFun := Function.comp ⇑(PiTensorProduct.tprod R) ⇑(Equiv.piCongrLeft' …
  -/
  congr
  /-
    🎉 no goals
  -/


/-- Re-indexing the components of the tensor product by an equivalence `e` is compatible
with `PiTensorProduct.map`. -/
theorem map_comp_reindex_eq (f : Π i, s i →ₗ[R] t i) (e : ι ≃ ι₂) :
    map (fun i ↦ f (e.symm i)) ∘ₗ reindex R s e = reindex R t e ∘ₗ map f := by
  /-
    ι : Type u_1
    ι₂ : Type u_2
    R : Type u_4
    inst✝⁴ : CommSemiring R
    s : ι → Type u_7
    inst✝³ : (i : ι) → AddCommMonoid (s i)
    inst✝² : (i : ι) → Module R (s i)
    t : ι → Type u_11
    inst✝¹ : (i : ι) → AddCommMonoid (t i)
    inst✝ : (i : ι) → Module R (t i)
    f : (i : ι) → LinearMap (RingHom.id R) (s i) (t i)
    e : Equiv ι ι₂
    ⊢ Eq ((PiTensorProduct.map fun i => f (e.symm i)).comp ↑(PiTensorProduct.reind …
  -/
  ext m
  simp only [LinearMap.compMultilinearMap_apply, LinearMap.coe_comp, LinearEquiv.coe_coe,
    LinearMap.comp_apply, reindex_tprod, map_tprod]


theorem map_reindex (f : Π i, s i →ₗ[R] t i) (e : ι ≃ ι₂) (x : ⨂[R] i, s i) :
    map (fun i ↦ f (e.symm i)) (reindex R s e x) = reindex R t e (map f x) :=
  DFunLike.congr_fun (map_comp_reindex_eq _ _) _


theorem map_comp_reindex_symm (f : Π i, s i →ₗ[R] t i) (e : ι ≃ ι₂) :
    map f ∘ₗ (reindex R s e).symm = (reindex R t e).symm ∘ₗ map (fun i => f (e.symm i)) := by
  /-
    ι : Type u_1
    ι₂ : Type u_2
    R : Type u_4
    inst✝⁴ : CommSemiring R
    s : ι → Type u_7
    inst✝³ : (i : ι) → AddCommMonoid (s i)
    inst✝² : (i : ι) → Module R (s i)
    t : ι → Type u_11
    inst✝¹ : (i : ι) → AddCommMonoid (t i)
    inst✝ : (i : ι) → Module R (t i)
    f : (i : ι) → LinearMap (RingHom.id R) (s i) (t i)
    e : Equiv ι ι₂
    ⊢ Eq ((PiTensorProduct.map f).comp ↑(PiTensorProduct.reindex R s e).symm) ((↑( …
  -/
  ext m
  /-
    case H.H
    ι : Type u_1
    ι₂ : Type u_2
    R : Type u_4
    inst✝⁴ : CommSemiring R
    s : ι → Type u_7
    inst✝³ : (i : ι) → AddCommMonoid (s i)
    inst✝² : (i : ι) → Module R (s i)
    t : ι → Type u_11
    inst✝¹ : (i : ι) → AddCommMonoid (t i)
    inst✝ : (i : ι) → Module R (t i)
    f : (i : ι) → LinearMap (RingHom.id R) (s i) (t i)
    e : Equiv ι ι₂
    m : (i : ι₂) → s (e.symm i)
    ⊢ Eq ((((PiTensorProduct.map f).comp ↑(PiTensorProduct.reindex R s e).symm).co …
  -/
  apply LinearEquiv.injective (reindex R t e)
  simp only [LinearMap.compMultilinearMap_apply, LinearMap.coe_comp, LinearEquiv.coe_coe,
    comp_apply, ← map_reindex, LinearEquiv.apply_symm_apply, map_tprod]


theorem map_reindex_symm (f : Π i, s i →ₗ[R] t i) (e : ι ≃ ι₂) (x : ⨂[R] i, s (e.symm i)) :
    map f ((reindex R s e).symm x) = (reindex R t e).symm (map (fun i ↦ f (e.symm i)) x) :=
  DFunLike.congr_fun (map_comp_reindex_symm _ _) _


attribute [local simp] eq_iff_true_of_subsingleton in
/-- The tensor product over an empty index type `ι` is isomorphic to the base ring. -/
@[simps symm_apply]
def isEmptyEquiv [IsEmpty ι] : (⨂[R] i : ι, s i) ≃ₗ[R] R where
  toFun := lift (constOfIsEmpty R _ 1)
  invFun r := r • tprod R (@isEmptyElim _ _ _)
  left_inv x := by
    /-
      ι : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      R : Type u_4
      inst✝¹⁰ : CommSemiring R
      R₁ : Type u_5
      R₂ : Type u_6
      s : ι → Type u_7
      inst✝⁹ : (i : ι) → AddCommMonoid (s i)
      inst✝⁸ : (i : ι) → Module R (s i)
      M : Type u_8
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      E : Type u_9
      inst✝⁵ : AddCommMonoid E
      inst✝⁴ : Module R E
      F : Type u_10
      inst✝³ : AddCommMonoid F
      t : ι → Type u_11
      inst✝² : (i : ι) → AddCommMonoid (t i)
      inst✝¹ : (i : ι) → Module R (t i)
      inst✝ : IsEmpty ι
      x : PiTensorProduct R fun i => s i
      ⊢ Eq ((fun r => HSMul.hSMul r ((PiTensorProduct.tprod R) isEmptyElim)) ({ toFu …
    -/
    refine x.induction_on ?_ ?_
      /-
        case refine_1
        ι : Type u_1
        ι₂ : Type u_2
        ι₃ : Type u_3
        R : Type u_4
        inst✝¹⁰ : CommSemiring R
        R₁ : Type u_5
        R₂ : Type u_6
        s : ι → Type u_7
        inst✝⁹ : (i : ι) → AddCommMonoid (s i)
        inst✝⁸ : (i : ι) → Module R (s i)
        M : Type u_8
        inst✝⁷ : AddCommMonoid M
        inst✝⁶ : Module R M
        E : Type u_9
        inst✝⁵ : AddCommMonoid E
        inst✝⁴ : Module R E
        F : Type u_10
        inst✝³ : AddCommMonoid F
        t : ι → Type u_11
        inst✝² : (i : ι) → AddCommMonoid (t i)
        inst✝¹ : (i : ι) → Module R (t i)
        inst✝ : IsEmpty ι
        x : PiTensorProduct R fun i => s i
        ⊢ ∀ (r : R) (f : (i : ι) → s i), Eq ((fun r => HSMul.hSMul r ((PiTensorProduct …
      -/
    · intro x y
      -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 had to change `map_smulₛₗ` into `map_smulₛₗ _`
      simp only [map_smulₛₗ _, RingHom.id_apply, lift.tprod, constOfIsEmpty_apply, const_apply,
        smul_eq_mul, mul_one]
      /-
        case refine_1
        ι : Type u_1
        ι₂ : Type u_2
        ι₃ : Type u_3
        R : Type u_4
        inst✝¹⁰ : CommSemiring R
        R₁ : Type u_5
        R₂ : Type u_6
        s : ι → Type u_7
        inst✝⁹ : (i : ι) → AddCommMonoid (s i)
        inst✝⁸ : (i : ι) → Module R (s i)
        M : Type u_8
        inst✝⁷ : AddCommMonoid M
        inst✝⁶ : Module R M
        E : Type u_9
        inst✝⁵ : AddCommMonoid E
        inst✝⁴ : Module R E
        F : Type u_10
        inst✝³ : AddCommMonoid F
        t : ι → Type u_11
        inst✝² : (i : ι) → AddCommMonoid (t i)
        inst✝¹ : (i : ι) → Module R (t i)
        inst✝ : IsEmpty ι
        x✝ : PiTensorProduct R fun i => s i
        x : R
        y : (i : ι) → s i
        ⊢ Eq (HSMul.hSMul x ((PiTensorProduct.tprod R) isEmptyElim)) (HSMul.hSMul x (( …
      -/
      congr
      /-
        case refine_1.e_a.h.e_6.h
        ι : Type u_1
        ι₂ : Type u_2
        ι₃ : Type u_3
        R : Type u_4
        inst✝¹⁰ : CommSemiring R
        R₁ : Type u_5
        R₂ : Type u_6
        s : ι → Type u_7
        inst✝⁹ : (i : ι) → AddCommMonoid (s i)
        inst✝⁸ : (i : ι) → Module R (s i)
        M : Type u_8
        inst✝⁷ : AddCommMonoid M
        inst✝⁶ : Module R M
        E : Type u_9
        inst✝⁵ : AddCommMonoid E
        inst✝⁴ : Module R E
        F : Type u_10
        inst✝³ : AddCommMonoid F
        t : ι → Type u_11
        inst✝² : (i : ι) → AddCommMonoid (t i)
        inst✝¹ : (i : ι) → Module R (t i)
        inst✝ : IsEmpty ι
        x✝ : PiTensorProduct R fun i => s i
        x : R
        y : (i : ι) → s i
        ⊢ Eq isEmptyElim y
      -/
      aesop
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        ι : Type u_1
        ι₂ : Type u_2
        ι₃ : Type u_3
        R : Type u_4
        inst✝¹⁰ : CommSemiring R
        R₁ : Type u_5
        R₂ : Type u_6
        s : ι → Type u_7
        inst✝⁹ : (i : ι) → AddCommMonoid (s i)
        inst✝⁸ : (i : ι) → Module R (s i)
        M : Type u_8
        inst✝⁷ : AddCommMonoid M
        inst✝⁶ : Module R M
        E : Type u_9
        inst✝⁵ : AddCommMonoid E
        inst✝⁴ : Module R E
        F : Type u_10
        inst✝³ : AddCommMonoid F
        t : ι → Type u_11
        inst✝² : (i : ι) → AddCommMonoid (t i)
        inst✝¹ : (i : ι) → Module R (t i)
        inst✝ : IsEmpty ι
        x : PiTensorProduct R fun i => s i
        ⊢ ∀ (x y : PiTensorProduct R fun i => s i), Eq ((fun r => HSMul.hSMul r ((PiTe …
      -/
    /-
      ι : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      R : Type u_4
      inst✝¹⁰ : CommSemiring R
      R₁ : Type u_5
      R₂ : Type u_6
      s : ι → Type u_7
      inst✝⁹ : (i : ι) → AddCommMonoid (s i)
      inst✝⁸ : (i : ι) → Module R (s i)
      M : Type u_8
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      E : Type u_9
      inst✝⁵ : AddCommMonoid E
      inst✝⁴ : Module R E
      F : Type u_10
      inst✝³ : AddCommMonoid F
      t : ι → Type u_11
      inst✝² : (i : ι) → AddCommMonoid (t i)
      inst✝¹ : (i : ι) → Module R (t i)
      inst✝ : IsEmpty ι
      r : R
      x : PiTensorProduct R fun i => s i
      ⊢ Eq ({ toFun := ⇑(PiTensorProduct.lift (MultilinearMap.constOfIsEmpty R s 1)) …
    -/
    · simp only
    /-
      ι : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      R : Type u_4
      inst✝¹⁰ : CommSemiring R
      R₁ : Type u_5
      R₂ : Type u_6
      s : ι → Type u_7
      inst✝⁹ : (i : ι) → AddCommMonoid (s i)
      inst✝⁸ : (i : ι) → Module R (s i)
      M : Type u_8
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      E : Type u_9
      inst✝⁵ : AddCommMonoid E
      inst✝⁴ : Module R E
      F : Type u_10
      inst✝³ : AddCommMonoid F
      t : ι → Type u_11
      inst✝² : (i : ι) → AddCommMonoid (t i)
      inst✝¹ : (i : ι) → Module R (t i)
      inst✝ : IsEmpty ι
      r : R
      x : PiTensorProduct R fun i => s i
      ⊢ Eq ((PiTensorProduct.lift (MultilinearMap.constOfIsEmpty R s 1)) (HSMul.hSMu …
    -/
      /-
        case refine_2
        ι : Type u_1
        ι₂ : Type u_2
        ι₃ : Type u_3
        R : Type u_4
        inst✝¹⁰ : CommSemiring R
        R₁ : Type u_5
        R₂ : Type u_6
        s : ι → Type u_7
        inst✝⁹ : (i : ι) → AddCommMonoid (s i)
        inst✝⁸ : (i : ι) → Module R (s i)
        M : Type u_8
        inst✝⁷ : AddCommMonoid M
        inst✝⁶ : Module R M
        E : Type u_9
        inst✝⁵ : AddCommMonoid E
        inst✝⁴ : Module R E
        F : Type u_10
        inst✝³ : AddCommMonoid F
        t : ι → Type u_11
        inst✝² : (i : ι) → AddCommMonoid (t i)
        inst✝¹ : (i : ι) → Module R (t i)
        inst✝ : IsEmpty ι
        x : PiTensorProduct R fun i => s i
        ⊢ ∀ (x y : PiTensorProduct R fun i => s i), Eq (HSMul.hSMul ((PiTensorProduct. …
      -/
    /-
      🎉 no goals
    -/
      intro x y hx hy
      /-
        case refine_2
        ι : Type u_1
        ι₂ : Type u_2
        ι₃ : Type u_3
        R : Type u_4
        inst✝¹⁰ : CommSemiring R
        R₁ : Type u_5
        R₂ : Type u_6
        s : ι → Type u_7
        inst✝⁹ : (i : ι) → AddCommMonoid (s i)
        inst✝⁸ : (i : ι) → Module R (s i)
        M : Type u_8
        inst✝⁷ : AddCommMonoid M
        inst✝⁶ : Module R M
        E : Type u_9
        inst✝⁵ : AddCommMonoid E
        inst✝⁴ : Module R E
        F : Type u_10
        inst✝³ : AddCommMonoid F
        t : ι → Type u_11
        inst✝² : (i : ι) → AddCommMonoid (t i)
        inst✝¹ : (i : ι) → Module R (t i)
        inst✝ : IsEmpty ι
        x✝ x y : PiTensorProduct R fun i => s i
        hx : Eq (HSMul.hSMul ((PiTensorProduct.lift (MultilinearMap.constOfIsEmpty R s …
        hy : Eq (HSMul.hSMul ((PiTensorProduct.lift (MultilinearMap.constOfIsEmpty R s …
        ⊢ Eq (HSMul.hSMul ((PiTensorProduct.lift (MultilinearMap.constOfIsEmpty R s 1) …
      -/
      rw [map_add, add_smul, hx, hy]
      /-
        🎉 no goals
      -/
                    /-
                      ι : Type u_1
                      ι₂ : Type u_2
                      ι₃ : Type u_3
                      R : Type u_4
                      inst✝¹⁰ : CommSemiring R
                      R₁ : Type u_5
                      R₂ : Type u_6
                      s : ι → Type u_7
                      inst✝⁹ : (i : ι) → AddCommMonoid (s i)
                      inst✝⁸ : (i : ι) → Module R (s i)
                      M : Type u_8
                      inst✝⁷ : AddCommMonoid M
                      inst✝⁶ : Module R M
                      E : Type u_9
                      inst✝⁵ : AddCommMonoid E
                      inst✝⁴ : Module R E
                      F : Type u_10
                      inst✝³ : AddCommMonoid F
                      t✝ : ι → Type u_11
                      inst✝² : (i : ι) → AddCommMonoid (t✝ i)
                      inst✝¹ : (i : ι) → Module R (t✝ i)
                      inst✝ : IsEmpty ι
                      t : R
                      ⊢ Eq ({ toFun := ⇑(PiTensorProduct.lift (MultilinearMap.constOfIsEmpty R s 1)) …
                    -/
  right_inv t := by simp
                    /-
                      🎉 no goals
                    -/
  map_add' := LinearMap.map_add _
  map_smul' := fun r x => by
    simp only
    exact LinearMap.map_smul _ r x


@[simp]
theorem isEmptyEquiv_apply_tprod [IsEmpty ι] (f : Π i, s i) :
    isEmptyEquiv ι (tprod R f) = 1 :=
  lift.tprod _


/--
Tensor product of `M` over a singleton set is equivalent to `M`
-/
@[simps symm_apply]
def subsingletonEquiv [Subsingleton ι] (i₀ : ι) : (⨂[R] _ : ι, M) ≃ₗ[R] M where
  toFun := lift (MultilinearMap.ofSubsingleton R M M i₀ .id)
  invFun m := tprod R fun _ ↦ m
  left_inv x := by
    /-
      ι : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      R : Type u_4
      inst✝¹⁰ : CommSemiring R
      R₁ : Type u_5
      R₂ : Type u_6
      s : ι → Type u_7
      inst✝⁹ : (i : ι) → AddCommMonoid (s i)
      inst✝⁸ : (i : ι) → Module R (s i)
      M : Type u_8
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      E : Type u_9
      inst✝⁵ : AddCommMonoid E
      inst✝⁴ : Module R E
      F : Type u_10
      inst✝³ : AddCommMonoid F
      t : ι → Type u_11
      inst✝² : (i : ι) → AddCommMonoid (t i)
      inst✝¹ : (i : ι) → Module R (t i)
      inst✝ : Subsingleton ι
      i₀ : ι
      x : PiTensorProduct R fun x => M
      ⊢ Eq ((fun m => (PiTensorProduct.tprod R) fun x => m) ({ toFun := ⇑(PiTensorPr …
    -/
    dsimp only
    have : ∀ (f : ι → M) (z : M), (fun _ : ι ↦ z) = update f i₀ z := fun f z ↦ by
      ext i
      rw [Subsingleton.elim i i₀, Function.update_self]
    /-
      ι : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      R : Type u_4
      inst✝¹⁰ : CommSemiring R
      R₁ : Type u_5
      R₂ : Type u_6
      s : ι → Type u_7
      inst✝⁹ : (i : ι) → AddCommMonoid (s i)
      inst✝⁸ : (i : ι) → Module R (s i)
      M : Type u_8
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      E : Type u_9
      inst✝⁵ : AddCommMonoid E
      inst✝⁴ : Module R E
      F : Type u_10
      inst✝³ : AddCommMonoid F
      t : ι → Type u_11
      inst✝² : (i : ι) → AddCommMonoid (t i)
      inst✝¹ : (i : ι) → Module R (t i)
      inst✝ : Subsingleton ι
      i₀ : ι
      x : PiTensorProduct R fun x => M
      this : ∀ (f : ι → M) (z : M), Eq (fun x => z) (Function.update f i₀ z)
      ⊢ Eq ((PiTensorProduct.tprod R) fun x_1 => (PiTensorProduct.lift ((Multilinear …
    -/
    refine x.induction_on ?_ ?_
      /-
        case refine_1
        ι : Type u_1
        ι₂ : Type u_2
        ι₃ : Type u_3
        R : Type u_4
        inst✝¹⁰ : CommSemiring R
        R₁ : Type u_5
        R₂ : Type u_6
        s : ι → Type u_7
        inst✝⁹ : (i : ι) → AddCommMonoid (s i)
        inst✝⁸ : (i : ι) → Module R (s i)
        M : Type u_8
        inst✝⁷ : AddCommMonoid M
        inst✝⁶ : Module R M
        E : Type u_9
        inst✝⁵ : AddCommMonoid E
        inst✝⁴ : Module R E
        F : Type u_10
        inst✝³ : AddCommMonoid F
        t : ι → Type u_11
        inst✝² : (i : ι) → AddCommMonoid (t i)
        inst✝¹ : (i : ι) → Module R (t i)
        inst✝ : Subsingleton ι
        i₀ : ι
        x : PiTensorProduct R fun x => M
        this : ∀ (f : ι → M) (z : M), Eq (fun x => z) (Function.update f i₀ z)
        ⊢ ∀ (r : R) (f : ι → M), Eq ((PiTensorProduct.tprod R) fun x => (PiTensorProdu …
      -/
    · intro r f
      simp only [LinearMap.map_smul, LinearMap.id_apply, lift.tprod, ofSubsingleton_apply_apply,
        this f, MultilinearMap.map_update_smul, update_eq_self]
      /-
        case refine_2
        ι : Type u_1
        ι₂ : Type u_2
        ι₃ : Type u_3
        R : Type u_4
        inst✝¹⁰ : CommSemiring R
        R₁ : Type u_5
        R₂ : Type u_6
        s : ι → Type u_7
        inst✝⁹ : (i : ι) → AddCommMonoid (s i)
        inst✝⁸ : (i : ι) → Module R (s i)
        M : Type u_8
        inst✝⁷ : AddCommMonoid M
        inst✝⁶ : Module R M
        E : Type u_9
        inst✝⁵ : AddCommMonoid E
        inst✝⁴ : Module R E
        F : Type u_10
        inst✝³ : AddCommMonoid F
        t : ι → Type u_11
        inst✝² : (i : ι) → AddCommMonoid (t i)
        inst✝¹ : (i : ι) → Module R (t i)
        inst✝ : Subsingleton ι
        i₀ : ι
        x : PiTensorProduct R fun x => M
        this : ∀ (f : ι → M) (z : M), Eq (fun x => z) (Function.update f i₀ z)
        ⊢ ∀ (x y : PiTensorProduct R fun i => M), Eq ((PiTensorProduct.tprod R) fun x_ …
      -/
    · intro x y hx hy
      rw [LinearMap.map_add, this 0 (_ + _), MultilinearMap.map_update_add, ← this 0 (lift _ _), hx,
    /-
      ι : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      R : Type u_4
      inst✝¹⁰ : CommSemiring R
      R₁ : Type u_5
      R₂ : Type u_6
      s : ι → Type u_7
      inst✝⁹ : (i : ι) → AddCommMonoid (s i)
      inst✝⁸ : (i : ι) → Module R (s i)
      M : Type u_8
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      E : Type u_9
      inst✝⁵ : AddCommMonoid E
      inst✝⁴ : Module R E
      F : Type u_10
      inst✝³ : AddCommMonoid F
      t : ι → Type u_11
      inst✝² : (i : ι) → AddCommMonoid (t i)
      inst✝¹ : (i : ι) → Module R (t i)
      inst✝ : Subsingleton ι
      i₀ : ι
      r : R
      x : PiTensorProduct R fun x => M
      ⊢ Eq ({ toFun := ⇑(PiTensorProduct.lift ((MultilinearMap.ofSubsingleton R M M  …
    -/
        ← this 0 (lift _ _), hy]
    /-
      ι : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      R : Type u_4
      inst✝¹⁰ : CommSemiring R
      R₁ : Type u_5
      R₂ : Type u_6
      s : ι → Type u_7
      inst✝⁹ : (i : ι) → AddCommMonoid (s i)
      inst✝⁸ : (i : ι) → Module R (s i)
      M : Type u_8
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      E : Type u_9
      inst✝⁵ : AddCommMonoid E
      inst✝⁴ : Module R E
      F : Type u_10
      inst✝³ : AddCommMonoid F
      t : ι → Type u_11
      inst✝² : (i : ι) → AddCommMonoid (t i)
      inst✝¹ : (i : ι) → Module R (t i)
      inst✝ : Subsingleton ι
      i₀ : ι
      r : R
      x : PiTensorProduct R fun x => M
      ⊢ Eq ((PiTensorProduct.lift ((MultilinearMap.ofSubsingleton R M M i₀) LinearMa …
    -/
                    /-
                      ι : Type u_1
                      ι₂ : Type u_2
                      ι₃ : Type u_3
                      R : Type u_4
                      inst✝¹⁰ : CommSemiring R
                      R₁ : Type u_5
                      R₂ : Type u_6
                      s : ι → Type u_7
                      inst✝⁹ : (i : ι) → AddCommMonoid (s i)
                      inst✝⁸ : (i : ι) → Module R (s i)
                      M : Type u_8
                      inst✝⁷ : AddCommMonoid M
                      inst✝⁶ : Module R M
                      E : Type u_9
                      inst✝⁵ : AddCommMonoid E
                      inst✝⁴ : Module R E
                      F : Type u_10
                      inst✝³ : AddCommMonoid F
                      t✝ : ι → Type u_11
                      inst✝² : (i : ι) → AddCommMonoid (t✝ i)
                      inst✝¹ : (i : ι) → Module R (t✝ i)
                      inst✝ : Subsingleton ι
                      i₀ : ι
                      t : M
                      ⊢ Eq ({ toFun := ⇑(PiTensorProduct.lift ((MultilinearMap.ofSubsingleton R M M  …
                    -/
    /-
      🎉 no goals
    -/
  right_inv t := by simp only [ofSubsingleton_apply_apply, LinearMap.id_apply, lift.tprod]
                    /-
                      🎉 no goals
                    -/
  map_add' := LinearMap.map_add _
  map_smul' := fun r x => by
    simp only
    exact LinearMap.map_smul _ r x


@[simp]
theorem subsingletonEquiv_apply_tprod [Subsingleton ι] (i : ι) (f : ι → M) :
    subsingletonEquiv i (tprod R f) = f i :=
  lift.tprod _


/-- Collapse a `TensorProduct` of `PiTensorProduct`s. -/
private def tmul : ((⨂[R] _ : ι, M) ⊗[R] ⨂[R] _ : ι₂, M) →ₗ[R] ⨂[R] _ : ι ⊕ ι₂, M :=
  TensorProduct.lift
    { toFun := fun a ↦
        PiTensorProduct.lift <|
          PiTensorProduct.lift (MultilinearMap.currySumEquiv R _ _ M _ (tprod R)) a
                               /-
                                 ι : Type u_1
                                 ι₂ : Type u_2
                                 ι₃ : Type u_3
                                 R : Type u_4
                                 inst✝⁹ : CommSemiring R
                                 R₁ : Type u_5
                                 R₂ : Type u_6
                                 s : ι → Type u_7
                                 inst✝⁸ : (i : ι) → AddCommMonoid (s i)
                                 inst✝⁷ : (i : ι) → Module R (s i)
                                 M : Type u_8
                                 inst✝⁶ : AddCommMonoid M
                                 inst✝⁵ : Module R M
                                 E : Type u_9
                                 inst✝⁴ : AddCommMonoid E
                                 inst✝³ : Module R E
                                 F : Type u_10
                                 inst✝² : AddCommMonoid F
                                 t : ι → Type u_11
                                 inst✝¹ : (i : ι) → AddCommMonoid (t i)
                                 inst✝ : (i : ι) → Module R (t i)
                                 a b : PiTensorProduct R fun x => M
                                 ⊢ Eq ((fun a => PiTensorProduct.lift ((PiTensorProduct.lift ((MultilinearMap.c …
                               -/
      map_add' := fun a b ↦ by simp only [LinearEquiv.map_add, LinearMap.map_add]
                               /-
                                 🎉 no goals
                               -/
      map_smul' := fun r a ↦ by
        /-
          ι : Type u_1
          ι₂ : Type u_2
          ι₃ : Type u_3
          R : Type u_4
          inst✝⁹ : CommSemiring R
          R₁ : Type u_5
          R₂ : Type u_6
          s : ι → Type u_7
          inst✝⁸ : (i : ι) → AddCommMonoid (s i)
          inst✝⁷ : (i : ι) → Module R (s i)
          M : Type u_8
          inst✝⁶ : AddCommMonoid M
          inst✝⁵ : Module R M
          E : Type u_9
          inst✝⁴ : AddCommMonoid E
          inst✝³ : Module R E
          F : Type u_10
          inst✝² : AddCommMonoid F
          t : ι → Type u_11
          inst✝¹ : (i : ι) → AddCommMonoid (t i)
          inst✝ : (i : ι) → Module R (t i)
          r : R
          a : PiTensorProduct R fun x => M
          ⊢ Eq ({ toFun := fun a => PiTensorProduct.lift ((PiTensorProduct.lift ((Multil …
        -/
        simp only [LinearEquiv.map_smul, LinearMap.map_smul, RingHom.id_apply] }
        /-
          🎉 no goals
        -/


private theorem tmul_apply (a : ι → M) (b : ι₂ → M) :
    tmul ((⨂ₜ[R] i, a i) ⊗ₜ[R] ⨂ₜ[R] i, b i) = ⨂ₜ[R] i, Sum.elim a b i := by
  /-
    ι : Type u_1
    ι₂ : Type u_2
    R : Type u_4
    inst✝² : CommSemiring R
    M : Type u_8
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    a : ι → M
    b : ι₂ → M
    ⊢ Eq (PiTensorProduct.tmul (TensorProduct.tmul R ((PiTensorProduct.tprod R) fu …
  -/
  erw [TensorProduct.lift.tmul, PiTensorProduct.lift.tprod, PiTensorProduct.lift.tprod]
  /-
    ι : Type u_1
    ι₂ : Type u_2
    R : Type u_4
    inst✝² : CommSemiring R
    M : Type u_8
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    a : ι → M
    b : ι₂ → M
    ⊢ Eq ((((MultilinearMap.currySumEquiv R ι (PiTensorProduct R fun i => M) M ι₂) …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Expand `PiTensorProduct` into a `TensorProduct` of two factors. -/
private def tmulSymm : (⨂[R] _ : ι ⊕ ι₂, M) →ₗ[R] (⨂[R] _ : ι, M) ⊗[R] ⨂[R] _ : ι₂, M :=
  -- by using tactic mode, we avoid the need for a lot of `@`s and `_`s
  PiTensorProduct.lift <| MultilinearMap.domCoprod (tprod R) (tprod R)


private theorem tmulSymm_apply (a : ι ⊕ ι₂ → M) :
    tmulSymm (⨂ₜ[R] i, a i) = (⨂ₜ[R] i, a (Sum.inl i)) ⊗ₜ[R] ⨂ₜ[R] i, a (Sum.inr i) :=
  PiTensorProduct.lift.tprod _


/-- Equivalence between a `TensorProduct` of `PiTensorProduct`s and a single
`PiTensorProduct` indexed by a `Sum` type.

For simplicity, this is defined only for homogeneously- (rather than dependently-) typed components.
-/
def tmulEquiv : ((⨂[R] _ : ι, M) ⊗[R] ⨂[R] _ : ι₂, M) ≃ₗ[R] ⨂[R] _ : ι ⊕ ι₂, M :=
  LinearEquiv.ofLinear tmul tmulSymm
    (by
      /-
        ι : Type u_1
        ι₂ : Type u_2
        ι₃ : Type u_3
        R : Type u_4
        inst✝⁹ : CommSemiring R
        R₁ : Type u_5
        R₂ : Type u_6
        s : ι → Type u_7
        inst✝⁸ : (i : ι) → AddCommMonoid (s i)
        inst✝⁷ : (i : ι) → Module R (s i)
        M : Type u_8
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        E : Type u_9
        inst✝⁴ : AddCommMonoid E
        inst✝³ : Module R E
        F : Type u_10
        inst✝² : AddCommMonoid F
        t : ι → Type u_11
        inst✝¹ : (i : ι) → AddCommMonoid (t i)
        inst✝ : (i : ι) → Module R (t i)
        ⊢ Eq (PiTensorProduct.tmul.comp PiTensorProduct.tmulSymm) LinearMap.id
      -/
      ext x
      /-
        case H.H
        ι : Type u_1
        ι₂ : Type u_2
        ι₃ : Type u_3
        R : Type u_4
        inst✝⁹ : CommSemiring R
        R₁ : Type u_5
        R₂ : Type u_6
        s : ι → Type u_7
        inst✝⁸ : (i : ι) → AddCommMonoid (s i)
        inst✝⁷ : (i : ι) → Module R (s i)
        M : Type u_8
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        E : Type u_9
        inst✝⁴ : AddCommMonoid E
        inst✝³ : Module R E
        F : Type u_10
        inst✝² : AddCommMonoid F
        t : ι → Type u_11
        inst✝¹ : (i : ι) → AddCommMonoid (t i)
        inst✝ : (i : ι) → Module R (t i)
        x : Sum ι ι₂ → M
        ⊢ Eq (((PiTensorProduct.tmul.comp PiTensorProduct.tmulSymm).compMultilinearMap …
      -/
      show tmul (tmulSymm (tprod R x)) = tprod R x -- Speed up the call to `simp`.
      /-
        case H.H
        ι : Type u_1
        ι₂ : Type u_2
        ι₃ : Type u_3
        R : Type u_4
        inst✝⁹ : CommSemiring R
        R₁ : Type u_5
        R₂ : Type u_6
        s : ι → Type u_7
        inst✝⁸ : (i : ι) → AddCommMonoid (s i)
        inst✝⁷ : (i : ι) → Module R (s i)
        M : Type u_8
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        E : Type u_9
        inst✝⁴ : AddCommMonoid E
        inst✝³ : Module R E
        F : Type u_10
        inst✝² : AddCommMonoid F
        t : ι → Type u_11
        inst✝¹ : (i : ι) → AddCommMonoid (t i)
        inst✝ : (i : ι) → Module R (t i)
        x : Sum ι ι₂ → M
        ⊢ Eq (PiTensorProduct.tmul (PiTensorProduct.tmulSymm ((PiTensorProduct.tprod R …
      -/
      simp only [tmulSymm_apply, tmul_apply]
      -- Porting note (https://github.com/leanprover-community/mathlib4/issues/5026):
      -- was part of `simp only` above
      /-
        case H.H
        ι : Type u_1
        ι₂ : Type u_2
        ι₃ : Type u_3
        R : Type u_4
        inst✝⁹ : CommSemiring R
        R₁ : Type u_5
        R₂ : Type u_6
        s : ι → Type u_7
        inst✝⁸ : (i : ι) → AddCommMonoid (s i)
        inst✝⁷ : (i : ι) → Module R (s i)
        M : Type u_8
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        E : Type u_9
        inst✝⁴ : AddCommMonoid E
        inst✝³ : Module R E
        F : Type u_10
        inst✝² : AddCommMonoid F
        t : ι → Type u_11
        inst✝¹ : (i : ι) → AddCommMonoid (t i)
        inst✝ : (i : ι) → Module R (t i)
        x : Sum ι ι₂ → M
        ⊢ Eq ((PiTensorProduct.tprod R) fun i => Sum.elim (fun i => x (Sum.inl i)) (fu …
      -/
      erw [Sum.elim_comp_inl_inr])
      /-
        🎉 no goals
      -/
    (by
      /-
        ι : Type u_1
        ι₂ : Type u_2
        ι₃ : Type u_3
        R : Type u_4
        inst✝⁹ : CommSemiring R
        R₁ : Type u_5
        R₂ : Type u_6
        s : ι → Type u_7
        inst✝⁸ : (i : ι) → AddCommMonoid (s i)
        inst✝⁷ : (i : ι) → Module R (s i)
        M : Type u_8
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        E : Type u_9
        inst✝⁴ : AddCommMonoid E
        inst✝³ : Module R E
        F : Type u_10
        inst✝² : AddCommMonoid F
        t : ι → Type u_11
        inst✝¹ : (i : ι) → AddCommMonoid (t i)
        inst✝ : (i : ι) → Module R (t i)
        ⊢ Eq (PiTensorProduct.tmulSymm.comp PiTensorProduct.tmul) LinearMap.id
      -/
      ext x y
      /-
        case H.H.H.H.H
        ι : Type u_1
        ι₂ : Type u_2
        ι₃ : Type u_3
        R : Type u_4
        inst✝⁹ : CommSemiring R
        R₁ : Type u_5
        R₂ : Type u_6
        s : ι → Type u_7
        inst✝⁸ : (i : ι) → AddCommMonoid (s i)
        inst✝⁷ : (i : ι) → Module R (s i)
        M : Type u_8
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        E : Type u_9
        inst✝⁴ : AddCommMonoid E
        inst✝³ : Module R E
        F : Type u_10
        inst✝² : AddCommMonoid F
        t : ι → Type u_11
        inst✝¹ : (i : ι) → AddCommMonoid (t i)
        inst✝ : (i : ι) → Module R (t i)
        x : ι → M
        y : ι₂ → M
        ⊢ Eq ((((((TensorProduct.mk R (PiTensorProduct R fun x => M) (PiTensorProduct  …
      -/
      show tmulSymm (tmul (tprod R x ⊗ₜ[R] tprod R y)) = tprod R x ⊗ₜ[R] tprod R y
      /-
        case H.H.H.H.H
        ι : Type u_1
        ι₂ : Type u_2
        ι₃ : Type u_3
        R : Type u_4
        inst✝⁹ : CommSemiring R
        R₁ : Type u_5
        R₂ : Type u_6
        s : ι → Type u_7
        inst✝⁸ : (i : ι) → AddCommMonoid (s i)
        inst✝⁷ : (i : ι) → Module R (s i)
        M : Type u_8
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        E : Type u_9
        inst✝⁴ : AddCommMonoid E
        inst✝³ : Module R E
        F : Type u_10
        inst✝² : AddCommMonoid F
        t : ι → Type u_11
        inst✝¹ : (i : ι) → AddCommMonoid (t i)
        inst✝ : (i : ι) → Module R (t i)
        x : ι → M
        y : ι₂ → M
        ⊢ Eq (PiTensorProduct.tmulSymm (PiTensorProduct.tmul (TensorProduct.tmul R ((P …
      -/
      simp only [tmul_apply, tmulSymm_apply, Sum.elim_inl, Sum.elim_inr])
      /-
        🎉 no goals
      -/


@[simp]
theorem tmulEquiv_apply (a : ι → M) (b : ι₂ → M) :
    tmulEquiv (ι := ι) (ι₂ := ι₂) R M ((⨂ₜ[R] i, a i) ⊗ₜ[R] ⨂ₜ[R] i, b i) =
    ⨂ₜ[R] i, Sum.elim a b i :=
  tmul_apply a b


@[simp]
theorem tmulEquiv_symm_apply (a : ι ⊕ ι₂ → M) :
    (tmulEquiv (ι := ι) (ι₂ := ι₂) R M).symm (⨂ₜ[R] i, a i) =
    (⨂ₜ[R] i, a (Sum.inl i)) ⊗ₜ[R] ⨂ₜ[R] i, a (Sum.inr i) :=
  tmulSymm_apply a


instance : AddCommGroup (⨂[R] i, s i) :=
  Module.addCommMonoidToAddCommGroup R


