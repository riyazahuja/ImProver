/-- These are `SlashInvariantForm`'s that are holomorphic and bounded at infinity. -/
structure ModularForm extends SlashInvariantForm Γ k where
  holo' : MDifferentiable 𝓘(ℂ) 𝓘(ℂ) (toSlashInvariantForm : ℍ → ℂ)
  bdd_at_infty' : ∀ A : SL(2, ℤ), IsBoundedAtImInfty (toSlashInvariantForm ∣[k] A)


/-- These are `SlashInvariantForm`s that are holomorphic and zero at infinity. -/
structure CuspForm extends SlashInvariantForm Γ k where
  holo' : MDifferentiable 𝓘(ℂ) 𝓘(ℂ) (toSlashInvariantForm : ℍ → ℂ)
  zero_at_infty' : ∀ A : SL(2, ℤ), IsZeroAtImInfty (toSlashInvariantForm ∣[k] A)


/-- `ModularFormClass F Γ k` says that `F` is a type of bundled functions that extend
`SlashInvariantFormClass` by requiring that the functions be holomorphic and bounded
at infinity. -/
class ModularFormClass (F : Type*) (Γ : outParam <| Subgroup (SL(2, ℤ))) (k : outParam ℤ)
    [FunLike F ℍ ℂ] extends SlashInvariantFormClass F Γ k : Prop where
  holo : ∀ f : F, MDifferentiable 𝓘(ℂ) 𝓘(ℂ) (f : ℍ → ℂ)
  bdd_at_infty : ∀ (f : F) (A : SL(2, ℤ)), IsBoundedAtImInfty (f ∣[k] A)


/-- `CuspFormClass F Γ k` says that `F` is a type of bundled functions that extend
`SlashInvariantFormClass` by requiring that the functions be holomorphic and zero
at infinity. -/
class CuspFormClass (F : Type*) (Γ : outParam <| Subgroup (SL(2, ℤ))) (k : outParam ℤ)
    [FunLike F ℍ ℂ] extends SlashInvariantFormClass F Γ k : Prop where
  holo : ∀ f : F, MDifferentiable 𝓘(ℂ) 𝓘(ℂ) (f : ℍ → ℂ)
  zero_at_infty : ∀ (f : F) (A : SL(2, ℤ)), IsZeroAtImInfty (f ∣[k] A)


instance (priority := 100) ModularForm.funLike :
    FunLike (ModularForm Γ k) ℍ ℂ where
  coe f := f.toFun
                             /-
                               F : Type u_1
                               Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
                               k : Int
                               f g : ModularForm Γ k
                               h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; congr; exact DFunLike.ext' h
                                                      /-
                                                        🎉 no goals
                                                      -/


instance (priority := 100) ModularFormClass.modularForm :
    ModularFormClass (ModularForm Γ k) Γ k where
  slash_action_eq f := f.slash_action_eq'
  holo := ModularForm.holo'
  bdd_at_infty := ModularForm.bdd_at_infty'


instance (priority := 100) CuspForm.funLike : FunLike (CuspForm Γ k) ℍ ℂ where
  coe f := f.toFun
                             /-
                               F : Type u_1
                               Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
                               k : Int
                               f g : CuspForm Γ k
                               h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; congr; exact DFunLike.ext' h
                                                      /-
                                                        🎉 no goals
                                                      -/


instance (priority := 100) CuspFormClass.cuspForm : CuspFormClass (CuspForm Γ k) Γ k where
  slash_action_eq f := f.slash_action_eq'
  holo := CuspForm.holo'
  zero_at_infty := CuspForm.zero_at_infty'


theorem ModularForm.toFun_eq_coe (f : ModularForm Γ k) : f.toFun = (f : ℍ → ℂ) :=
  rfl


@[simp]
theorem ModularForm.toSlashInvariantForm_coe (f : ModularForm Γ k) : ⇑f.1 = f :=
  rfl


theorem CuspForm.toFun_eq_coe {f : CuspForm Γ k} : f.toFun = (f : ℍ → ℂ) :=
  rfl


@[simp]
theorem CuspForm.toSlashInvariantForm_coe (f : CuspForm Γ k) : ⇑f.1 = f := rfl


@[ext]
theorem ModularForm.ext {f g : ModularForm Γ k} (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext f g h


@[ext]
theorem CuspForm.ext {f g : CuspForm Γ k} (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext f g h


/-- Copy of a `ModularForm` with a new `toFun` equal to the old one. Useful to fix
definitional equalities. -/
protected def ModularForm.copy (f : ModularForm Γ k) (f' : ℍ → ℂ) (h : f' = ⇑f) :
    ModularForm Γ k where
  toSlashInvariantForm := f.1.copy f' h
  holo' := h.symm ▸ f.holo'
  bdd_at_infty' A := h.symm ▸ f.bdd_at_infty' A


/-- Copy of a `CuspForm` with a new `toFun` equal to the old one. Useful to fix
definitional equalities. -/
protected def CuspForm.copy (f : CuspForm Γ k) (f' : ℍ → ℂ) (h : f' = ⇑f) : CuspForm Γ k where
  toSlashInvariantForm := f.1.copy f' h
  holo' := h.symm ▸ f.holo'
  zero_at_infty' A := h.symm ▸ f.zero_at_infty' A


instance add : Add (ModularForm Γ k) :=
  ⟨fun f g =>
    { toSlashInvariantForm := f + g
      holo' := f.holo'.add g.holo'
                                   /-
                                     Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
                                     k : Int
                                     f g : ModularForm Γ k
                                     A : Matrix.SpecialLinearGroup (Fin 2) Int
                                     ⊢ UpperHalfPlane.IsBoundedAtImInfty (SlashAction.map Complex k A ⇑(HAdd.hAdd { …
                                   -/
      bdd_at_infty' := fun A => by simpa using (f.bdd_at_infty' A).add (g.bdd_at_infty' A) }⟩
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem coe_add (f g : ModularForm Γ k) : ⇑(f + g) = f + g :=
  rfl


@[simp]
theorem add_apply (f g : ModularForm Γ k) (z : ℍ) : (f + g) z = f z + g z :=
  rfl


instance instZero : Zero (ModularForm Γ k) :=
  ⟨ { toSlashInvariantForm := 0
      holo' := fun _ => mdifferentiableAt_const
                                   /-
                                     Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
                                     k : Int
                                     A : Matrix.SpecialLinearGroup (Fin 2) Int
                                     ⊢ UpperHalfPlane.IsBoundedAtImInfty (SlashAction.map Complex k A ⇑0)
                                   -/
      bdd_at_infty' := fun A => by simpa using zero_form_isBoundedAtImInfty } ⟩
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem coe_zero : ⇑(0 : ModularForm Γ k) = (0 : ℍ → ℂ) :=
  rfl


@[simp]
theorem zero_apply (z : ℍ) : (0 : ModularForm Γ k) z = 0 :=
  rfl


instance instSMul : SMul α (ModularForm Γ k) :=
  ⟨fun c f =>
    { toSlashInvariantForm := c • f.1
                  /-
                    Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
                    k : Int
                    α : Type u_1
                    inst✝¹ : SMul α Complex
                    inst✝ : IsScalarTower α Complex Complex
                    c : α
                    f : ModularForm Γ k
                    ⊢ MDifferentiable (modelWithCornersSelf Complex Complex) (modelWithCornersSelf …
                  -/
      holo' := by simpa using f.holo'.const_smul (c • (1 : ℂ))
                  /-
                    🎉 no goals
                  -/
                                   /-
                                     Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
                                     k : Int
                                     α : Type u_1
                                     inst✝¹ : SMul α Complex
                                     inst✝ : IsScalarTower α Complex Complex
                                     c : α
                                     f : ModularForm Γ k
                                     A : Matrix.SpecialLinearGroup (Fin 2) Int
                                     ⊢ UpperHalfPlane.IsBoundedAtImInfty (SlashAction.map Complex k A ⇑(HSMul.hSMul …
                                   -/
      bdd_at_infty' := fun A => by simpa using (f.bdd_at_infty' A).const_smul_left (c • (1 : ℂ)) }⟩
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem coe_smul (f : ModularForm Γ k) (n : α) : ⇑(n • f) = n • ⇑f :=
  rfl


@[simp]
theorem smul_apply (f : ModularForm Γ k) (n : α) (z : ℍ) : (n • f) z = n • f z :=
  rfl


instance instNeg : Neg (ModularForm Γ k) :=
  ⟨fun f =>
    { toSlashInvariantForm := -f.1
      holo' := f.holo'.neg
                                   /-
                                     Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
                                     k : Int
                                     f : ModularForm Γ k
                                     A : Matrix.SpecialLinearGroup (Fin 2) Int
                                     ⊢ UpperHalfPlane.IsBoundedAtImInfty (SlashAction.map Complex k A ⇑(Neg.neg f.t …
                                   -/
      bdd_at_infty' := fun A => by simpa using (f.bdd_at_infty' A).neg }⟩
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem coe_neg (f : ModularForm Γ k) : ⇑(-f) = -f :=
  rfl


@[simp]
theorem neg_apply (f : ModularForm Γ k) (z : ℍ) : (-f) z = -f z :=
  rfl


instance instSub : Sub (ModularForm Γ k) :=
  ⟨fun f g => f + -g⟩


@[simp]
theorem coe_sub (f g : ModularForm Γ k) : ⇑(f - g) = f - g :=
  rfl


@[simp]
theorem sub_apply (f g : ModularForm Γ k) (z : ℍ) : (f - g) z = f z - g z :=
  rfl


instance : AddCommGroup (ModularForm Γ k) :=
  DFunLike.coe_injective.addCommGroup _ rfl coe_add coe_neg coe_sub coe_smul coe_smul


/-- Additive coercion from `ModularForm` to `ℍ → ℂ`. -/
@[simps]
def coeHom : ModularForm Γ k →+ ℍ → ℂ where
  toFun f := f
  map_zero' := coe_zero
  map_add' _ _ := rfl


instance : Module ℂ (ModularForm Γ k) :=
  Function.Injective.module ℂ coeHom DFunLike.coe_injective fun _ _ => rfl


instance : Inhabited (ModularForm Γ k) :=
  ⟨0⟩


/-- The modular form of weight `k_1 + k_2` given by the product of two modular forms of weights
`k_1` and `k_2`. -/
def mul {k_1 k_2 : ℤ} {Γ : Subgroup SL(2, ℤ)} (f : ModularForm Γ k_1) (g : ModularForm Γ k_2) :
    ModularForm Γ (k_1 + k_2) where
  toSlashInvariantForm := f.1.mul g.1
  holo' := f.holo'.mul g.holo'
  bdd_at_infty' A := by
    /-
      Γ✝ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
      k k_1 k_2 : Int
      Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
      f : ModularForm Γ k_1
      g : ModularForm Γ k_2
      A : Matrix.SpecialLinearGroup (Fin 2) Int
      ⊢ UpperHalfPlane.IsBoundedAtImInfty (SlashAction.map Complex (HAdd.hAdd k_1 k_ …
    -/
    simpa only [coe_mul, mul_slash_SL2] using (f.bdd_at_infty' A).mul (g.bdd_at_infty' A)
    /-
      🎉 no goals
    -/


@[simp]
theorem mul_coe {k_1 k_2 : ℤ} {Γ : Subgroup SL(2, ℤ)} (f : ModularForm Γ k_1)
    (g : ModularForm Γ k_2) : (f.mul g : ℍ → ℂ) = f * g :=
  rfl


/-- The constant function with value `x : ℂ` as a modular form of weight 0 and any level. -/
def const (x : ℂ) : ModularForm Γ 0 where
  toSlashInvariantForm := .const x
  holo' _ := mdifferentiableAt_const
  bdd_at_infty' A := by
    simpa only [SlashInvariantForm.const_toFun,
      ModularForm.is_invariant_const] using atImInfty.const_boundedAtFilter x


@[simp]
lemma const_apply (x : ℂ) (τ : ℍ) : (const x : ModularForm Γ 0) τ = x := rfl


instance : One (ModularForm Γ 0) where
  one := { const 1 with toSlashInvariantForm := 1 }


@[simp]
theorem one_coe_eq_one : ⇑(1 : ModularForm Γ 0) = 1 :=
  rfl


instance (Γ : Subgroup SL(2, ℤ)) : NatCast (ModularForm Γ 0) where
  natCast n := const n


@[simp, norm_cast]
lemma coe_natCast (Γ : Subgroup SL(2, ℤ)) (n : ℕ) :
    ⇑(n : ModularForm Γ 0) = n := rfl


lemma toSlashInvariantForm_natCast (Γ : Subgroup SL(2, ℤ)) (n : ℕ) :
    (n : ModularForm Γ 0).toSlashInvariantForm = n := rfl


instance (Γ : Subgroup SL(2, ℤ)) : IntCast (ModularForm Γ 0) where
  intCast z := const z


@[simp, norm_cast]
lemma coe_intCast (Γ : Subgroup SL(2, ℤ)) (z : ℤ) :
    ⇑(z : ModularForm Γ 0) = z := rfl


lemma toSlashInvariantForm_intCast (Γ : Subgroup SL(2, ℤ)) (z : ℤ) :
    (z : ModularForm Γ 0).toSlashInvariantForm = z := rfl


instance hasAdd : Add (CuspForm Γ k) :=
  ⟨fun f g =>
    { toSlashInvariantForm := f + g
      holo' := f.holo'.add g.holo'
                                    /-
                                      F : Type u_1
                                      Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
                                      k : Int
                                      f g : CuspForm Γ k
                                      A : Matrix.SpecialLinearGroup (Fin 2) Int
                                      ⊢ UpperHalfPlane.IsZeroAtImInfty (SlashAction.map Complex k A ⇑(HAdd.hAdd { to …
                                    -/
      zero_at_infty' := fun A => by simpa using (f.zero_at_infty' A).add (g.zero_at_infty' A) }⟩
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem coe_add (f g : CuspForm Γ k) : ⇑(f + g) = f + g :=
  rfl


@[simp]
theorem add_apply (f g : CuspForm Γ k) (z : ℍ) : (f + g) z = f z + g z :=
  rfl


instance instZero : Zero (CuspForm Γ k) :=
  ⟨ { toSlashInvariantForm := 0
      holo' := fun _ => mdifferentiableAt_const
                           /-
                             F : Type u_1
                             Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
                             k : Int
                             ⊢ ∀ (A : Matrix.SpecialLinearGroup (Fin 2) Int), UpperHalfPlane.IsZeroAtImInft …
                           -/
      zero_at_infty' := by simpa using Filter.zero_zeroAtFilter _ } ⟩
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem coe_zero : ⇑(0 : CuspForm Γ k) = (0 : ℍ → ℂ) :=
  rfl


@[simp]
theorem zero_apply (z : ℍ) : (0 : CuspForm Γ k) z = 0 :=
  rfl


instance instSMul : SMul α (CuspForm Γ k) :=
  ⟨fun c f =>
    { toSlashInvariantForm := c • f.1
                  /-
                    F : Type u_1
                    Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
                    k : Int
                    α : Type u_2
                    inst✝¹ : SMul α Complex
                    inst✝ : IsScalarTower α Complex Complex
                    c : α
                    f : CuspForm Γ k
                    ⊢ MDifferentiable (modelWithCornersSelf Complex Complex) (modelWithCornersSelf …
                  -/
      holo' := by simpa using f.holo'.const_smul (c • (1 : ℂ))
                  /-
                    🎉 no goals
                  -/
                                    /-
                                      F : Type u_1
                                      Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
                                      k : Int
                                      α : Type u_2
                                      inst✝¹ : SMul α Complex
                                      inst✝ : IsScalarTower α Complex Complex
                                      c : α
                                      f : CuspForm Γ k
                                      A : Matrix.SpecialLinearGroup (Fin 2) Int
                                      ⊢ UpperHalfPlane.IsZeroAtImInfty (SlashAction.map Complex k A ⇑(HSMul.hSMul c  …
                                    -/
      zero_at_infty' := fun A => by simpa using (f.zero_at_infty' A).smul (c • (1 : ℂ)) }⟩
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem coe_smul (f : CuspForm Γ k) (n : α) : ⇑(n • f) = n • ⇑f :=
  rfl


@[simp]
theorem smul_apply (f : CuspForm Γ k) (n : α) {z : ℍ} : (n • f) z = n • f z :=
  rfl


instance instNeg : Neg (CuspForm Γ k) :=
  ⟨fun f =>
    { toSlashInvariantForm := -f.1
      holo' := f.holo'.neg
                                    /-
                                      F : Type u_1
                                      Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
                                      k : Int
                                      f : CuspForm Γ k
                                      A : Matrix.SpecialLinearGroup (Fin 2) Int
                                      ⊢ UpperHalfPlane.IsZeroAtImInfty (SlashAction.map Complex k A ⇑(Neg.neg f.toSl …
                                    -/
      zero_at_infty' := fun A => by simpa using (f.zero_at_infty' A).neg }⟩
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem coe_neg (f : CuspForm Γ k) : ⇑(-f) = -f :=
  rfl


@[simp]
theorem neg_apply (f : CuspForm Γ k) (z : ℍ) : (-f) z = -f z :=
  rfl


instance instSub : Sub (CuspForm Γ k) :=
  ⟨fun f g => f + -g⟩


@[simp]
theorem coe_sub (f g : CuspForm Γ k) : ⇑(f - g) = f - g :=
  rfl


@[simp]
theorem sub_apply (f g : CuspForm Γ k) (z : ℍ) : (f - g) z = f z - g z :=
  rfl


instance : AddCommGroup (CuspForm Γ k) :=
  DFunLike.coe_injective.addCommGroup _ rfl coe_add coe_neg coe_sub coe_smul coe_smul


/-- Additive coercion from `CuspForm` to `ℍ → ℂ`. -/
@[simps]
def coeHom : CuspForm Γ k →+ ℍ → ℂ where
  toFun f := f
  map_zero' := CuspForm.coe_zero
  map_add' _ _ := rfl


instance : Module ℂ (CuspForm Γ k) :=
  Function.Injective.module ℂ coeHom DFunLike.coe_injective fun _ _ => rfl


instance : Inhabited (CuspForm Γ k) :=
  ⟨0⟩


instance (priority := 99) [FunLike F ℍ ℂ] [CuspFormClass F Γ k] : ModularFormClass F Γ k where
  slash_action_eq := SlashInvariantFormClass.slash_action_eq
  holo := CuspFormClass.holo
  bdd_at_infty _ _ := (CuspFormClass.zero_at_infty _ _).boundedAtFilter


/-- Cast for modular forms, which is useful for avoiding `Heq`s. -/
def mcast {a b : ℤ} {Γ : Subgroup SL(2, ℤ)} (h : a = b) (f : ModularForm Γ a) :
    ModularForm Γ b where
  toFun := (f : ℍ → ℂ)
  slash_action_eq' A := h ▸ f.slash_action_eq' A
  holo' := f.holo'
  bdd_at_infty' A := h ▸ f.bdd_at_infty' A


@[ext (iff := false)]
theorem gradedMonoid_eq_of_cast {Γ : Subgroup SL(2, ℤ)} {a b : GradedMonoid (ModularForm Γ)}
    (h : a.fst = b.fst) (h2 : mcast h a.snd = b.snd) : a = b := by
  /-
    Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
    a b : GradedMonoid (ModularForm Γ)
    h : Eq a.fst b.fst
    h2 : Eq (ModularForm.mcast h a.snd) b.snd
    ⊢ Eq a b
  -/
  obtain ⟨i, a⟩ := a
  /-
    case mk
    Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
    b : GradedMonoid (ModularForm Γ)
    i : Int
    a : ModularForm Γ i
    h : Eq ⟨i, a⟩.fst b.fst
    h2 : Eq (ModularForm.mcast h ⟨i, a⟩.snd) b.snd
    ⊢ Eq ⟨i, a⟩ b
  -/
  obtain ⟨j, b⟩ := b
  /-
    case mk.mk
    Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
    i : Int
    a : ModularForm Γ i
    j : Int
    b : ModularForm Γ j
    h : Eq ⟨i, a⟩.fst ⟨j, b⟩.fst
    h2 : Eq (ModularForm.mcast h ⟨i, a⟩.snd) ⟨j, b⟩.snd
    ⊢ Eq ⟨i, a⟩ ⟨j, b⟩
  -/
  cases h
  /-
    case mk.mk.refl
    Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
    i : Int
    a b : ModularForm Γ i
    h2 : Eq (ModularForm.mcast ⋯ ⟨i, a⟩.snd) ⟨i, b⟩.snd
    ⊢ Eq ⟨i, a⟩ ⟨i, b⟩
  -/
  exact congr_arg _ h2
  /-
    🎉 no goals
  -/


instance (Γ : Subgroup SL(2, ℤ)) : GradedMonoid.GOne (ModularForm Γ) where
  one := 1


instance (Γ : Subgroup SL(2, ℤ)) : GradedMonoid.GMul (ModularForm Γ) where
  mul f g := f.mul g


instance instGCommRing (Γ : Subgroup SL(2, ℤ)) : DirectSum.GCommRing (ModularForm Γ) where
  one_mul _ := gradedMonoid_eq_of_cast (zero_add _) (ext fun _ => one_mul _)
  mul_one _ := gradedMonoid_eq_of_cast (add_zero _) (ext fun _ => mul_one _)
  mul_assoc _ _ _ := gradedMonoid_eq_of_cast (add_assoc _ _ _) (ext fun _ => mul_assoc _ _ _)
  mul_zero {_ _} _ := ext fun _ => mul_zero _
  zero_mul {_ _} _ := ext fun _ => zero_mul _
  mul_add {_ _} _ _ _ := ext fun _ => mul_add _ _ _
  add_mul {_ _} _ _ _ := ext fun _ => add_mul _ _ _
  mul_comm _ _ := gradedMonoid_eq_of_cast (add_comm _ _) (ext fun _ => mul_comm _ _)
  natCast := Nat.cast
  natCast_zero := ext fun _ => Nat.cast_zero
  natCast_succ _ := ext fun _ => Nat.cast_succ _
  intCast := Int.cast
  intCast_ofNat _ := ext fun _ => AddGroupWithOne.intCast_ofNat _
  intCast_negSucc_ofNat _ := ext fun _ => AddGroupWithOne.intCast_negSucc _


instance instGAlgebra (Γ : Subgroup SL(2, ℤ)) : DirectSum.GAlgebra ℂ (ModularForm Γ) where
  toFun := { toFun := const, map_zero' := rfl, map_add' := fun _ _ => rfl }
  map_one := rfl
  map_mul _x _y := rfl
  commutes _c _x := gradedMonoid_eq_of_cast (add_comm _ _) (ext fun _ => mul_comm _ _)
  smul_def _x _x := gradedMonoid_eq_of_cast (zero_add _).symm (ext fun _ => rfl)


