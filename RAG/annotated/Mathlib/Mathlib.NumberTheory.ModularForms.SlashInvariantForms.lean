/-- Functions `ℍ → ℂ` that are invariant under the `SlashAction`. -/
structure SlashInvariantForm where
  toFun : ℍ → ℂ
  slash_action_eq' : ∀ γ ∈ Γ, toFun ∣[k] γ = toFun


/-- `SlashInvariantFormClass F Γ k` asserts `F` is a type of bundled functions that are invariant
under the `SlashAction`. -/
class SlashInvariantFormClass [FunLike F ℍ ℂ] : Prop where
  slash_action_eq : ∀ (f : F), ∀ γ ∈ Γ, (f : ℍ → ℂ) ∣[k] γ = f


instance (priority := 100) SlashInvariantForm.funLike :
    FunLike (SlashInvariantForm Γ k) ℍ ℂ where
  coe := SlashInvariantForm.toFun
                             /-
                               F : Type u_1
                               Γ : outParam (Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int))
                               k : outParam Int
                               f g : SlashInvariantForm Γ k
                               h : Eq f.toFun g.toFun
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; congr
                                               /-
                                                 🎉 no goals
                                               -/


instance (priority := 100) SlashInvariantFormClass.slashInvariantForm :
    SlashInvariantFormClass (SlashInvariantForm Γ k) Γ k where
  slash_action_eq := SlashInvariantForm.slash_action_eq'


@[simp]
theorem SlashInvariantForm.toFun_eq_coe {f : SlashInvariantForm Γ k} : f.toFun = (f : ℍ → ℂ) :=
  rfl


@[simp]
theorem SlashInvariantForm.coe_mk (f : ℍ → ℂ) (hf : ∀ γ ∈ Γ, f ∣[k] γ = f) : ⇑(mk f hf) = f := rfl


@[ext]
theorem SlashInvariantForm.ext {f g : SlashInvariantForm Γ k} (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext f g h


/-- Copy of a `SlashInvariantForm` with a new `toFun` equal to the old one.
Useful to fix definitional equalities. -/
protected def SlashInvariantForm.copy (f : SlashInvariantForm Γ k) (f' : ℍ → ℂ) (h : f' = ⇑f) :
    SlashInvariantForm Γ k where
  toFun := f'
  slash_action_eq' := h.symm ▸ f.slash_action_eq'


theorem slash_action_eqn [SlashInvariantFormClass F Γ k] (f : F) (γ) (hγ : γ ∈ Γ) :
    ↑f ∣[k] γ = ⇑f :=
  SlashInvariantFormClass.slash_action_eq f γ hγ


theorem slash_action_eqn' {k : ℤ} {Γ : Subgroup SL(2, ℤ)} [SlashInvariantFormClass F Γ k]
    (f : F) {γ} (hγ : γ ∈ Γ) (z : ℍ) :
    f (γ • z) = (γ 1 0 * z + γ 1 1) ^ k * f z := by
  /-
    F : Type u_1
    inst✝¹ : FunLike F UpperHalfPlane Complex
    k : Int
    Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
    inst✝ : SlashInvariantFormClass F Γ k
    f : F
    γ : Matrix.SpecialLinearGroup (Fin 2) Int
    hγ : Membership.mem Γ γ
    z : UpperHalfPlane
    ⊢ Eq (f (HSMul.hSMul γ z)) (HMul.hMul (HPow.hPow (HAdd.hAdd (HMul.hMul ↑(↑γ 1  …
  -/
  rw [← ModularForm.slash_action_eq'_iff, slash_action_eqn f γ hγ]
  /-
    🎉 no goals
  -/


/--Every `SlashInvariantForm` `f` satisfies ` f (γ • z) = (denom γ z) ^ k * f z`. -/
theorem slash_action_eqn'' {F : Type*} [FunLike F ℍ ℂ] {k : ℤ} {Γ : Subgroup SL(2, ℤ)}
    [SlashInvariantFormClass F Γ k] (f : F) {γ : SL(2, ℤ)} (hγ : γ ∈ Γ) (z : ℍ) :
    f (γ • z) = (denom γ z) ^ k * f z :=
  SlashInvariantForm.slash_action_eqn' f hγ z


instance [SlashInvariantFormClass F Γ k] : CoeTC F (SlashInvariantForm Γ k) :=
  ⟨fun f ↦ { slash_action_eq' := slash_action_eqn f }⟩


instance instAdd : Add (SlashInvariantForm Γ k) :=
  ⟨fun f g ↦
    { toFun := f + g
      slash_action_eq' := fun γ hγ ↦ by
        /-
          F : Type u_1
          Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
          k : Int
          inst✝ : FunLike F UpperHalfPlane Complex
          f g : SlashInvariantForm Γ k
          γ : Matrix.SpecialLinearGroup (Fin 2) Int
          hγ : Membership.mem Γ γ
          ⊢ Eq (SlashAction.map Complex k γ (HAdd.hAdd ⇑f ⇑g)) (HAdd.hAdd ⇑f ⇑g)
        -/
        rw [SlashAction.add_slash, slash_action_eqn f γ hγ, slash_action_eqn g γ hγ] }⟩
        /-
          🎉 no goals
        -/


@[simp]
theorem coe_add (f g : SlashInvariantForm Γ k) : ⇑(f + g) = f + g :=
  rfl


@[simp]
theorem add_apply (f g : SlashInvariantForm Γ k) (z : ℍ) : (f + g) z = f z + g z :=
  rfl


instance instZero : Zero (SlashInvariantForm Γ k) :=
  ⟨{toFun := 0
    slash_action_eq' := fun _ _ ↦ SlashAction.zero_slash _ _}⟩


@[simp]
theorem coe_zero : ⇑(0 : SlashInvariantForm Γ k) = (0 : ℍ → ℂ) :=
  rfl


instance instSMul : SMul α (SlashInvariantForm Γ k) :=
  ⟨fun c f =>
    { toFun := c • ↑f
      slash_action_eq' := fun γ hγ => by
        /-
          F : Type u_1
          Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
          k : Int
          inst✝² : FunLike F UpperHalfPlane Complex
          α : Type u_2
          inst✝¹ : SMul α Complex
          inst✝ : IsScalarTower α Complex Complex
          c : α
          f : SlashInvariantForm Γ k
          γ : Matrix.SpecialLinearGroup (Fin 2) Int
          hγ : Membership.mem Γ γ
          ⊢ Eq (SlashAction.map Complex k γ (HSMul.hSMul c ⇑f)) (HSMul.hSMul c ⇑f)
        -/
        rw [SlashAction.smul_slash_of_tower, slash_action_eqn f _ hγ]}⟩
        /-
          🎉 no goals
        -/


@[simp]
theorem coe_smul (f : SlashInvariantForm Γ k) (n : α) : ⇑(n • f) = n • ⇑f :=
  rfl


@[simp]
theorem smul_apply (f : SlashInvariantForm Γ k) (n : α) (z : ℍ) : (n • f) z = n • f z :=
  rfl


instance instNeg : Neg (SlashInvariantForm Γ k) :=
  ⟨fun f =>
    { toFun := -f
                                         /-
                                           F : Type u_1
                                           Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
                                           k : Int
                                           inst✝ : FunLike F UpperHalfPlane Complex
                                           f : SlashInvariantForm Γ k
                                           γ : Matrix.SpecialLinearGroup (Fin 2) Int
                                           hγ : Membership.mem Γ γ
                                           ⊢ Eq (SlashAction.map Complex k γ (Neg.neg ⇑f)) (Neg.neg ⇑f)
                                         -/
      slash_action_eq' := fun γ hγ => by rw [SlashAction.neg_slash, slash_action_eqn f γ hγ] }⟩
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
theorem coe_neg (f : SlashInvariantForm Γ k) : ⇑(-f) = -f :=
  rfl


@[simp]
theorem neg_apply (f : SlashInvariantForm Γ k) (z : ℍ) : (-f) z = -f z :=
  rfl


instance instSub : Sub (SlashInvariantForm Γ k) :=
  ⟨fun f g => f + -g⟩


@[simp]
theorem coe_sub (f g : SlashInvariantForm Γ k) : ⇑(f - g) = f - g :=
  rfl


@[simp]
theorem sub_apply (f g : SlashInvariantForm Γ k) (z : ℍ) : (f - g) z = f z - g z :=
  rfl


instance : AddCommGroup (SlashInvariantForm Γ k) :=
  DFunLike.coe_injective.addCommGroup _ rfl coe_add coe_neg coe_sub coe_smul coe_smul


/-- Additive coercion from `SlashInvariantForm` to `ℍ → ℂ`. -/
def coeHom : SlashInvariantForm Γ k →+ ℍ → ℂ where
  toFun f := f
  map_zero' := rfl
  map_add' _ _ := rfl


theorem coeHom_injective : Function.Injective (@coeHom Γ k) :=
  DFunLike.coe_injective


instance : Module ℂ (SlashInvariantForm Γ k) :=
  coeHom_injective.module ℂ coeHom fun _ _ => rfl


/-- The `SlashInvariantForm` corresponding to `Function.const _ x`. -/
@[simps (config := .asFn)]
def const (x : ℂ) : SlashInvariantForm Γ 0 where
  toFun := Function.const _ x
  slash_action_eq' A _ := ModularForm.is_invariant_const A x


instance : One (SlashInvariantForm Γ 0) where
  one := { const 1 with toFun := 1 }


@[simp]
theorem one_coe_eq_one : ((1 : SlashInvariantForm Γ 0) : ℍ → ℂ) = 1 :=
  rfl


instance : Inhabited (SlashInvariantForm Γ k) :=
  ⟨0⟩


/-- The slash invariant form of weight `k₁ + k₂` given by the product of two modular forms of
weights `k₁` and `k₂`. -/
def mul {k₁ k₂ : ℤ} {Γ : Subgroup SL(2, ℤ)} (f : SlashInvariantForm Γ k₁)
    (g : SlashInvariantForm Γ k₂) : SlashInvariantForm Γ (k₁ + k₂) where
  toFun := f * g
  slash_action_eq' A hA := by rw [ModularForm.mul_slash_SL2,
    SlashInvariantFormClass.slash_action_eq f A hA, SlashInvariantFormClass.slash_action_eq g A hA]


@[simp]
theorem coe_mul {k₁ k₂ : ℤ} {Γ : Subgroup SL(2, ℤ)} (f : SlashInvariantForm Γ k₁)
    (g : SlashInvariantForm Γ k₂) : ⇑(f.mul g) = ⇑f * ⇑g :=
  rfl


instance (Γ : Subgroup SL(2, ℤ)) : NatCast (SlashInvariantForm Γ 0) where
  natCast n := const n


@[simp, norm_cast]
theorem coe_natCast (n : ℕ) : ⇑(n : SlashInvariantForm Γ 0) = n := rfl


instance (Γ : Subgroup SL(2, ℤ)) : IntCast (SlashInvariantForm Γ 0) where
  intCast z := const z


@[simp, norm_cast]
theorem coe_intCast (z : ℤ) : ⇑(z : SlashInvariantForm Γ 0) = z := rfl


