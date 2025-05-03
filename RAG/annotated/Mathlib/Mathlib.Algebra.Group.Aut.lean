/-- The group of multiplicative automorphisms. -/
@[reducible, to_additive "The group of additive automorphisms."]
def MulAut (M : Type*) [Mul M] :=
  M ≃* M

-- Note that `(attr := reducible)` in `to_additive` currently doesn't work,
-- so we add the reducible attribute manually.

/-- The group operation on multiplicative automorphisms is defined by `g h => MulEquiv.trans h g`.
This means that multiplication agrees with composition, `(g*h)(x) = g (h x)`.
-/
instance : Group (MulAut M) where
  mul g h := MulEquiv.trans h g
  one := MulEquiv.refl _
  inv := MulEquiv.symm
  mul_assoc _ _ _ := rfl
  one_mul _ := rfl
  mul_one _ := rfl
  inv_mul_cancel := MulEquiv.self_trans_symm


instance : Inhabited (MulAut M) :=
  ⟨1⟩


@[simp]
theorem coe_mul (e₁ e₂ : MulAut M) : ⇑(e₁ * e₂) = e₁ ∘ e₂ :=
  rfl


@[simp]
theorem coe_one : ⇑(1 : MulAut M) = id :=
  rfl


theorem mul_def (e₁ e₂ : MulAut M) : e₁ * e₂ = e₂.trans e₁ :=
  rfl


theorem one_def : (1 : MulAut M) = MulEquiv.refl _ :=
  rfl


theorem inv_def (e₁ : MulAut M) : e₁⁻¹ = e₁.symm :=
  rfl


@[simp]
theorem mul_apply (e₁ e₂ : MulAut M) (m : M) : (e₁ * e₂) m = e₁ (e₂ m) :=
  rfl


@[simp]
theorem one_apply (m : M) : (1 : MulAut M) m = m :=
  rfl


@[simp]
theorem apply_inv_self (e : MulAut M) (m : M) : e (e⁻¹ m) = m :=
  MulEquiv.apply_symm_apply _ _


@[simp]
theorem inv_apply_self (e : MulAut M) (m : M) : e⁻¹ (e m) = m :=
  MulEquiv.apply_symm_apply _ _


/-- Monoid hom from the group of multiplicative automorphisms to the group of permutations. -/
def toPerm : MulAut M →* Equiv.Perm M where
  toFun := MulEquiv.toEquiv
  map_one' := rfl
  map_mul' _ _ := rfl


/-- The tautological action by `MulAut M` on `M`.

This generalizes `Function.End.applyMulAction`. -/
instance applyMulAction {M} [Monoid M] : MulAction (MulAut M) M where
  smul := (· <| ·)
  one_smul _ := rfl
  mul_smul _ _ _ := rfl


@[simp]
protected theorem smul_def {M} [Monoid M] (f : MulAut M) (a : M) : f • a = f a :=
  rfl


/-- `MulAut.applyDistribMulAction` is faithful. -/
instance apply_faithfulSMul {M} [Monoid M] : FaithfulSMul (MulAut M) M :=
  ⟨ fun h => MulEquiv.ext h ⟩


/-- Group conjugation, `MulAut.conj g h = g * h * g⁻¹`, as a monoid homomorphism
mapping multiplication in `G` into multiplication in the automorphism group `MulAut G`.
See also the type `ConjAct G` for any group `G`, which has a `MulAction (ConjAct G) G` instance
where `conj G` acts on `G` by conjugation. -/
def conj [Group G] : G →* MulAut G where
  toFun g :=
    { toFun := fun h => g * h * g⁻¹
      invFun := fun h => g⁻¹ * h * g
                              /-
                                A : Type u_1
                                M : Type u_2
                                G : Type u_3
                                inst✝¹ : Mul M
                                inst✝ : Group G
                                g x✝ : G
                                ⊢ Eq ((fun h => HMul.hMul (HMul.hMul (Inv.inv g) h) g) ((fun h => HMul.hMul (H …
                              -/
      left_inv := fun _ => by simp only [mul_assoc, inv_mul_cancel_left, inv_mul_cancel, mul_one]
                              /-
                                🎉 no goals
                              -/
                               /-
                                 A : Type u_1
                                 M : Type u_2
                                 G : Type u_3
                                 inst✝¹ : Mul M
                                 inst✝ : Group G
                                 g x✝ : G
                                 ⊢ Eq ((fun h => HMul.hMul (HMul.hMul g h) (Inv.inv g)) ((fun h => HMul.hMul (H …
                               -/
      right_inv := fun _ => by simp only [mul_assoc, mul_inv_cancel_left, mul_inv_cancel, mul_one]
                               /-
                                 🎉 no goals
                               -/
                     /-
                       A : Type u_1
                       M : Type u_2
                       G : Type u_3
                       inst✝¹ : Mul M
                       inst✝ : Group G
                       g : G
                       ⊢ ∀ (x y : G), Eq ({ toFun := fun h => HMul.hMul (HMul.hMul g h) (Inv.inv g),  …
                     -/
      map_mul' := by simp only [mul_assoc, inv_mul_cancel_left, forall_const] }
                     /-
                       🎉 no goals
                     -/
  map_mul' g₁ g₂ := by
    /-
      A : Type u_1
      M : Type u_2
      G : Type u_3
      inst✝¹ : Mul M
      inst✝ : Group G
      g₁ g₂ : G
      ⊢ Eq ({ toFun := fun g => { toFun := fun h => HMul.hMul (HMul.hMul g h) (Inv.i …
    -/
    ext h
    /-
      case h
      A : Type u_1
      M : Type u_2
      G : Type u_3
      inst✝¹ : Mul M
      inst✝ : Group G
      g₁ g₂ h : G
      ⊢ Eq (({ toFun := fun g => { toFun := fun h => HMul.hMul (HMul.hMul g h) (Inv. …
    -/
                 /-
                   A : Type u_1
                   M : Type u_2
                   G : Type u_3
                   inst✝¹ : Mul M
                   inst✝ : Group G
                   ⊢ Eq ((fun g => { toFun := fun h => HMul.hMul (HMul.hMul g h) (Inv.inv g), inv …
                 -/
    show g₁ * g₂ * h * (g₁ * g₂)⁻¹ = g₁ * (g₂ * h * g₂⁻¹) * g₁⁻¹
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
    /-
      case h
      A : Type u_1
      M : Type u_2
      G : Type u_3
      inst✝¹ : Mul M
      inst✝ : Group G
      g₁ g₂ h : G
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul g₁ g₂) h) (Inv.inv (HMul.hMul g₁ g₂))) ( …
    -/
    simp only [mul_assoc, mul_inv_rev]
    /-
      🎉 no goals
    -/
  map_one' := by ext; simp only [one_mul, inv_one, mul_one, one_apply]; rfl


@[simp]
theorem conj_apply [Group G] (g h : G) : conj g h = g * h * g⁻¹ :=
  rfl


@[simp]
theorem conj_symm_apply [Group G] (g h : G) : (conj g).symm h = g⁻¹ * h * g :=
  rfl


@[simp]
theorem conj_inv_apply [Group G] (g h : G) : (conj g)⁻¹ h = g⁻¹ * h * g :=
  rfl


/-- Isomorphic groups have isomorphic automorphism groups. -/
@[simps]
def congr [Group G] {H : Type*} [Group H] (ϕ : G ≃* H) :
    MulAut G ≃* MulAut H where
  toFun f := ϕ.symm.trans (f.trans ϕ)
  invFun f := ϕ.trans (f.trans ϕ.symm)
                   /-
                     A : Type u_1
                     M : Type u_2
                     G : Type u_3
                     inst✝² : Mul M
                     inst✝¹ : Group G
                     H : Type u_4
                     inst✝ : Group H
                     ϕ : MulEquiv G H
                     x✝ : MulAut G
                     ⊢ Eq ((fun f => ϕ.trans (MulEquiv.trans f ϕ.symm)) ((fun f => ϕ.symm.trans (Mu …
                   -/
  left_inv _ := by simp [DFunLike.ext_iff]
                   /-
                     🎉 no goals
                   -/
                    /-
                      A : Type u_1
                      M : Type u_2
                      G : Type u_3
                      inst✝² : Mul M
                      inst✝¹ : Group G
                      H : Type u_4
                      inst✝ : Group H
                      ϕ : MulEquiv G H
                      x✝ : MulAut H
                      ⊢ Eq ((fun f => ϕ.symm.trans (MulEquiv.trans f ϕ)) ((fun f => ϕ.trans (MulEqui …
                    -/
  right_inv _ := by simp [DFunLike.ext_iff]
                    /-
                      🎉 no goals
                    -/
                 /-
                   A : Type u_1
                   M : Type u_2
                   G : Type u_3
                   inst✝² : Mul M
                   inst✝¹ : Group G
                   H : Type u_4
                   inst✝ : Group H
                   ϕ : MulEquiv G H
                   ⊢ ∀ (x y : MulAut G), Eq ({ toFun := fun f => ϕ.symm.trans (MulEquiv.trans f ϕ …
                 -/
  map_mul' := by simp [DFunLike.ext_iff]
                 /-
                   🎉 no goals
                 -/


/-- The group operation on additive automorphisms is defined by `g h => AddEquiv.trans h g`.
This means that multiplication agrees with composition, `(g*h)(x) = g (h x)`.
-/
instance group : Group (AddAut A) where
  mul g h := AddEquiv.trans h g
  one := AddEquiv.refl _
  inv := AddEquiv.symm
  mul_assoc _ _ _ := rfl
  one_mul _ := rfl
  mul_one _ := rfl
  inv_mul_cancel := AddEquiv.self_trans_symm


instance : Inhabited (AddAut A) :=
  ⟨1⟩


@[simp]
theorem coe_mul (e₁ e₂ : AddAut A) : ⇑(e₁ * e₂) = e₁ ∘ e₂ :=
  rfl


@[simp]
theorem coe_one : ⇑(1 : AddAut A) = id :=
  rfl


theorem mul_def (e₁ e₂ : AddAut A) : e₁ * e₂ = e₂.trans e₁ :=
  rfl


theorem one_def : (1 : AddAut A) = AddEquiv.refl _ :=
  rfl


theorem inv_def (e₁ : AddAut A) : e₁⁻¹ = e₁.symm :=
  rfl


@[simp]
theorem mul_apply (e₁ e₂ : AddAut A) (a : A) : (e₁ * e₂) a = e₁ (e₂ a) :=
  rfl


@[simp]
theorem one_apply (a : A) : (1 : AddAut A) a = a :=
  rfl


@[simp]
theorem apply_inv_self (e : AddAut A) (a : A) : e⁻¹ (e a) = a :=
  AddEquiv.apply_symm_apply _ _


@[simp]
theorem inv_apply_self (e : AddAut A) (a : A) : e (e⁻¹ a) = a :=
  AddEquiv.apply_symm_apply _ _


/-- Monoid hom from the group of multiplicative automorphisms to the group of permutations. -/
def toPerm : AddAut A →* Equiv.Perm A where
  toFun := AddEquiv.toEquiv
  map_one' := rfl
  map_mul' _ _ := rfl


/-- The tautological action by `AddAut A` on `A`.

This generalizes `Function.End.applyMulAction`. -/
instance applyMulAction {A} [AddMonoid A] : MulAction (AddAut A) A where
  smul := (· <| ·)
  one_smul _ := rfl
  mul_smul _ _ _ := rfl


@[simp]
protected theorem smul_def {A} [AddMonoid A] (f : AddAut A) (a : A) : f • a = f a :=
  rfl


/-- `AddAut.applyDistribMulAction` is faithful. -/
instance apply_faithfulSMul {A} [AddMonoid A] : FaithfulSMul (AddAut A) A :=
  ⟨fun h => AddEquiv.ext h⟩


/-- Additive group conjugation, `AddAut.conj g h = g + h - g`, as an additive monoid
homomorphism mapping addition in `G` into multiplication in the automorphism group `AddAut G`
(written additively in order to define the map). -/
def conj [AddGroup G] : G →+ Additive (AddAut G) where
  toFun g :=
    @Additive.ofMul (AddAut G)
      { toFun := fun h => g + h + -g
        -- this definition is chosen to match `MulAut.conj`
        invFun := fun h => -g + h + g
        left_inv := fun _ => by
          /-
            A : Type u_1
            M : Type u_2
            G : Type u_3
            inst✝¹ : Add A
            inst✝ : AddGroup G
            g x✝ : G
            ⊢ Eq ((fun h => HAdd.hAdd (HAdd.hAdd (Neg.neg g) h) g) ((fun h => HAdd.hAdd (H …
          -/
          simp only [add_assoc, neg_add_cancel_left, neg_add_cancel, add_zero]
          /-
            🎉 no goals
          -/
        right_inv := fun _ => by
          /-
            A : Type u_1
            M : Type u_2
            G : Type u_3
            inst✝¹ : Add A
            inst✝ : AddGroup G
            g x✝ : G
            ⊢ Eq ((fun h => HAdd.hAdd (HAdd.hAdd g h) (Neg.neg g)) ((fun h => HAdd.hAdd (H …
          -/
          simp only [add_assoc, add_neg_cancel_left, add_neg_cancel, add_zero]
          /-
            🎉 no goals
          -/
                       /-
                         A : Type u_1
                         M : Type u_2
                         G : Type u_3
                         inst✝¹ : Add A
                         inst✝ : AddGroup G
                         g : G
                         ⊢ ∀ (x y : G), Eq ({ toFun := fun h => HAdd.hAdd (HAdd.hAdd g h) (Neg.neg g),  …
                       -/
        map_add' := by simp only [add_assoc, neg_add_cancel_left, forall_const] }
                       /-
                         🎉 no goals
                       -/
  map_add' g₁ g₂ := by
    /-
      A : Type u_1
      M : Type u_2
      G : Type u_3
      inst✝¹ : Add A
      inst✝ : AddGroup G
      g₁ g₂ : G
      ⊢ Eq ({ toFun := fun g => Additive.ofMul { toFun := fun h => HAdd.hAdd (HAdd.h …
    -/
    apply Additive.toMul.injective; ext h
    /-
      case a.h
      A : Type u_1
      M : Type u_2
      G : Type u_3
      inst✝¹ : Add A
      inst✝ : AddGroup G
      g₁ g₂ h : G
      ⊢ Eq ((Additive.toMul ({ toFun := fun g => Additive.ofMul { toFun := fun h =>  …
    -/
    show g₁ + g₂ + h + -(g₁ + g₂) = g₁ + (g₂ + h + -g₂) + -g₁
    /-
      A : Type u_1
      M : Type u_2
      G : Type u_3
      inst✝¹ : Add A
      inst✝ : AddGroup G
      ⊢ Eq ((fun g => Additive.ofMul { toFun := fun h => HAdd.hAdd (HAdd.hAdd g h) ( …
    -/
    /-
      case a.h
      A : Type u_1
      M : Type u_2
      G : Type u_3
      inst✝¹ : Add A
      inst✝ : AddGroup G
      g₁ g₂ h : G
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd g₁ g₂) h) (Neg.neg (HAdd.hAdd g₁ g₂))) ( …
    -/
    /-
      case a.h
      A : Type u_1
      M : Type u_2
      G : Type u_3
      inst✝¹ : Add A
      inst✝ : AddGroup G
      x✝ : G
      ⊢ Eq ((Additive.toMul ((fun g => Additive.ofMul { toFun := fun h => HAdd.hAdd  …
    -/
    simp only [add_assoc, neg_add_rev]
    /-
      case a.h
      A : Type u_1
      M : Type u_2
      G : Type u_3
      inst✝¹ : Add A
      inst✝ : AddGroup G
      x✝ : G
      ⊢ Eq ({ toFun := fun h => h, invFun := fun h => h, left_inv := ⋯, right_inv := …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  map_zero' := by
    apply Additive.toMul.injective; ext
    simp only [zero_add, neg_zero, add_zero, toMul_ofMul, toMul_zero, one_apply]
    rfl


@[simp]
theorem conj_apply [AddGroup G] (g h : G) : conj g h = g + h + -g :=
  rfl


@[simp]
theorem conj_symm_apply [AddGroup G] (g h : G) : (conj g).symm h = -g + h + g :=
  rfl

-- Porting note: the exact translation of this mathlib3 lemma would be`(-conj g) h = -g + h + g`,
-- but this no longer pass the simp_nf linter, as the LHS simplifies by `toMul_neg` to
-- `(conj g).toMul⁻¹`.

@[simp]
theorem conj_inv_apply [AddGroup G] (g h : G) : (conj g).toMul⁻¹ h = -g + h + g :=
  rfl


/-- Isomorphic additive groups have isomorphic automorphism groups. -/
@[simps]
def congr [AddGroup G] {H : Type*} [AddGroup H] (ϕ : G ≃+ H) :
    AddAut G ≃* AddAut H where
  toFun f := ϕ.symm.trans (f.trans ϕ)
  invFun f := ϕ.trans (f.trans ϕ.symm)
                   /-
                     A : Type u_1
                     M : Type u_2
                     G : Type u_3
                     inst✝² : Add A
                     inst✝¹ : AddGroup G
                     H : Type u_4
                     inst✝ : AddGroup H
                     ϕ : AddEquiv G H
                     x✝ : AddAut G
                     ⊢ Eq ((fun f => ϕ.trans (AddEquiv.trans f ϕ.symm)) ((fun f => ϕ.symm.trans (Ad …
                   -/
  left_inv _ := by simp [DFunLike.ext_iff]
                   /-
                     🎉 no goals
                   -/
                    /-
                      A : Type u_1
                      M : Type u_2
                      G : Type u_3
                      inst✝² : Add A
                      inst✝¹ : AddGroup G
                      H : Type u_4
                      inst✝ : AddGroup H
                      ϕ : AddEquiv G H
                      x✝ : AddAut H
                      ⊢ Eq ((fun f => ϕ.symm.trans (AddEquiv.trans f ϕ)) ((fun f => ϕ.trans (AddEqui …
                    -/
  right_inv _ := by simp [DFunLike.ext_iff]
                    /-
                      🎉 no goals
                    -/
                 /-
                   A : Type u_1
                   M : Type u_2
                   G : Type u_3
                   inst✝² : Add A
                   inst✝¹ : AddGroup G
                   H : Type u_4
                   inst✝ : AddGroup H
                   ϕ : AddEquiv G H
                   ⊢ ∀ (x y : AddAut G), Eq ({ toFun := fun f => ϕ.symm.trans (AddEquiv.trans f ϕ …
                 -/
  map_mul' := by simp [DFunLike.ext_iff]
                 /-
                   🎉 no goals
                 -/


/-- `Multiplicative G` and `G` have isomorphic automorphism groups. -/
@[simps!]
def MulAutMultiplicative [AddGroup G] : MulAut (Multiplicative G) ≃* AddAut G :=
  { AddEquiv.toMultiplicative.symm with map_mul' := fun _ _ ↦ rfl }


/-- `Additive G` and `G` have isomorphic automorphism groups. -/
@[simps!]
def AddAutAdditive [Group G] : AddAut (Additive G) ≃* MulAut G :=
  { MulEquiv.toAdditive.symm with map_mul' := fun _ _ ↦ rfl }

