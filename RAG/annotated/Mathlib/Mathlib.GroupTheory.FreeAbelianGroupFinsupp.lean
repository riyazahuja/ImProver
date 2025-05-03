/-- The group homomorphism `FreeAbelianGroup X →+ (X →₀ ℤ)`. -/
def FreeAbelianGroup.toFinsupp : FreeAbelianGroup X →+ X →₀ ℤ :=
  FreeAbelianGroup.lift fun x => Finsupp.single x (1 : ℤ)


/-- The group homomorphism `(X →₀ ℤ) →+ FreeAbelianGroup X`. -/
def Finsupp.toFreeAbelianGroup : (X →₀ ℤ) →+ FreeAbelianGroup X :=
  Finsupp.liftAddHom fun x => (smulAddHom ℤ (FreeAbelianGroup X)).flip (FreeAbelianGroup.of x)


@[simp]
theorem Finsupp.toFreeAbelianGroup_comp_singleAddHom (x : X) :
    Finsupp.toFreeAbelianGroup.comp (Finsupp.singleAddHom x) =
      (smulAddHom ℤ (FreeAbelianGroup X)).flip (of x) := by
  /-
    X : Type u_1
    x : X
    ⊢ Eq (Finsupp.toFreeAbelianGroup.comp (Finsupp.singleAddHom x)) ((smulAddHom I …
  -/
  ext
  simp only [AddMonoidHom.coe_comp, Finsupp.singleAddHom_apply, Function.comp_apply, one_smul,
    toFreeAbelianGroup, Finsupp.liftAddHom_apply_single]


@[simp]
theorem FreeAbelianGroup.toFinsupp_comp_toFreeAbelianGroup :
    toFinsupp.comp toFreeAbelianGroup = AddMonoidHom.id (X →₀ ℤ) := by
  /-
    X : Type u_1
    ⊢ Eq (FreeAbelianGroup.toFinsupp.comp Finsupp.toFreeAbelianGroup) (AddMonoidHo …
  -/
  ext x y; simp only [AddMonoidHom.id_comp]
  /-
    case H.h1.h
    X : Type u_1
    x y : X
    ⊢ Eq ((((FreeAbelianGroup.toFinsupp.comp Finsupp.toFreeAbelianGroup).comp (Fin …
  -/
  rw [AddMonoidHom.comp_assoc, Finsupp.toFreeAbelianGroup_comp_singleAddHom]
  simp only [toFinsupp, AddMonoidHom.coe_comp, Finsupp.singleAddHom_apply, Function.comp_apply,
    one_smul, lift.of, AddMonoidHom.flip_apply, smulAddHom_apply, AddMonoidHom.id_apply]


@[simp]
theorem Finsupp.toFreeAbelianGroup_comp_toFinsupp :
    toFreeAbelianGroup.comp toFinsupp = AddMonoidHom.id (FreeAbelianGroup X) := by
  /-
    X : Type u_1
    ⊢ Eq (Finsupp.toFreeAbelianGroup.comp FreeAbelianGroup.toFinsupp) (AddMonoidHo …
  -/
  ext
  rw [toFreeAbelianGroup, toFinsupp, AddMonoidHom.comp_apply, lift.of,
    liftAddHom_apply_single, AddMonoidHom.flip_apply, smulAddHom_apply, one_smul,
    AddMonoidHom.id_apply]


@[simp]
theorem Finsupp.toFreeAbelianGroup_toFinsupp {X} (x : FreeAbelianGroup X) :
    Finsupp.toFreeAbelianGroup (FreeAbelianGroup.toFinsupp x) = x := by
  /-
    X : Type u_2
    x : FreeAbelianGroup X
    ⊢ Eq (Finsupp.toFreeAbelianGroup (FreeAbelianGroup.toFinsupp x)) x
  -/
  rw [← AddMonoidHom.comp_apply, Finsupp.toFreeAbelianGroup_comp_toFinsupp, AddMonoidHom.id_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinsupp_of (x : X) : toFinsupp (of x) = Finsupp.single x 1 := by
  /-
    X : Type u_1
    x : X
    ⊢ Eq (FreeAbelianGroup.toFinsupp (FreeAbelianGroup.of x)) (Finsupp.single x 1)
  -/
  simp only [toFinsupp, lift.of]
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinsupp_toFreeAbelianGroup (f : X →₀ ℤ) :
    FreeAbelianGroup.toFinsupp (Finsupp.toFreeAbelianGroup f) = f := by
  /-
    X : Type u_1
    f : Finsupp X Int
    ⊢ Eq (FreeAbelianGroup.toFinsupp (Finsupp.toFreeAbelianGroup f)) f
  -/
  rw [← AddMonoidHom.comp_apply, toFinsupp_comp_toFreeAbelianGroup, AddMonoidHom.id_apply]
  /-
    🎉 no goals
  -/


/-- The additive equivalence between `FreeAbelianGroup X` and `(X →₀ ℤ)`. -/
@[simps!]
def equivFinsupp : FreeAbelianGroup X ≃+ (X →₀ ℤ) where
  toFun := toFinsupp
  invFun := toFreeAbelianGroup
  left_inv := toFreeAbelianGroup_toFinsupp
  right_inv := toFinsupp_toFreeAbelianGroup
  map_add' := toFinsupp.map_add


/-- `A` is a basis of the ℤ-module `FreeAbelianGroup A`. -/
noncomputable def basis (α : Type*) : Basis α ℤ (FreeAbelianGroup α) :=
  ⟨(FreeAbelianGroup.equivFinsupp α).toIntLinearEquiv⟩


/-- Isomorphic free abelian groups (as modules) have equivalent bases. -/
def Equiv.ofFreeAbelianGroupLinearEquiv {α β : Type*}
    (e : FreeAbelianGroup α ≃ₗ[ℤ] FreeAbelianGroup β) : α ≃ β :=
  let t : Basis α ℤ (FreeAbelianGroup β) := (FreeAbelianGroup.basis α).map e
  t.indexEquiv <| FreeAbelianGroup.basis _


/-- Isomorphic free abelian groups (as additive groups) have equivalent bases. -/
def Equiv.ofFreeAbelianGroupEquiv {α β : Type*} (e : FreeAbelianGroup α ≃+ FreeAbelianGroup β) :
    α ≃ β :=
  Equiv.ofFreeAbelianGroupLinearEquiv e.toIntLinearEquiv


/-- Isomorphic free groups have equivalent bases. -/
def Equiv.ofFreeGroupEquiv {α β : Type*} (e : FreeGroup α ≃* FreeGroup β) : α ≃ β :=
  Equiv.ofFreeAbelianGroupEquiv (MulEquiv.toAdditive e.abelianizationCongr)


/-- Isomorphic free groups have equivalent bases (`IsFreeGroup` variant). -/
def Equiv.ofIsFreeGroupEquiv {G H : Type*} [Group G] [Group H] [IsFreeGroup G] [IsFreeGroup H]
    (e : G ≃* H) : Generators G ≃ Generators H :=
  Equiv.ofFreeGroupEquiv <| MulEquiv.trans (toFreeGroup G).symm <| MulEquiv.trans e <| toFreeGroup H


/-- `coeff x` is the additive group homomorphism `FreeAbelianGroup X →+ ℤ`
that sends `a` to the multiplicity of `x : X` in `a`. -/
def coeff (x : X) : FreeAbelianGroup X →+ ℤ :=
  (Finsupp.applyAddHom x).comp toFinsupp


/-- `support a` for `a : FreeAbelianGroup X` is the finite set of `x : X`
that occur in the formal sum `a`. -/
def support (a : FreeAbelianGroup X) : Finset X :=
  a.toFinsupp.support


theorem mem_support_iff (x : X) (a : FreeAbelianGroup X) : x ∈ a.support ↔ coeff x a ≠ 0 := by
  /-
    X : Type u_1
    x : X
    a : FreeAbelianGroup X
    ⊢ Iff (Membership.mem a.support x) (Ne ((FreeAbelianGroup.coeff x) a) 0)
  -/
  rw [support, Finsupp.mem_support_iff]
  /-
    X : Type u_1
    x : X
    a : FreeAbelianGroup X
    ⊢ Iff (Ne ((FreeAbelianGroup.toFinsupp a) x) 0) (Ne ((FreeAbelianGroup.coeff x …
  -/
  exact Iff.rfl
  /-
    🎉 no goals
  -/


theorem not_mem_support_iff (x : X) (a : FreeAbelianGroup X) : x ∉ a.support ↔ coeff x a = 0 := by
  /-
    X : Type u_1
    x : X
    a : FreeAbelianGroup X
    ⊢ Iff (Not (Membership.mem a.support x)) (Eq ((FreeAbelianGroup.coeff x) a) 0)
  -/
  rw [support, Finsupp.not_mem_support_iff]
  /-
    X : Type u_1
    x : X
    a : FreeAbelianGroup X
    ⊢ Iff (Eq ((FreeAbelianGroup.toFinsupp a) x) 0) (Eq ((FreeAbelianGroup.coeff x …
  -/
  exact Iff.rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem support_zero : support (0 : FreeAbelianGroup X) = ∅ := by
  /-
    X : Type u_1
    ⊢ Eq (FreeAbelianGroup.support 0) EmptyCollection.emptyCollection
  -/
  simp only [support, Finsupp.support_zero, AddMonoidHom.map_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem support_of (x : X) : support (of x) = {x} := by
  /-
    X : Type u_1
    x : X
    ⊢ Eq (FreeAbelianGroup.of x).support (Singleton.singleton x)
  -/
  rw [support, toFinsupp_of, Finsupp.support_single_ne_zero _ one_ne_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem support_neg (a : FreeAbelianGroup X) : support (-a) = support a := by
  /-
    X : Type u_1
    a : FreeAbelianGroup X
    ⊢ Eq (Neg.neg a).support a.support
  -/
  simp only [support, AddMonoidHom.map_neg, Finsupp.support_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem support_zsmul (k : ℤ) (h : k ≠ 0) (a : FreeAbelianGroup X) :
    support (k • a) = support a := by
  /-
    X : Type u_1
    k : Int
    h : Ne k 0
    a : FreeAbelianGroup X
    ⊢ Eq (HSMul.hSMul k a).support a.support
  -/
  ext x
  /-
    case h
    X : Type u_1
    k : Int
    h : Ne k 0
    a : FreeAbelianGroup X
    x : X
    ⊢ Iff (Membership.mem (HSMul.hSMul k a).support x) (Membership.mem a.support x)
  -/
  simp only [mem_support_iff, AddMonoidHom.map_zsmul]
  /-
    case h
    X : Type u_1
    k : Int
    h : Ne k 0
    a : FreeAbelianGroup X
    x : X
    ⊢ Iff (Ne (HSMul.hSMul k ((FreeAbelianGroup.coeff x) a)) 0) (Ne ((FreeAbelianG …
  -/
  simp only [h, zsmul_int_int, false_or, Ne, mul_eq_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem support_nsmul (k : ℕ) (h : k ≠ 0) (a : FreeAbelianGroup X) :
    support (k • a) = support a := by
  /-
    X : Type u_1
    k : Nat
    h : Ne k 0
    a : FreeAbelianGroup X
    ⊢ Eq (HSMul.hSMul k a).support a.support
  -/
  apply support_zsmul k _ a
  /-
    X : Type u_1
    k : Nat
    h : Ne k 0
    a : FreeAbelianGroup X
    ⊢ Ne (↑k) 0
  -/
  exact mod_cast h
  /-
    🎉 no goals
  -/


theorem support_add (a b : FreeAbelianGroup X) : support (a + b) ⊆ a.support ∪ b.support := by
  /-
    X : Type u_1
    a b : FreeAbelianGroup X
    ⊢ HasSubset.Subset (HAdd.hAdd a b).support (Union.union a.support b.support)
  -/
  simp only [support, AddMonoidHom.map_add]
  /-
    X : Type u_1
    a b : FreeAbelianGroup X
    ⊢ HasSubset.Subset (HAdd.hAdd (FreeAbelianGroup.toFinsupp a) (FreeAbelianGroup …
  -/
  apply Finsupp.support_add
  /-
    🎉 no goals
  -/


