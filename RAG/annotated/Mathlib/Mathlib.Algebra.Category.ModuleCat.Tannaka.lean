/-- An ingredient of Tannaka duality for rings:
A ring `R` is equivalent to
the endomorphisms of the additive forgetful functor `Module R ⥤ AddCommGroup`.
-/
def ringEquivEndForget₂ (R : Type u) [Ring R] :
    R ≃+* End (AdditiveFunctor.of (forget₂ (ModuleCat.{u} R) AddCommGrp.{u})) where
  toFun r :=
    { app := fun M =>
        @AddCommGrp.ofHom M.carrier M.carrier _ _ (DistribMulAction.toAddMonoidHom M r) }
  invFun φ := φ.app (ModuleCat.of R R) (1 : R)
  left_inv := by
    /-
      R : Type u
      inst✝ : Ring R
      ⊢ Function.LeftInverse (fun φ => (φ.app (ModuleCat.of R R)) 1) fun r => { app  …
    -/
    intro r
    /-
      R : Type u
      inst✝ : Ring R
      r : R
      ⊢ Eq ((fun φ => (φ.app (ModuleCat.of R R)) 1) ((fun r => { app := fun M => Add …
    -/
    simp
    /-
      🎉 no goals
    -/
  right_inv := by
    /-
      R : Type u
      inst✝ : Ring R
      ⊢ Function.RightInverse (fun φ => (φ.app (ModuleCat.of R R)) 1) fun r => { app …
    -/
    intro φ
    /-
      R : Type u
      inst✝ : Ring R
      φ : CategoryTheory.End (CategoryTheory.AdditiveFunctor.of (CategoryTheory.forg …
      ⊢ Eq ((fun r => { app := fun M => AddCommGrp.ofHom (DistribMulAction.toAddMono …
    -/
    apply NatTrans.ext
    /-
      case app
      R : Type u
      inst✝ : Ring R
      φ : CategoryTheory.End (CategoryTheory.AdditiveFunctor.of (CategoryTheory.forg …
      ⊢ Eq ((fun r => { app := fun M => AddCommGrp.ofHom (DistribMulAction.toAddMono …
    -/
    ext M (x : M)
    have w := congr_fun ((forget _).congr_map
      (φ.naturality (ModuleCat.ofHom (LinearMap.toSpanSingleton R M x)))) (1 : R)
    /-
      case app.h.w
      R : Type u
      inst✝ : Ring R
      φ : CategoryTheory.End (CategoryTheory.AdditiveFunctor.of (CategoryTheory.forg …
      M : ModuleCat R
      x : ↑M
      w : Eq ((CategoryTheory.forget AddCommGrp).map (CategoryTheory.CategoryStruct. …
      ⊢ Eq ((((fun r => { app := fun M => AddCommGrp.ofHom (DistribMulAction.toAddMo …
    -/
    exact w.symm.trans (congr_arg (φ.app M) (one_smul R x))
    /-
      🎉 no goals
    -/
  map_add' := by
    /-
      R : Type u
      inst✝ : Ring R
      ⊢ ∀ (x y : R), Eq ({ toFun := fun r => { app := fun M => AddCommGrp.ofHom (Dis …
    -/
    intros
    /-
      R : Type u
      inst✝ : Ring R
      x✝ y✝ : R
      ⊢ Eq ({ toFun := fun r => { app := fun M => AddCommGrp.ofHom (DistribMulAction …
    -/
    apply NatTrans.ext
    /-
      case app
      R : Type u
      inst✝ : Ring R
      x✝ y✝ : R
      ⊢ Eq ({ toFun := fun r => { app := fun M => AddCommGrp.ofHom (DistribMulAction …
    -/
    ext
    simp only [AdditiveFunctor.of_fst, ModuleCat.forget₂_obj, AddCommGrp.coe_of,
    /-
      R : Type u
      inst✝ : Ring R
      ⊢ ∀ (x y : R), Eq ({ toFun := fun r => { app := fun M => AddCommGrp.ofHom (Dis …
    -/
      AddCommGrp.ofHom_apply, DistribMulAction.toAddMonoidHom_apply, add_smul]
    /-
      R : Type u
      inst✝ : Ring R
      x✝ y✝ : R
      ⊢ Eq ({ toFun := fun r => { app := fun M => AddCommGrp.ofHom (DistribMulAction …
    -/
    /-
      case app.h.w
      R : Type u
      inst✝ : Ring R
      x✝² y✝ : R
      x✝¹ : ModuleCat R
      x✝ : ↑((CategoryTheory.AdditiveFunctor.of (CategoryTheory.forget₂ (ModuleCat R …
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul x✝² x✝) (HSMul.hSMul y✝ x✝)) (((HAdd.hAdd { app : …
    -/
    /-
      case app
      R : Type u
      inst✝ : Ring R
      x✝ y✝ : R
      ⊢ Eq ({ toFun := fun r => { app := fun M => AddCommGrp.ofHom (DistribMulAction …
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_mul' := by
    /-
      case app.h.w
      R : Type u
      inst✝ : Ring R
      x✝² y✝ : R
      x✝¹ : ModuleCat R
      x✝ : ↑((CategoryTheory.AdditiveFunctor.of (CategoryTheory.forget₂ (ModuleCat R …
      ⊢ Eq (HSMul.hSMul x✝² (HSMul.hSMul y✝ x✝)) (((HMul.hMul { app := fun M => AddC …
    -/
    intros
    /-
      🎉 no goals
    -/
    apply NatTrans.ext
    ext
    simp only [AdditiveFunctor.of_fst, ModuleCat.forget₂_obj, AddCommGrp.coe_of,
      AddCommGrp.ofHom_apply, DistribMulAction.toAddMonoidHom_apply, mul_smul]
    rfl

