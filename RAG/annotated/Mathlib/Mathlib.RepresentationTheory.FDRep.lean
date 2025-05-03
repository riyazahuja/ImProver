/-- The category of finite dimensional `k`-linear representations of a monoid `G`. -/
abbrev FDRep (k G : Type u) [Field k] [Monoid G] :=
  Action (FGModuleCat.{u} k) (MonCat.of G)


@[deprecated (since := "2024-07-05")]
alias FdRep := FDRep


instance : LargeCategory (FDRep k G) := inferInstance

instance : ConcreteCategory (FDRep k G) := inferInstance

instance : Preadditive (FDRep k G) := inferInstance

instance : HasFiniteLimits (FDRep k G) := inferInstance


                                      /-
                                        k G : Type u
                                        inst✝¹ : Field k
                                        inst✝ : Monoid G
                                        ⊢ CategoryTheory.Linear k (FDRep k G)
                                      -/
instance : Linear k (FDRep k G) := by infer_instance
                                      /-
                                        🎉 no goals
                                      -/


instance : CoeSort (FDRep k G) (Type u) :=
  ConcreteCategory.hasCoeToSort _


instance (V : FDRep k G) : AddCommGroup V := by
  /-
    k G : Type u
    inst✝¹ : Field k
    inst✝ : Monoid G
    V : FDRep k G
    ⊢ AddCommGroup (CoeSort.coe V)
  -/
  change AddCommGroup ((forget₂ (FDRep k G) (FGModuleCat k)).obj V).obj; infer_instance
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


instance (V : FDRep k G) : Module k V := by
  /-
    k G : Type u
    inst✝¹ : Field k
    inst✝ : Monoid G
    V : FDRep k G
    ⊢ Module k (CoeSort.coe V)
  -/
  change Module k ((forget₂ (FDRep k G) (FGModuleCat k)).obj V).obj; infer_instance
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


instance (V : FDRep k G) : FiniteDimensional k V := by
  /-
    k G : Type u
    inst✝¹ : Field k
    inst✝ : Monoid G
    V : FDRep k G
    ⊢ FiniteDimensional k (CoeSort.coe V)
  -/
  change FiniteDimensional k ((forget₂ (FDRep k G) (FGModuleCat k)).obj V); infer_instance
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


/-- All hom spaces are finite dimensional. -/
instance (V W : FDRep k G) : FiniteDimensional k (V ⟶ W) :=
  FiniteDimensional.of_injective ((forget₂ (FDRep k G) (FGModuleCat k)).mapLinearMap k)
    (Functor.map_injective (forget₂ (FDRep k G) (FGModuleCat k)))


/-- The monoid homomorphism corresponding to the action of `G` onto `V : FDRep k G`. -/
def ρ (V : FDRep k G) : G →* V →ₗ[k] V :=
  (ModuleCat.endMulEquiv _).toMonoidHom.comp (Action.ρ V)


@[simp]
lemma endMulEquiv_symm_comp_ρ (V : FDRep k G) :
    (MonoidHomClass.toMonoidHom (ModuleCat.endMulEquiv V.V.obj).symm).comp (ρ V) = Action.ρ V := rfl


@[simp]
lemma endMulEquiv_comp_ρ (V : FDRep k G) :
    (MonoidHomClass.toMonoidHom (ModuleCat.endMulEquiv V.V.obj)).comp (Action.ρ V) = ρ V := rfl


@[simp]
lemma hom_action_ρ (V : FDRep k G) (g : G) : (Action.ρ V g).hom = ρ V g := rfl


/-- The underlying `LinearEquiv` of an isomorphism of representations. -/
def isoToLinearEquiv {V W : FDRep k G} (i : V ≅ W) : V ≃ₗ[k] W :=
  FGModuleCat.isoToLinearEquiv ((Action.forget (FGModuleCat k) (MonCat.of G)).mapIso i)


theorem Iso.conj_ρ {V W : FDRep k G} (i : V ≅ W) (g : G) :
    W.ρ g = (FDRep.isoToLinearEquiv i).conj (V.ρ g) := by
  -- Porting note: Changed `rw` to `erw`
  /-
    k G : Type u
    inst✝¹ : Field k
    inst✝ : Monoid G
    V W : FDRep k G
    i : CategoryTheory.Iso V W
    g : G
    ⊢ Eq (W.ρ g) ((FDRep.isoToLinearEquiv i).conj (V.ρ g))
  -/
  erw [FDRep.isoToLinearEquiv, ← hom_action_ρ V, ← FGModuleCat.Iso.conj_hom_eq_conj, Iso.conj_apply]
  rw [← ModuleCat.hom_ofHom (W.ρ g), ← ModuleCat.hom_ext_iff,
      Iso.eq_inv_comp ((Action.forget (FGModuleCat k) (MonCat.of G)).mapIso i)]
  /-
    k G : Type u
    inst✝¹ : Field k
    inst✝ : Monoid G
    V W : FDRep k G
    i : CategoryTheory.Iso V W
    g : G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Action.forget (FGModuleCat k) (MonC …
  -/
  exact (i.hom.comm g).symm
  /-
    🎉 no goals
  -/


/-- Lift an unbundled representation to `FDRep`. -/
@[simps ρ]
def of {V : Type u} [AddCommGroup V] [Module k V] [FiniteDimensional k V]
    (ρ : Representation k G V) : FDRep k G :=
  ⟨FGModuleCat.of k V, ρ ≫ MonCat.ofHom (ModuleCat.endMulEquiv _).symm.toMonoidHom⟩


instance : HasForget₂ (FDRep k G) (Rep k G) where
  forget₂ := (forget₂ (FGModuleCat k) (ModuleCat k)).mapAction (MonCat.of G)


theorem forget₂_ρ (V : FDRep k G) : ((forget₂ (FDRep k G) (Rep k G)).obj V).ρ = V.ρ := by
  /-
    k G : Type u
    inst✝¹ : Field k
    inst✝ : Monoid G
    V : FDRep k G
    ⊢ Eq ((CategoryTheory.forget₂ (FDRep k G) (Rep k G)).obj V).ρ V.ρ
  -/
  ext g v; rfl
           /-
             🎉 no goals
           -/

-- Verify that the monoidal structure is available.

                                        /-
                                          k G : Type u
                                          inst✝¹ : Field k
                                          inst✝ : Monoid G
                                          ⊢ CategoryTheory.Limits.HasKernels (FDRep k G)
                                        -/
instance : HasKernels (FDRep k G) := by infer_instance
                                        /-
                                          🎉 no goals
                                        -/


/-- Schur's Lemma: the dimension of the `Hom`-space between two irreducible representation is `0` if
they are not isomorphic, and `1` if they are. -/
theorem finrank_hom_simple_simple [IsAlgClosed k] (V W : FDRep k G) [Simple V] [Simple W] :
    finrank k (V ⟶ W) = if Nonempty (V ≅ W) then 1 else 0 :=
  CategoryTheory.finrank_hom_simple_simple k V W


/-- The forgetful functor to `Rep k G` preserves hom-sets and their vector space structure. -/
def forget₂HomLinearEquiv (X Y : FDRep k G) :
    ((forget₂ (FDRep k G) (Rep k G)).obj X ⟶
      (forget₂ (FDRep k G) (Rep k G)).obj Y) ≃ₗ[k] X ⟶ Y where
  toFun f := ⟨f.hom, f.comm⟩
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  invFun f := ⟨(forget₂ (FGModuleCat k) (ModuleCat k)).map f.hom, f.comm⟩
                   /-
                     k G : Type u
                     inst✝¹ : Field k
                     inst✝ : Monoid G
                     X Y : FDRep k G
                     x✝ : Quiver.Hom ((CategoryTheory.forget₂ (FDRep k G) (Rep k G)).obj X) ((Categ …
                     ⊢ Eq ((fun f => { hom := (CategoryTheory.forget₂ (FGModuleCat k) (ModuleCat k) …
                   -/
  left_inv _ := by ext; rfl
                        /-
                          🎉 no goals
                        -/
                    /-
                      k G : Type u
                      inst✝¹ : Field k
                      inst✝ : Monoid G
                      X Y : FDRep k G
                      x✝ : Quiver.Hom X Y
                      ⊢ Eq ({ toFun := fun f => { hom := f.hom, comm := ⋯ }, map_add' := ⋯, map_smul …
                    -/
  right_inv _ := by ext; rfl
                         /-
                           🎉 no goals
                         -/


noncomputable instance : RightRigidCategory (FDRep k G) := by
  /-
    k G : Type u
    inst✝¹ : Field k
    inst✝ : Group G
    ⊢ CategoryTheory.RightRigidCategory (FDRep k G)
  -/
  change RightRigidCategory (Action (FGModuleCat k) (Grp.of G)); infer_instance
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- Auxiliary definition for `FDRep.dualTensorIsoLinHom`. -/
noncomputable def dualTensorIsoLinHomAux :
    (FDRep.of ρV.dual ⊗ W).V ≅ (FDRep.of (linHom ρV W.ρ)).V :=
  -- Porting note: had to make all types explicit arguments
  @LinearEquiv.toFGModuleCatIso k _ (FDRep.of ρV.dual ⊗ W).V (V →ₗ[k] W)
    _ _ _ _ _ _ (dualTensorHomEquiv k V W)


/-- When `V` and `W` are finite dimensional representations of a group `G`, the isomorphism
`dualTensorHomEquiv k V W` of vector spaces induces an isomorphism of representations. -/
noncomputable def dualTensorIsoLinHom : FDRep.of ρV.dual ⊗ W ≅ FDRep.of (linHom ρV W.ρ) := by
  /-
    k G V : Type u
    inst✝⁴ : Field k
    inst✝³ : Group G
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : FiniteDimensional k V
    ρV : Representation k G V
    W : FDRep k G
    ⊢ CategoryTheory.Iso (CategoryTheory.MonoidalCategoryStruct.tensorObj (FDRep.o …
  -/
  refine Action.mkIso (dualTensorIsoLinHomAux ρV W) (fun g => ?_)
  /-
    k G V : Type u
    inst✝⁴ : Field k
    inst✝³ : Group G
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : FiniteDimensional k V
    ρV : Representation k G V
    W : FDRep k G
    g : ↑(MonCat.of G)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategoryStru …
  -/
  ext : 1
  /-
    case h
    k G V : Type u
    inst✝⁴ : Field k
    inst✝³ : Group G
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : FiniteDimensional k V
    ρV : Representation k G V
    W : FDRep k G
    g : ↑(MonCat.of G)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategoryStru …
  -/
  exact dualTensorHom_comm ρV W.ρ g
  /-
    🎉 no goals
  -/


@[simp]
theorem dualTensorIsoLinHom_hom_hom :
    (dualTensorIsoLinHom ρV W).hom.hom = ModuleCat.ofHom (dualTensorHom k V W) :=
  rfl


