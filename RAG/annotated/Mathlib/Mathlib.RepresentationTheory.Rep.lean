/-- The category of `k`-linear representations of a monoid `G`. -/
abbrev Rep (k G : Type u) [Ring k] [Monoid G] :=
  Action (ModuleCat.{u} k) (MonCat.of G)


                                                                           /-
                                                                             k G : Type u
                                                                             inst✝¹ : CommRing k
                                                                             inst✝ : Monoid G
                                                                             ⊢ CategoryTheory.Linear k (Rep k G)
                                                                           -/
instance (k G : Type u) [CommRing k] [Monoid G] : Linear k (Rep k G) := by infer_instance
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


instance : CoeSort (Rep k G) (Type u) :=
  ConcreteCategory.hasCoeToSort _


instance (V : Rep k G) : AddCommGroup V := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    V : Rep k G
    ⊢ AddCommGroup (CoeSort.coe V)
  -/
  change AddCommGroup ((forget₂ (Rep k G) (ModuleCat k)).obj V); infer_instance
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


instance (V : Rep k G) : Module k V := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    V : Rep k G
    ⊢ Module k (CoeSort.coe V)
  -/
  change Module k ((forget₂ (Rep k G) (ModuleCat k)).obj V)
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    V : Rep k G
    ⊢ Module k ↑((CategoryTheory.forget₂ (Rep k G) (ModuleCat k)).obj V)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Specialize the existing `Action.ρ`, changing the type to `Representation k G V`.
-/
def ρ (V : Rep k G) : Representation k G V :=
-- Porting note: was `V.ρ`
  (ModuleCat.endMulEquiv V.V).toMonoidHom.comp (Action.ρ V)


/-- Lift an unbundled representation to `Rep`. -/
def of {V : Type u} [AddCommGroup V] [Module k V] (ρ : G →* V →ₗ[k] V) : Rep k G :=
  ⟨ModuleCat.of k V, MonCat.ofHom ((ModuleCat.endMulEquiv _).symm.toMonoidHom.comp ρ) ⟩


@[simp]
theorem coe_of {V : Type u} [AddCommGroup V] [Module k V] (ρ : G →* V →ₗ[k] V) :
    (of ρ : Type u) = V :=
  rfl


@[simp]
theorem of_ρ {V : Type u} [AddCommGroup V] [Module k V] (ρ : G →* V →ₗ[k] V) : (of ρ).ρ = ρ :=
  rfl


theorem Action_ρ_eq_ρ {A : Rep k G} :
    Action.ρ A = (ModuleCat.endMulEquiv _).symm.toMonoidHom.comp A.ρ :=
  rfl


@[simp]
lemma ρ_hom {X : Rep k G} (g : G) : (Action.ρ X g).hom = X.ρ g := rfl


@[simp]
lemma ofHom_ρ {X : Rep k G} (g : G) : ModuleCat.ofHom (X.ρ g) = Action.ρ X g := rfl


/-- Allows us to apply lemmas about the underlying `ρ`, which would take an element `g : G` rather
than `g : MonCat.of G` as an argument. -/
theorem of_ρ_apply {V : Type u} [AddCommGroup V] [Module k V] (ρ : Representation k G V)
    (g : MonCat.of G) : (Rep.of ρ).ρ g = ρ (g : G) :=
  rfl


@[simp]
theorem ρ_inv_self_apply {G : Type u} [Group G] (A : Rep k G) (g : G) (x : A) :
    A.ρ g⁻¹ (A.ρ g x) = x :=
                                  /-
                                    k : Type u
                                    inst✝¹ : CommRing k
                                    G : Type u
                                    inst✝ : Group G
                                    A : Rep k G
                                    g : G
                                    x : CoeSort.coe A
                                    ⊢ Eq ((HMul.hMul (A.ρ (Inv.inv g)) (A.ρ g)) x) x
                                  -/
  show (A.ρ g⁻¹ * A.ρ g) x = x by rw [← map_mul, inv_mul_cancel, map_one, LinearMap.one_apply]
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem ρ_self_inv_apply {G : Type u} [Group G] {A : Rep k G} (g : G) (x : A) :
    A.ρ g (A.ρ g⁻¹ x) = x :=
                                  /-
                                    k : Type u
                                    inst✝¹ : CommRing k
                                    G : Type u
                                    inst✝ : Group G
                                    A : Rep k G
                                    g : G
                                    x : CoeSort.coe A
                                    ⊢ Eq ((HMul.hMul (A.ρ g) (A.ρ (Inv.inv g))) x) x
                                  -/
  show (A.ρ g * A.ρ g⁻¹) x = x by rw [← map_mul, mul_inv_cancel, map_one, LinearMap.one_apply]
                                  /-
                                    🎉 no goals
                                  -/


theorem hom_comm_apply {A B : Rep k G} (f : A ⟶ B) (g : G) (x : A) :
    f.hom (A.ρ g x) = B.ρ g (f.hom x) :=
  LinearMap.ext_iff.1 (ModuleCat.hom_ext_iff.mp (f.comm g)) x


/-- The trivial `k`-linear `G`-representation on a `k`-module `V.` -/
def trivial (V : Type u) [AddCommGroup V] [Module k V] : Rep k G :=
  Rep.of (@Representation.trivial k G V _ _ _ _)


theorem trivial_def {V : Type u} [AddCommGroup V] [Module k V] (g : G) (v : V) :
    (trivial k G V).ρ g v = v :=
  rfl


/-- A predicate for representations that fix every element. -/
abbrev IsTrivial (A : Rep k G) := A.ρ.IsTrivial


instance {V : Type u} [AddCommGroup V] [Module k V] :
    IsTrivial (Rep.trivial k G V) where


instance {V : Type u} [AddCommGroup V] [Module k V] (ρ : Representation k G V) [ρ.IsTrivial] :
    IsTrivial (Rep.of ρ) where

-- Porting note: the two following instances were found automatically in mathlib3

noncomputable instance : PreservesLimits (forget₂ (Rep k G) (ModuleCat.{u} k)) :=
  Action.preservesLimits_forget.{u} _ _


noncomputable instance : PreservesColimits (forget₂ (Rep k G) (ModuleCat.{u} k)) :=
  Action.preservesColimits_forget.{u} _ _

/- Porting note: linter complains `simp` unfolds some types in the LHS, so
have removed `@[simp]`. -/

theorem MonoidalCategory.braiding_hom_apply {A B : Rep k G} (x : A) (y : B) :
    Action.Hom.hom (β_ A B).hom (TensorProduct.tmul k x y) = TensorProduct.tmul k y x :=
  rfl

/- Porting note: linter complains `simp` unfolds some types in the LHS, so
have removed `@[simp]`. -/

theorem MonoidalCategory.braiding_inv_apply {A B : Rep k G} (x : A) (y : B) :
    Action.Hom.hom (β_ A B).inv (TensorProduct.tmul k y x) = TensorProduct.tmul k x y :=
  rfl


/-- The monoidal functor sending a type `H` with a `G`-action to the induced `k`-linear
`G`-representation on `k[H].` -/
noncomputable def linearization : (Action (Type u) (MonCat.of G)) ⥤ (Rep k G) :=
  (ModuleCat.free k).mapAction (MonCat.of G)


instance : (linearization k G).Monoidal := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    ⊢ (Rep.linearization k G).Monoidal
  -/
  dsimp only [linearization]
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    ⊢ ((ModuleCat.free k).mapAction (MonCat.of G)).Monoidal
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[simp]
theorem linearization_obj_ρ (X : Action (Type u) (MonCat.of G)) (g : G) (x : X.V →₀ k) :
    ((linearization k G).obj X).ρ g x = Finsupp.lmapDomain k k (X.ρ g) x :=
  rfl


theorem linearization_of (X : Action (Type u) (MonCat.of G)) (g : G) (x : X.V) :
    ((linearization k G).obj X).ρ g (Finsupp.single x (1 : k))
      = Finsupp.single (X.ρ g x) (1 : k) := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    X : Action (Type u) (MonCat.of G)
    g : G
    x : X.V
    ⊢ Eq ((((Rep.linearization k G).obj X).ρ g) (Finsupp.single x 1)) (Finsupp.sin …
  -/
  rw [linearization_obj_ρ, Finsupp.lmapDomain_apply, Finsupp.mapDomain_single]
  /-
    🎉 no goals
  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): helps fixing `linearizationTrivialIso` since change in behaviour of `ext`.

theorem linearization_single (X : Action (Type u) (MonCat.of G)) (g : G) (x : X.V) (r : k) :
    ((linearization k G).obj X).ρ g (Finsupp.single x r) = Finsupp.single (X.ρ g x) r := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    X : Action (Type u) (MonCat.of G)
    g : G
    x : X.V
    r : k
    ⊢ Eq ((((Rep.linearization k G).obj X).ρ g) (Finsupp.single x r)) (Finsupp.sin …
  -/
  rw [linearization_obj_ρ, Finsupp.lmapDomain_apply, Finsupp.mapDomain_single]
  /-
    🎉 no goals
  -/


@[simp]
theorem linearization_map_hom : ((linearization k G).map f).hom =
    ModuleCat.ofHom (Finsupp.lmapDomain k k f.hom) :=
  rfl


theorem linearization_map_hom_single (x : X.V) (r : k) :
    ((linearization k G).map f).hom (Finsupp.single x r) = Finsupp.single (f.hom x) r :=
  Finsupp.mapDomain_single


@[simp]
theorem linearization_μ_hom (X Y : Action (Type u) (MonCat.of G)) :
    (μ (linearization k G) X Y).hom =
      ModuleCat.ofHom (finsuppTensorFinsupp' k X.V Y.V).toLinearMap :=
  rfl


@[simp]
theorem linearization_δ_hom (X Y : Action (Type u) (MonCat.of G)) :
    (δ (linearization k G) X Y).hom =
      ModuleCat.ofHom (finsuppTensorFinsupp' k X.V Y.V).symm.toLinearMap :=
  rfl


@[simp]
theorem linearization_ε_hom : (ε (linearization k G)).hom =
    ModuleCat.ofHom (Finsupp.lsingle PUnit.unit) :=
  rfl


theorem linearization_η_hom_apply (r : k) :
    (η (linearization k G)).hom (Finsupp.single PUnit.unit r) = r :=
  (εIso (linearization k G)).hom_inv_id_apply r


/-- The linearization of a type `X` on which `G` acts trivially is the trivial `G`-representation
on `k[X]`. -/
@[simps!]
noncomputable def linearizationTrivialIso (X : Type u) :
    (linearization k G).obj (Action.mk X 1) ≅ trivial k G (X →₀ k) :=
  Action.mkIso (Iso.refl _) fun _ => ModuleCat.hom_ext <| Finsupp.lhom_ext' fun _ => LinearMap.ext
    fun _ => linearization_single ..


/-- Given a `G`-action on `H`, this is `k[H]` bundled with the natural representation
`G →* End(k[H])` as a term of type `Rep k G`. -/
noncomputable abbrev ofMulAction (H : Type u) [MulAction G H] : Rep k G :=
  of <| Representation.ofMulAction k G H


/-- The `k`-linear `G`-representation on `k[G]`, induced by left multiplication. -/
noncomputable def leftRegular : Rep k G :=
  ofMulAction k G G


/-- The `k`-linear `G`-representation on `k[Gⁿ]`, induced by left multiplication. -/
noncomputable def diagonal (n : ℕ) : Rep k G :=
  ofMulAction k G (Fin n → G)


/-- The linearization of a type `H` with a `G`-action is definitionally isomorphic to the
`k`-linear `G`-representation on `k[H]` induced by the `G`-action on `H`. -/
noncomputable def linearizationOfMulActionIso (H : Type u) [MulAction G H] :
    (linearization k G).obj (Action.ofMulAction G H) ≅ ofMulAction k G H :=
  Iso.refl _


/-- Turns a `k`-module `A` with a compatible `DistribMulAction` of a monoid `G` into a
`k`-linear `G`-representation on `A`. -/
def ofDistribMulAction : Rep k G := Rep.of (Representation.ofDistribMulAction k G A)


@[simp] theorem ofDistribMulAction_ρ_apply_apply (g : G) (a : A) :
    (ofDistribMulAction k G A).ρ g a = g • a := rfl


/-- Given an `R`-algebra `S`, the `ℤ`-linear representation associated to the natural action of
`S ≃ₐ[R] S` on `S`. -/
@[simp] def ofAlgebraAut (R S : Type) [CommRing R] [CommRing S] [Algebra R S] :
    Rep ℤ (S ≃ₐ[R] S) := ofDistribMulAction ℤ (S ≃ₐ[R] S) S


/-- Turns a `CommGroup` `G` with a `MulDistribMulAction` of a monoid `M` into a
`ℤ`-linear `M`-representation on `Additive G`. -/
def ofMulDistribMulAction : Rep ℤ M := Rep.of (Representation.ofMulDistribMulAction M G)


@[simp] theorem ofMulDistribMulAction_ρ_apply_apply (g : M) (a : Additive G) :
    (ofMulDistribMulAction M G).ρ g a = Additive.ofMul (g • a.toMul) := rfl


/-- Given an `R`-algebra `S`, the `ℤ`-linear representation associated to the natural action of
`S ≃ₐ[R] S` on `Sˣ`. -/
@[simp] def ofAlgebraAutOnUnits (R S : Type) [CommRing R] [CommRing S] [Algebra R S] :
    Rep ℤ (S ≃ₐ[R] S) := Rep.ofMulDistribMulAction (S ≃ₐ[R] S) Sˣ


/-- Given an element `x : A`, there is a natural morphism of representations `k[G] ⟶ A` sending
`g ↦ A.ρ(g)(x).` -/
@[simps]
noncomputable def leftRegularHom (A : Rep k G) (x : A) : Rep.ofMulAction k G G ⟶ A where
  hom := ModuleCat.ofHom <| Finsupp.lift _ _ _ fun g => A.ρ g x
  comm g := by
    /-
      k G : Type u
      inst✝¹ : CommRing k
      inst✝ : Monoid G
      X Y : Action (Type u) (MonCat.of G)
      f : Quiver.Hom X Y
      A : Rep k G
      x : CoeSort.coe A
      g : ↑(MonCat.of G)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Rep.ofMulAction k G G).ρ g) (Module …
    -/
    ext : 1
    /-
      case hf
      k G : Type u
      inst✝¹ : CommRing k
      inst✝ : Monoid G
      X Y : Action (Type u) (MonCat.of G)
      f : Quiver.Hom X Y
      A : Rep k G
      x : CoeSort.coe A
      g : ↑(MonCat.of G)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Rep.ofMulAction k G G).ρ g) (Module …
    -/
    refine Finsupp.lhom_ext' fun y => LinearMap.ext_ring ?_
/- Porting note: rest of broken proof was
    simpa only [LinearMap.comp_apply, ModuleCat.comp_def, Finsupp.lsingle_apply, Finsupp.lift_apply,
      Action_ρ_eq_ρ, of_ρ_apply, Representation.ofMulAction_single, Finsupp.sum_single_index,
      zero_smul, one_smul, smul_eq_mul, A.ρ.map_mul] -/
    /-
      case hf
      k G : Type u
      inst✝¹ : CommRing k
      inst✝ : Monoid G
      X Y : Action (Type u) (MonCat.of G)
      f : Quiver.Hom X Y
      A : Rep k G
      x : CoeSort.coe A
      g : ↑(MonCat.of G)
      y : G
      ⊢ Eq (((CategoryTheory.CategoryStruct.comp ((Rep.ofMulAction k G G).ρ g) (Modu …
    -/
    simp only [LinearMap.comp_apply, ModuleCat.hom_comp, Finsupp.lsingle_apply]
    /-
      case hf
      k G : Type u
      inst✝¹ : CommRing k
      inst✝ : Monoid G
      X Y : Action (Type u) (MonCat.of G)
      f : Quiver.Hom X Y
      A : Rep k G
      x : CoeSort.coe A
      g : ↑(MonCat.of G)
      y : G
      ⊢ Eq (((Finsupp.lift (↑A.1) k G) fun g => (A.ρ g) x) (((Rep.ofMulAction k G G) …
    -/
    erw [Finsupp.lift_apply, Finsupp.lift_apply, Representation.ofMulAction_single (G := G)]
    /-
      case hf
      k G : Type u
      inst✝¹ : CommRing k
      inst✝ : Monoid G
      X Y : Action (Type u) (MonCat.of G)
      f : Quiver.Hom X Y
      A : Rep k G
      x : CoeSort.coe A
      g : ↑(MonCat.of G)
      y : G
      ⊢ Eq ((Finsupp.single (HSMul.hSMul g y) 1).sum fun x_1 r => HSMul.hSMul r ((A. …
    -/
    simp only [Finsupp.sum_single_index, zero_smul, one_smul, smul_eq_mul, A.ρ.map_mul, of_ρ]
    /-
      case hf
      k G : Type u
      inst✝¹ : CommRing k
      inst✝ : Monoid G
      X Y : Action (Type u) (MonCat.of G)
      f : Quiver.Hom X Y
      A : Rep k G
      x : CoeSort.coe A
      g : ↑(MonCat.of G)
      y : G
      ⊢ Eq ((HMul.hMul (A.ρ g) (A.ρ y)) x) ((A.ρ g).hom ((A.ρ y) x))
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem leftRegularHom_apply {A : Rep k G} (x : A) :
    (leftRegularHom A x).hom (Finsupp.single 1 1) = x := by
  -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
  erw [leftRegularHom_hom_hom, Finsupp.lift_apply, Finsupp.sum_single_index, one_smul,
    A.ρ.map_one, LinearMap.one_apply]
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    A : Rep k G
    x : CoeSort.coe A
    ⊢ Eq (HSMul.hSMul 0 ((A.ρ 1) x)) 0
  -/
  rw [zero_smul]
  /-
    🎉 no goals
  -/


/-- Given a `k`-linear `G`-representation `A`, there is a `k`-linear isomorphism between
representation morphisms `Hom(k[G], A)` and `A`. -/
@[simps]
noncomputable def leftRegularHomEquiv (A : Rep k G) : (Rep.ofMulAction k G G ⟶ A) ≃ₗ[k] A where
  toFun f := f.hom (Finsupp.single 1 1)
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  invFun x := leftRegularHom A x
  left_inv f := by
    /-
      k G : Type u
      inst✝¹ : CommRing k
      inst✝ : Monoid G
      X Y : Action (Type u) (MonCat.of G)
      f✝ : Quiver.Hom X Y
      A : Rep k G
      f : Quiver.Hom (Rep.ofMulAction k G G) A
      ⊢ Eq ((fun x => A.leftRegularHom x) ({ toFun := fun f => f.hom.hom (Finsupp.si …
    -/
    refine Action.Hom.ext (ModuleCat.hom_ext (Finsupp.lhom_ext' fun x : G => LinearMap.ext_ring ?_))
    have :
      f.hom ((ofMulAction k G G).ρ x (Finsupp.single (1 : G) (1 : k))) =
        A.ρ x (f.hom (Finsupp.single (1 : G) (1 : k))) :=
      LinearMap.ext_iff.1 (ModuleCat.hom_ext_iff.mp (f.comm x)) (Finsupp.single 1 1)
    simp only [leftRegularHom_hom_hom, LinearMap.comp_apply, Finsupp.lsingle_apply,
      Finsupp.lift_apply, ← this, coe_of, of_ρ, Representation.ofMulAction_single x (1 : G) (1 : k),
      smul_eq_mul, mul_one, zero_smul, Finsupp.sum_single_index, one_smul]
    -- Mismatched `Zero k` instances
    /-
      k G : Type u
      inst✝¹ : CommRing k
      inst✝ : Monoid G
      X Y : Action (Type u) (MonCat.of G)
      f✝ : Quiver.Hom X Y
      A : Rep k G
      f : Quiver.Hom (Rep.ofMulAction k G G) A
      x : G
      this : Eq (f.hom.hom (((Rep.ofMulAction k G G).ρ x) (Finsupp.single 1 1))) ((A …
      ⊢ Eq (f.hom.hom (Finsupp.single x 1)) (f.hom.hom (Finsupp.single x 1))
    -/
    rfl
    /-
      🎉 no goals
    -/
  right_inv x := leftRegularHom_apply x


theorem leftRegularHomEquiv_symm_single {A : Rep k G} (x : A) (g : G) :
    ((leftRegularHomEquiv A).symm x).hom (Finsupp.single g 1) = A.ρ g x := by
  -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
  erw [leftRegularHomEquiv_symm_apply, leftRegularHom_hom_hom, Finsupp.lift_apply,
    Finsupp.sum_single_index, one_smul]
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    A : Rep k G
    x : CoeSort.coe A
    g : G
    ⊢ Eq (HSMul.hSMul 0 ((A.ρ g) x)) 0
  -/
  rw [zero_smul]
  /-
    🎉 no goals
  -/


/-- Given a `k`-linear `G`-representation `(A, ρ₁)`, this is the 'internal Hom' functor sending
`(B, ρ₂)` to the representation `Homₖ(A, B)` that maps `g : G` and `f : A →ₗ[k] B` to
`(ρ₂ g) ∘ₗ f ∘ₗ (ρ₁ g⁻¹)`. -/
@[simps]
protected def ihom (A : Rep k G) : Rep k G ⥤ Rep k G where
  obj B := Rep.of (Representation.linHom A.ρ B.ρ)
  map := fun {X} {Y} f =>
    { hom := ModuleCat.ofHom (LinearMap.llcomp k _ _ _ f.hom.hom)
      comm := fun g => ModuleCat.hom_ext <| LinearMap.ext fun x => LinearMap.ext fun y => by
        /-
          k G : Type u
          inst✝¹ : CommRing k
          inst✝ : Group G
          A✝ B C A X Y : Rep k G
          f : Quiver.Hom X Y
          g : ↑(MonCat.of G)
          x : ↑((fun B => Rep.of (A.ρ.linHom B.ρ)) X).V
          y : CoeSort.coe A
          ⊢ Eq (((CategoryTheory.CategoryStruct.comp (((fun B => Rep.of (A.ρ.linHom B.ρ) …
        -/
        show f.hom (X.ρ g _) = _
        /-
          k G : Type u
          inst✝¹ : CommRing k
          inst✝ : Group G
          A✝ B C A X Y : Rep k G
          f : Quiver.Hom X Y
          g : ↑(MonCat.of G)
          x : ↑((fun B => Rep.of (A.ρ.linHom B.ρ)) X).V
          y : CoeSort.coe A
          ⊢ Eq (f.hom.hom ((X.ρ g) ((LinearMap.comp x (A.ρ (Inv.inv g))) y))) (((Categor …
        -/
        simp only [hom_comm_apply]; rfl }
                                    /-
                                      🎉 no goals
                                    -/
                        /-
                          k G : Type u
                          inst✝¹ : CommRing k
                          inst✝ : Group G
                          A✝ B C A x✝ : Rep k G
                          ⊢ Eq ({ obj := fun B => Rep.of (A.ρ.linHom B.ρ), map := fun {X Y} f => { hom : …
                        -/
  map_id := fun _ => by ext; rfl
                             /-
                               🎉 no goals
                             -/
                            /-
                              k G : Type u
                              inst✝¹ : CommRing k
                              inst✝ : Group G
                              A✝ B C A X✝ Y✝ Z✝ : Rep k G
                              x✝¹ : Quiver.Hom X✝ Y✝
                              x✝ : Quiver.Hom Y✝ Z✝
                              ⊢ Eq ({ obj := fun B => Rep.of (A.ρ.linHom B.ρ), map := fun {X Y} f => { hom : …
                            -/
  map_comp := fun _ _ => by ext; rfl
                                 /-
                                   🎉 no goals
                                 -/


@[simp] theorem ihom_obj_ρ_apply {A B : Rep k G} (g : G) (x : A →ₗ[k] B) :
    ((Rep.ihom A).obj B).ρ g x = B.ρ g ∘ₗ x ∘ₗ A.ρ g⁻¹ :=
  rfl


/-- Given a `k`-linear `G`-representation `A`, this is the Hom-set bijection in the adjunction
`A ⊗ - ⊣ ihom(A, -)`. It sends `f : A ⊗ B ⟶ C` to a `Rep k G` morphism defined by currying the
`k`-linear map underlying `f`, giving a map `A →ₗ[k] B →ₗ[k] C`, then flipping the arguments. -/
def homEquiv (A B C : Rep k G) : (A ⊗ B ⟶ C) ≃ (B ⟶ (Rep.ihom A).obj C) where
  toFun f :=
    { hom := ModuleCat.ofHom <| (TensorProduct.curry f.hom.hom).flip
      comm := fun g => by
        /-
          k G : Type u
          inst✝¹ : CommRing k
          inst✝ : Group G
          A✝ B✝ C✝ A B C : Rep k G
          f : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj A B) C
          g : ↑(MonCat.of G)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (B.ρ g) (ModuleCat.ofHom (TensorProdu …
        -/
        ext x : 2
        /-
          case hf.h
          k G : Type u
          inst✝¹ : CommRing k
          inst✝ : Group G
          A✝ B✝ C✝ A B C : Rep k G
          f : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj A B) C
          g : ↑(MonCat.of G)
          x : ↑B.V
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (B.ρ g) (ModuleCat.ofHom (TensorProd …
        -/
        refine LinearMap.ext fun y => ?_
        /-
          case hf.h
          k G : Type u
          inst✝¹ : CommRing k
          inst✝ : Group G
          A✝ B✝ C✝ A B C : Rep k G
          f : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj A B) C
          g : ↑(MonCat.of G)
          x : ↑B.V
          y : CoeSort.coe A
          ⊢ Eq (((CategoryTheory.CategoryStruct.comp (B.ρ g) (ModuleCat.ofHom (TensorPro …
        -/
        change f.hom (_ ⊗ₜ[k] _) = C.ρ g (f.hom (_ ⊗ₜ[k] _))
        /-
          case hf.h
          k G : Type u
          inst✝¹ : CommRing k
          inst✝ : Group G
          A✝ B✝ C✝ A B C : Rep k G
          f : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj A B) C
          g : ↑(MonCat.of G)
          x : ↑B.V
          y : CoeSort.coe A
          ⊢ Eq (f.hom.hom (TensorProduct.tmul k y ((B.ρ g).hom x))) ((C.ρ g) (f.hom.hom  …
        -/
        rw [← hom_comm_apply]
        /-
          case hf.h
          k G : Type u
          inst✝¹ : CommRing k
          inst✝ : Group G
          A✝ B✝ C✝ A B C : Rep k G
          f : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj A B) C
          g : ↑(MonCat.of G)
          x : ↑B.V
          y : CoeSort.coe A
          ⊢ Eq (f.hom.hom (TensorProduct.tmul k y ((B.ρ g).hom x))) (f.hom.hom (((Catego …
        -/
        change _ = f.hom ((A.ρ g * A.ρ g⁻¹) y ⊗ₜ[k] _)
        /-
          case hf.h
          k G : Type u
          inst✝¹ : CommRing k
          inst✝ : Group G
          A✝ B✝ C✝ A B C : Rep k G
          f : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj A B) C
          g : ↑(MonCat.of G)
          x : ↑B.V
          y : CoeSort.coe A
          ⊢ Eq (f.hom.hom (TensorProduct.tmul k y ((B.ρ g).hom x))) (f.hom.hom (TensorPr …
        -/
        simp only [← map_mul, mul_inv_cancel, map_one]
        /-
          case hf.h
          k G : Type u
          inst✝¹ : CommRing k
          inst✝ : Group G
          A✝ B✝ C✝ A B C : Rep k G
          f : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj A B) C
          g : ↑(MonCat.of G)
          x : ↑B.V
          y : CoeSort.coe A
          ⊢ Eq (f.hom.hom (TensorProduct.tmul k y ((B.ρ g).hom x))) (f.hom.hom (TensorPr …
        -/
        rfl }
        /-
          🎉 no goals
        -/
  invFun f :=
    { hom := ModuleCat.ofHom <| TensorProduct.uncurry k _ _ _ f.hom.hom.flip
      comm := fun g => ModuleCat.hom_ext <| TensorProduct.ext' fun x y => by
          /- Porting note: rest of broken proof was
        dsimp only [MonoidalCategory.tensorLeft_obj, ModuleCat.comp_def, LinearMap.comp_apply,
          tensor_ρ, ModuleCat.MonoidalCategory.hom_apply, TensorProduct.map_tmul]
        simp only [TensorProduct.uncurry_apply f.hom.flip, LinearMap.flip_apply, Action_ρ_eq_ρ,
          hom_comm_apply f g y, Rep.ihom_obj_ρ_apply, LinearMap.comp_apply, ρ_inv_self_apply] -/
        change TensorProduct.uncurry k _ _ _ f.hom.hom.flip (A.ρ g x ⊗ₜ[k] B.ρ g y) =
          C.ρ g (TensorProduct.uncurry k _ _ _ f.hom.hom.flip (x ⊗ₜ[k] y))
        -- The next 3 tactics used to be `rw` before https://github.com/leanprover/lean4/pull/2644
        erw [TensorProduct.uncurry_apply, LinearMap.flip_apply, hom_comm_apply,
          Rep.ihom_obj_ρ_apply,
          LinearMap.comp_apply, LinearMap.comp_apply] --, ρ_inv_self_apply (A := C)]
        /-
          k G : Type u
          inst✝¹ : CommRing k
          inst✝ : Group G
          A✝ B✝ C✝ A B C : Rep k G
          f : Quiver.Hom B (A.ihom.obj C)
          g : ↑(MonCat.of G)
          x : ↑(((Action.functorCategoryEquivalence (ModuleCat k) (MonCat.of G)).symm.in …
          y : ↑(((Action.functorCategoryEquivalence (ModuleCat k) (MonCat.of G)).symm.in …
          ⊢ Eq ((C.ρ g) ((LinearMap.comp (f.hom.hom y) (A.ρ (Inv.inv g))) ((A.ρ g) x)))  …
        -/
        dsimp
        /-
          k G : Type u
          inst✝¹ : CommRing k
          inst✝ : Group G
          A✝ B✝ C✝ A B C : Rep k G
          f : Quiver.Hom B (A.ihom.obj C)
          g : ↑(MonCat.of G)
          x : ↑(((Action.functorCategoryEquivalence (ModuleCat k) (MonCat.of G)).symm.in …
          y : ↑(((Action.functorCategoryEquivalence (ModuleCat k) (MonCat.of G)).symm.in …
          ⊢ Eq ((C.ρ g) ((f.hom.hom y) ((A.ρ (Inv.inv g)) ((A.ρ g) x)))) ((C.ρ g) (((Ten …
        -/
        rw [ρ_inv_self_apply]
        /-
          k G : Type u
          inst✝¹ : CommRing k
          inst✝ : Group G
          A✝ B✝ C✝ A B C : Rep k G
          f : Quiver.Hom B (A.ihom.obj C)
          g : ↑(MonCat.of G)
          x : ↑(((Action.functorCategoryEquivalence (ModuleCat k) (MonCat.of G)).symm.in …
          y : ↑(((Action.functorCategoryEquivalence (ModuleCat k) (MonCat.of G)).symm.in …
          ⊢ Eq ((C.ρ g) ((f.hom.hom y) x)) ((C.ρ g) (((TensorProduct.uncurry k (CoeSort. …
        -/
        rfl }
        /-
          🎉 no goals
        -/
  left_inv _ := Action.Hom.ext (ModuleCat.hom_ext (TensorProduct.ext' fun _ _ => rfl))
                    /-
                      k G : Type u
                      inst✝¹ : CommRing k
                      inst✝ : Group G
                      A✝ B✝ C✝ A B C : Rep k G
                      f : Quiver.Hom B (A.ihom.obj C)
                      ⊢ Eq ((fun f => { hom := ModuleCat.ofHom (TensorProduct.curry f.hom.hom).flip, …
                    -/
  right_inv f := by ext; rfl
                         /-
                           🎉 no goals
                         -/


/-- Porting note: if we generate this with `@[simps]` the linter complains some types in the LHS
simplify. -/
theorem homEquiv_apply_hom (f : A ⊗ B ⟶ C) :
    (homEquiv A B C f).hom = ModuleCat.ofHom (TensorProduct.curry f.hom.hom).flip := rfl


/-- Porting note: if we generate this with `@[simps]` the linter complains some types in the LHS
simplify. -/
theorem homEquiv_symm_apply_hom (f : B ⟶ (Rep.ihom A).obj C) :
    ((homEquiv A B C).symm f).hom =
      ModuleCat.ofHom (TensorProduct.uncurry k A B C f.hom.hom.flip) := rfl


instance : MonoidalClosed (Rep k G) where
  closed A :=
    { rightAdj := Rep.ihom A
      adj := Adjunction.mkOfHomEquiv (
      { homEquiv := Rep.homEquiv A
        homEquiv_naturality_left_symm := fun _ _ => Action.Hom.ext
          (ModuleCat.hom_ext (TensorProduct.ext' fun _ _ => rfl))
        homEquiv_naturality_right := fun _ _ => Action.Hom.ext (ModuleCat.hom_ext (LinearMap.ext
          fun _ => LinearMap.ext fun _ => rfl)) })}


@[simp]
theorem ihom_obj_ρ_def (A B : Rep k G) : ((ihom A).obj B).ρ = ((Rep.ihom A).obj B).ρ :=
  rfl


@[simp]
theorem homEquiv_def (A B C : Rep k G) : (ihom.adjunction A).homEquiv B C = Rep.homEquiv A B C :=
  congrFun (congrFun (Adjunction.mkOfHomEquiv_homEquiv _) _) _


@[simp]
theorem ihom_ev_app_hom (A B : Rep k G) :
    Action.Hom.hom ((ihom.ev A).app B) = ModuleCat.ofHom
      (TensorProduct.uncurry k A (A →ₗ[k] B) B LinearMap.id.flip) := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A B : Rep k G
    ⊢ Eq ((CategoryTheory.ihom.ev A).app B).hom (ModuleCat.ofHom ((TensorProduct.u …
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


@[simp] theorem ihom_coev_app_hom (A B : Rep k G) :
    Action.Hom.hom ((ihom.coev A).app B) = ModuleCat.ofHom (TensorProduct.mk k _ _).flip :=
  ModuleCat.hom_ext <| LinearMap.ext fun _ => LinearMap.ext fun _ => rfl


/-- There is a `k`-linear isomorphism between the sets of representation morphisms`Hom(A ⊗ B, C)`
and `Hom(B, Homₖ(A, C))`. -/
def MonoidalClosed.linearHomEquiv : (A ⊗ B ⟶ C) ≃ₗ[k] B ⟶ A ⟶[Rep k G] C :=
  { (ihom.adjunction A).homEquiv _ _ with
    map_add' := fun _ _ => rfl
    map_smul' := fun _ _ => rfl }


/-- There is a `k`-linear isomorphism between the sets of representation morphisms`Hom(A ⊗ B, C)`
and `Hom(A, Homₖ(B, C))`. -/
def MonoidalClosed.linearHomEquivComm : (A ⊗ B ⟶ C) ≃ₗ[k] A ⟶ B ⟶[Rep k G] C :=
  Linear.homCongr k (β_ A B) (Iso.refl _) ≪≫ₗ MonoidalClosed.linearHomEquiv _ _ _


@[simp, nolint simpNF]
theorem MonoidalClosed.linearHomEquiv_hom (f : A ⊗ B ⟶ C) :
    (MonoidalClosed.linearHomEquiv A B C f).hom =
      ModuleCat.ofHom (TensorProduct.curry f.hom.hom).flip :=
  rfl

-- `simpNF` times out

@[simp, nolint simpNF]
theorem MonoidalClosed.linearHomEquivComm_hom (f : A ⊗ B ⟶ C) :
    (MonoidalClosed.linearHomEquivComm A B C f).hom =
      ModuleCat.ofHom (TensorProduct.curry f.hom.hom) :=
  rfl


theorem MonoidalClosed.linearHomEquiv_symm_hom (f : B ⟶ A ⟶[Rep k G] C) :
    ((MonoidalClosed.linearHomEquiv A B C).symm f).hom =
      ModuleCat.ofHom (TensorProduct.uncurry k A B C f.hom.hom.flip) := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A B C : Rep k G
    f : Quiver.Hom B ((CategoryTheory.ihom A).obj C)
    ⊢ Eq ((Rep.MonoidalClosed.linearHomEquiv A B C).symm f).hom (ModuleCat.ofHom ( …
  -/
  simp [linearHomEquiv]
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A B C : Rep k G
    f : Quiver.Hom B ((CategoryTheory.ihom A).obj C)
    ⊢ Eq ((A.homEquiv B C).symm f).hom (ModuleCat.ofHom ((TensorProduct.uncurry k  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem MonoidalClosed.linearHomEquivComm_symm_hom (f : A ⟶ B ⟶[Rep k G] C) :
    ((MonoidalClosed.linearHomEquivComm A B C).symm f).hom =
      ModuleCat.ofHom (TensorProduct.uncurry k A B C f.hom.hom) :=
  ModuleCat.hom_ext <| TensorProduct.ext' fun _ _ => rfl


/-- Tautological isomorphism to help Lean in typechecking. -/
def repOfTprodIso : Rep.of (ρ.tprod τ) ≅ Rep.of ρ ⊗ Rep.of τ :=
  Iso.refl _


theorem repOfTprodIso_apply (x : TensorProduct k V W) : (repOfTprodIso ρ τ).hom.hom x = x :=
  rfl


theorem repOfTprodIso_inv_apply (x : TensorProduct k V W) : (repOfTprodIso ρ τ).inv.hom x = x :=
  rfl


/-- Auxiliary lemma for `toModuleMonoidAlgebra`. -/
theorem to_Module_monoidAlgebra_map_aux {k G : Type*} [CommRing k] [Monoid G] (V W : Type*)
    [AddCommGroup V] [AddCommGroup W] [Module k V] [Module k W] (ρ : G →* V →ₗ[k] V)
    (σ : G →* W →ₗ[k] W) (f : V →ₗ[k] W) (w : ∀ g : G, f.comp (ρ g) = (σ g).comp f)
    (r : MonoidAlgebra k G) (x : V) :
    f ((((MonoidAlgebra.lift k G (V →ₗ[k] V)) ρ) r) x) =
      (((MonoidAlgebra.lift k G (W →ₗ[k] W)) σ) r) (f x) := by
  /-
    k : Type u_1
    G : Type u_2
    inst✝⁵ : CommRing k
    inst✝⁴ : Monoid G
    V : Type u_3
    W : Type u_4
    inst✝³ : AddCommGroup V
    inst✝² : AddCommGroup W
    inst✝¹ : Module k V
    inst✝ : Module k W
    ρ : MonoidHom G (LinearMap (RingHom.id k) V V)
    σ : MonoidHom G (LinearMap (RingHom.id k) W W)
    f : LinearMap (RingHom.id k) V W
    w : ∀ (g : G), Eq (f.comp (ρ g)) ((σ g).comp f)
    r : MonoidAlgebra k G
    x : V
    ⊢ Eq (f ((((MonoidAlgebra.lift k G (LinearMap (RingHom.id k) V V)) ρ) r) x)) ( …
  -/
  apply MonoidAlgebra.induction_on r
    /-
      case hM
      k : Type u_1
      G : Type u_2
      inst✝⁵ : CommRing k
      inst✝⁴ : Monoid G
      V : Type u_3
      W : Type u_4
      inst✝³ : AddCommGroup V
      inst✝² : AddCommGroup W
      inst✝¹ : Module k V
      inst✝ : Module k W
      ρ : MonoidHom G (LinearMap (RingHom.id k) V V)
      σ : MonoidHom G (LinearMap (RingHom.id k) W W)
      f : LinearMap (RingHom.id k) V W
      w : ∀ (g : G), Eq (f.comp (ρ g)) ((σ g).comp f)
      r : MonoidAlgebra k G
      x : V
      ⊢ ∀ (g : G), Eq (f ((((MonoidAlgebra.lift k G (LinearMap (RingHom.id k) V V))  …
    -/
  · intro g
    /-
      case hM
      k : Type u_1
      G : Type u_2
      inst✝⁵ : CommRing k
      inst✝⁴ : Monoid G
      V : Type u_3
      W : Type u_4
      inst✝³ : AddCommGroup V
      inst✝² : AddCommGroup W
      inst✝¹ : Module k V
      inst✝ : Module k W
      ρ : MonoidHom G (LinearMap (RingHom.id k) V V)
      σ : MonoidHom G (LinearMap (RingHom.id k) W W)
      f : LinearMap (RingHom.id k) V W
      w : ∀ (g : G), Eq (f.comp (ρ g)) ((σ g).comp f)
      r : MonoidAlgebra k G
      x : V
      g : G
      ⊢ Eq (f ((((MonoidAlgebra.lift k G (LinearMap (RingHom.id k) V V)) ρ) ((Monoid …
    -/
    simp only [one_smul, MonoidAlgebra.lift_single, MonoidAlgebra.of_apply]
    /-
      case hM
      k : Type u_1
      G : Type u_2
      inst✝⁵ : CommRing k
      inst✝⁴ : Monoid G
      V : Type u_3
      W : Type u_4
      inst✝³ : AddCommGroup V
      inst✝² : AddCommGroup W
      inst✝¹ : Module k V
      inst✝ : Module k W
      ρ : MonoidHom G (LinearMap (RingHom.id k) V V)
      σ : MonoidHom G (LinearMap (RingHom.id k) W W)
      f : LinearMap (RingHom.id k) V W
      w : ∀ (g : G), Eq (f.comp (ρ g)) ((σ g).comp f)
      r : MonoidAlgebra k G
      x : V
      g : G
      ⊢ Eq (f ((ρ g) x)) ((σ g) (f x))
    -/
    exact LinearMap.congr_fun (w g) x
    /-
      🎉 no goals
    -/
    /-
      case hadd
      k : Type u_1
      G : Type u_2
      inst✝⁵ : CommRing k
      inst✝⁴ : Monoid G
      V : Type u_3
      W : Type u_4
      inst✝³ : AddCommGroup V
      inst✝² : AddCommGroup W
      inst✝¹ : Module k V
      inst✝ : Module k W
      ρ : MonoidHom G (LinearMap (RingHom.id k) V V)
      σ : MonoidHom G (LinearMap (RingHom.id k) W W)
      f : LinearMap (RingHom.id k) V W
      w : ∀ (g : G), Eq (f.comp (ρ g)) ((σ g).comp f)
      r : MonoidAlgebra k G
      x : V
      ⊢ ∀ (f_1 g : MonoidAlgebra k G), Eq (f ((((MonoidAlgebra.lift k G (LinearMap ( …
    -/
  · intro g h gw hw; simp only [map_add, add_left_inj, LinearMap.add_apply, hw, gw]
                     /-
                       🎉 no goals
                     -/
    /-
      case hsmul
      k : Type u_1
      G : Type u_2
      inst✝⁵ : CommRing k
      inst✝⁴ : Monoid G
      V : Type u_3
      W : Type u_4
      inst✝³ : AddCommGroup V
      inst✝² : AddCommGroup W
      inst✝¹ : Module k V
      inst✝ : Module k W
      ρ : MonoidHom G (LinearMap (RingHom.id k) V V)
      σ : MonoidHom G (LinearMap (RingHom.id k) W W)
      f : LinearMap (RingHom.id k) V W
      w : ∀ (g : G), Eq (f.comp (ρ g)) ((σ g).comp f)
      r : MonoidAlgebra k G
      x : V
      ⊢ ∀ (r : k) (f_1 : MonoidAlgebra k G), Eq (f ((((MonoidAlgebra.lift k G (Linea …
    -/
  · intro r g w
    /-
      case hsmul
      k : Type u_1
      G : Type u_2
      inst✝⁵ : CommRing k
      inst✝⁴ : Monoid G
      V : Type u_3
      W : Type u_4
      inst✝³ : AddCommGroup V
      inst✝² : AddCommGroup W
      inst✝¹ : Module k V
      inst✝ : Module k W
      ρ : MonoidHom G (LinearMap (RingHom.id k) V V)
      σ : MonoidHom G (LinearMap (RingHom.id k) W W)
      f : LinearMap (RingHom.id k) V W
      w✝ : ∀ (g : G), Eq (f.comp (ρ g)) ((σ g).comp f)
      r✝ : MonoidAlgebra k G
      x : V
      r : k
      g : MonoidAlgebra k G
      w : Eq (f ((((MonoidAlgebra.lift k G (LinearMap (RingHom.id k) V V)) ρ) g) x)) …
      ⊢ Eq (f ((((MonoidAlgebra.lift k G (LinearMap (RingHom.id k) V V)) ρ) (HSMul.h …
    -/
    simp only [map_smul, w, RingHom.id_apply, LinearMap.smul_apply, LinearMap.map_smulₛₗ]
    /-
      🎉 no goals
    -/


/-- Auxiliary definition for `toModuleMonoidAlgebra`. -/
def toModuleMonoidAlgebraMap {V W : Rep k G} (f : V ⟶ W) :
    ModuleCat.of (MonoidAlgebra k G) V.ρ.asModule ⟶ ModuleCat.of (MonoidAlgebra k G) W.ρ.asModule :=
  ModuleCat.ofHom
    { f.hom.hom with
      map_smul' := fun r x => to_Module_monoidAlgebra_map_aux V.V W.V V.ρ W.ρ f.hom.hom
        (fun g => ModuleCat.hom_ext_iff.mp (f.comm g)) r x }


/-- Functorially convert a representation of `G` into a module over `MonoidAlgebra k G`. -/
def toModuleMonoidAlgebra : Rep k G ⥤ ModuleCat.{u} (MonoidAlgebra k G) where
  obj V := ModuleCat.of _ V.ρ.asModule
  map f := toModuleMonoidAlgebraMap f


/-- Functorially convert a module over `MonoidAlgebra k G` into a representation of `G`. -/
def ofModuleMonoidAlgebra : ModuleCat.{u} (MonoidAlgebra k G) ⥤ Rep k G where
  obj M := Rep.of (Representation.ofModule M)
  map f :=
    { hom := ModuleCat.ofHom
        { f.hom with
          map_smul' := fun r x => f.hom.map_smul (algebraMap k _ r) x }
                          /-
                            k G : Type u
                            inst✝¹ : CommRing k
                            inst✝ : Monoid G
                            X✝ Y✝ : ModuleCat (MonoidAlgebra k G)
                            f : Quiver.Hom X✝ Y✝
                            g : ↑(MonCat.of G)
                            ⊢ Eq
                                (CategoryTheory.CategoryStruct.comp (((fun M => Rep.of (Representation.ofM …
                                  (ModuleCat.ofHom
                                    (let __src := f.hom;
                                    { toAddHom := __src.toAddHom, map_smul' := ⋯ })))
                                (CategoryTheory.CategoryStruct.comp
                                  (ModuleCat.ofHom
                                    (let __src := f.hom;
                                    { toAddHom := __src.toAddHom, map_smul' := ⋯ }))
                                  (((fun M => Rep.of (Representation.ofModule ↑M)) Y✝).ρ g))
                          -/
      comm := fun g => by ext; apply f.hom.map_smul }
                               /-
                                 🎉 no goals
                               -/


theorem ofModuleMonoidAlgebra_obj_coe (M : ModuleCat.{u} (MonoidAlgebra k G)) :
    (ofModuleMonoidAlgebra.obj M : Type u) = RestrictScalars k (MonoidAlgebra k G) M :=
  rfl


theorem ofModuleMonoidAlgebra_obj_ρ (M : ModuleCat.{u} (MonoidAlgebra k G)) :
    (ofModuleMonoidAlgebra.obj M).ρ = Representation.ofModule M :=
  rfl


/-- Auxiliary definition for `equivalenceModuleMonoidAlgebra`. -/
def counitIsoAddEquiv {M : ModuleCat.{u} (MonoidAlgebra k G)} :
    (ofModuleMonoidAlgebra ⋙ toModuleMonoidAlgebra).obj M ≃+ M := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    M : ModuleCat (MonoidAlgebra k G)
    ⊢ AddEquiv ↑((Rep.ofModuleMonoidAlgebra.comp Rep.toModuleMonoidAlgebra).obj M) …
  -/
  dsimp [ofModuleMonoidAlgebra, toModuleMonoidAlgebra]
  exact (Representation.ofModule M).asModuleEquiv.trans
    (RestrictScalars.addEquiv k (MonoidAlgebra k G) _)


/-- Auxiliary definition for `equivalenceModuleMonoidAlgebra`. -/
def unitIsoAddEquiv {V : Rep k G} : V ≃+ (toModuleMonoidAlgebra ⋙ ofModuleMonoidAlgebra).obj V := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    V : Rep k G
    ⊢ AddEquiv (CoeSort.coe V) (CoeSort.coe ((Rep.toModuleMonoidAlgebra.comp Rep.o …
  -/
  dsimp [ofModuleMonoidAlgebra, toModuleMonoidAlgebra]
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    V : Rep k G
    ⊢ AddEquiv (CoeSort.coe V) (RestrictScalars k (MonoidAlgebra k G) V.ρ.asModule)
  -/
  refine V.ρ.asModuleEquiv.symm.trans ?_
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    V : Rep k G
    ⊢ AddEquiv V.ρ.asModule (RestrictScalars k (MonoidAlgebra k G) V.ρ.asModule)
  -/
  exact (RestrictScalars.addEquiv _ _ _).symm
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `equivalenceModuleMonoidAlgebra`. -/
def counitIso (M : ModuleCat.{u} (MonoidAlgebra k G)) :
    (ofModuleMonoidAlgebra ⋙ toModuleMonoidAlgebra).obj M ≅ M :=
  LinearEquiv.toModuleIso
    { counitIsoAddEquiv with
      map_smul' := fun r x => by
        set_option tactic.skipAssignedInstances false in
        dsimp [counitIsoAddEquiv]
        /- Porting note: rest of broken proof was `simp`. -/
        rw [AddEquiv.trans_apply]
        rw [AddEquiv.trans_apply]
        erw [@Representation.ofModule_asAlgebraHom_apply_apply k G _ _ _ _ (_)]
        exact AddEquiv.symm_apply_apply _ _}


theorem unit_iso_comm (V : Rep k G) (g : G) (x : V) :
    unitIsoAddEquiv ((V.ρ g).toFun x) = ((ofModuleMonoidAlgebra.obj
      (toModuleMonoidAlgebra.obj V)).ρ g).toFun (unitIsoAddEquiv x) := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    V : Rep k G
    g : G
    x : CoeSort.coe V
    ⊢ Eq (Rep.unitIsoAddEquiv ((V.ρ g).toFun x)) (((Rep.ofModuleMonoidAlgebra.obj  …
  -/
  dsimp [unitIsoAddEquiv, ofModuleMonoidAlgebra, toModuleMonoidAlgebra]
  simp only [AddEquiv.apply_eq_iff_eq, AddEquiv.apply_symm_apply,
    Representation.asModuleEquiv_symm_map_rho, Representation.ofModule_asModule_act]


/-- Auxiliary definition for `equivalenceModuleMonoidAlgebra`. -/
def unitIso (V : Rep k G) : V ≅ (toModuleMonoidAlgebra ⋙ ofModuleMonoidAlgebra).obj V :=
  Action.mkIso
    (LinearEquiv.toModuleIso
      { unitIsoAddEquiv with
        map_smul' := fun r x => by
          /-
            k G : Type u
            inst✝¹ : CommRing k
            inst✝ : Monoid G
            V : Rep k G
            r : k
            x : ↑V.1
            ⊢ Eq ({ toFun := __src✝.toFun, map_add' := ⋯ }.toFun (HSMul.hSMul r x)) (HSMul …
          -/
          dsimp [unitIsoAddEquiv]
/- Porting note: rest of broken proof was
          simp only [Representation.asModuleEquiv_symm_map_smul,
            RestrictScalars.addEquiv_symm_map_algebraMap_smul] -/
          -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
          erw [AddEquiv.trans_apply,
            Representation.asModuleEquiv_symm_map_smul]
          /-
            k G : Type u
            inst✝¹ : CommRing k
            inst✝ : Monoid G
            V : Rep k G
            r : k
            x : ↑V.1
            ⊢ Eq (HSMul.hSMul ((algebraMap k (MonoidAlgebra k G)) r) (V.ρ.asModuleEquiv.sy …
          -/
          rfl })
          /-
            🎉 no goals
          -/
                /-
                  k G : Type u
                  inst✝¹ : CommRing k
                  inst✝ : Monoid G
                  V : Rep k G
                  g : ↑(MonCat.of G)
                  ⊢ Eq
                      (CategoryTheory.CategoryStruct.comp (V.ρ g)
                        (let __src := Rep.unitIsoAddEquiv;
                            { toFun := __src.toFun, map_add' := ⋯, map_smul' := ⋯, invFun := __s …
                      (CategoryTheory.CategoryStruct.comp
                        (let __src := Rep.unitIsoAddEquiv;
                            { toFun := __src.toFun, map_add' := ⋯, map_smul' := ⋯, invFun := __s …
                        (((Rep.toModuleMonoidAlgebra.comp Rep.ofModuleMonoidAlgebra).obj V).ρ g))
                -/
    fun g => by ext; apply unit_iso_comm
                     /-
                       🎉 no goals
                     -/


/-- The categorical equivalence `Rep k G ≌ ModuleCat (MonoidAlgebra k G)`. -/
def equivalenceModuleMonoidAlgebra : Rep k G ≌ ModuleCat.{u} (MonoidAlgebra k G) where
  functor := toModuleMonoidAlgebra
  inverse := ofModuleMonoidAlgebra
                                                          /-
                                                            k G : Type u
                                                            inst✝¹ : CommRing k
                                                            inst✝ : Monoid G
                                                            ⊢ ∀ {X Y : Rep k G} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.co …
                                                          -/
  unitIso := NatIso.ofComponents (fun V => unitIso V) (by aesop_cat)
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                              /-
                                                                k G : Type u
                                                                inst✝¹ : CommRing k
                                                                inst✝ : Monoid G
                                                                ⊢ ∀ {X Y : ModuleCat (MonoidAlgebra k G)} (f : Quiver.Hom X Y), Eq (CategoryTh …
                                                              -/
  counitIso := NatIso.ofComponents (fun M => counitIso M) (by aesop_cat)
                                                              /-
                                                                🎉 no goals
                                                              -/

-- TODO Verify that the equivalence with `ModuleCat (MonoidAlgebra k G)` is a monoidal functor.

