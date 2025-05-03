instance : HasBinaryBiproducts (ModuleCat.{v} R) :=
  HasBinaryBiproducts.of_hasBinaryProducts


instance : HasFiniteBiproducts (ModuleCat.{v} R) :=
  HasFiniteBiproducts.of_hasFiniteProducts

-- We now construct explicit limit data,
-- so we can compare the biproducts to the usual unbundled constructions.

/-- Construct limit data for a binary product in `ModuleCat R`, using `ModuleCat.of R (M × N)`.
-/
@[simps cone_pt isLimit_lift]
def binaryProductLimitCone (M N : ModuleCat.{v} R) : Limits.LimitCone (pair M N) where
  cone :=
    { pt := ModuleCat.of R (M × N)
      π :=
        { app := fun j =>
            Discrete.casesOn j fun j =>
              WalkingPair.casesOn j (ofHom <| LinearMap.fst R M N) (ofHom <| LinearMap.snd R M N)
                           /-
                             R : Type u
                             inst✝ : Ring R
                             M N : ModuleCat R
                             ⊢ ∀ ⦃X Y : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair⦄ (f : Qui …
                           -/
                                                       /-
                                                         🎉 no goals
                                                       -/
          naturality := by rintro ⟨⟨⟩⟩ ⟨⟨⟩⟩ ⟨⟨⟨⟩⟩⟩ <;> rfl } }
                                                       /-
                                                         🎉 no goals
                                                       -/
  isLimit :=
    { lift := fun s => ofHom <| LinearMap.prod
        (s.π.app ⟨WalkingPair.left⟩).hom
        (s.π.app ⟨WalkingPair.right⟩).hom
                /-
                  R : Type u
                  inst✝ : Ring R
                  M N : ModuleCat R
                  ⊢ ∀ (s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair M N)) (j : Cat …
                -/
                                       /-
                                         🎉 no goals
                                       -/
      fac := by rintro s (⟨⟩ | ⟨⟩) <;> rfl
                                       /-
                                         🎉 no goals
                                       -/
      uniq := fun s m w => by
        /-
          R : Type u
          inst✝ : Ring R
          M N : ModuleCat R
          s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair M N)
          m : Quiver.Hom s.pt { pt := ModuleCat.of R (Prod ↑M ↑N), π := { app := fun j = …
          w : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
          ⊢ Eq m ((fun s => ModuleCat.ofHom ((s.π.app { as := CategoryTheory.Limits.Walk …
        -/
        simp_rw [← w ⟨WalkingPair.left⟩, ← w ⟨WalkingPair.right⟩]
        /-
          R : Type u
          inst✝ : Ring R
          M N : ModuleCat R
          s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair M N)
          m : Quiver.Hom s.pt { pt := ModuleCat.of R (Prod ↑M ↑N), π := { app := fun j = …
          w : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
          ⊢ Eq m (ModuleCat.ofHom ((CategoryTheory.CategoryStruct.comp m (ModuleCat.ofHo …
        -/
        rfl }
        /-
          🎉 no goals
        -/


@[simp]
theorem binaryProductLimitCone_cone_π_app_left (M N : ModuleCat.{v} R) :
    (binaryProductLimitCone M N).cone.π.app ⟨WalkingPair.left⟩ = ofHom (LinearMap.fst R M N) :=
  rfl


@[simp]
theorem binaryProductLimitCone_cone_π_app_right (M N : ModuleCat.{v} R) :
    (binaryProductLimitCone M N).cone.π.app ⟨WalkingPair.right⟩ = ofHom (LinearMap.snd R M N) :=
  rfl


/-- We verify that the biproduct in `ModuleCat R` is isomorphic to
the cartesian product of the underlying types:
-/
noncomputable def biprodIsoProd (M N : ModuleCat.{v} R) :
    (M ⊞ N : ModuleCat.{v} R) ≅ ModuleCat.of R (M × N) :=
  IsLimit.conePointUniqueUpToIso (BinaryBiproduct.isLimit M N) (binaryProductLimitCone M N).isLimit


@[simp, elementwise]
theorem biprodIsoProd_inv_comp_fst (M N : ModuleCat.{v} R) :
    (biprodIsoProd M N).inv ≫ biprod.fst = ofHom (LinearMap.fst R M N) :=
  IsLimit.conePointUniqueUpToIso_inv_comp _ _ (Discrete.mk WalkingPair.left)


@[simp, elementwise]
theorem biprodIsoProd_inv_comp_snd (M N : ModuleCat.{v} R) :
    (biprodIsoProd M N).inv ≫ biprod.snd = ofHom (LinearMap.snd R M N) :=
  IsLimit.conePointUniqueUpToIso_inv_comp _ _ (Discrete.mk WalkingPair.right)


/-- The map from an arbitrary cone over an indexed family of abelian groups
to the cartesian product of those groups.
-/
@[simps]
def lift (s : Fan f) : s.pt ⟶ ModuleCat.of R (∀ j, f j) :=
  ofHom
  { toFun := fun x j => s.π.app ⟨j⟩ x
    map_add' := fun x y => by
      /-
        R : Type u
        inst✝ : Ring R
        J : Type w
        f : J → ModuleCat R
        s : CategoryTheory.Limits.Fan f
        x y : ↑s.1
        ⊢ Eq ((fun x j => (s.π.app { as := j }).hom x) (HAdd.hAdd x y)) (HAdd.hAdd ((f …
      -/
      simp only [Functor.const_obj_obj, map_add]
      /-
        R : Type u
        inst✝ : Ring R
        J : Type w
        f : J → ModuleCat R
        s : CategoryTheory.Limits.Fan f
        x y : ↑s.1
        ⊢ Eq (fun j => HAdd.hAdd ((s.π.app { as := j }).hom x) ((s.π.app { as := j }). …
      -/
      rfl
      /-
        🎉 no goals
      -/
    map_smul' := fun r x => by
      /-
        R : Type u
        inst✝ : Ring R
        J : Type w
        f : J → ModuleCat R
        s : CategoryTheory.Limits.Fan f
        r : R
        x : ↑s.1
        ⊢ Eq ({ toFun := fun x j => (s.π.app { as := j }).hom x, map_add' := ⋯ }.toFun …
      -/
      simp only [Functor.const_obj_obj, map_smul]
      /-
        R : Type u
        inst✝ : Ring R
        J : Type w
        f : J → ModuleCat R
        s : CategoryTheory.Limits.Fan f
        r : R
        x : ↑s.1
        ⊢ Eq (fun j => HSMul.hSMul r ((s.π.app { as := j }).hom x)) (HSMul.hSMul ((Rin …
      -/
      rfl }
      /-
        🎉 no goals
      -/


/-- Construct limit data for a product in `ModuleCat R`, using `ModuleCat.of R (∀ j, F.obj j)`.
-/
@[simps]
def productLimitCone : Limits.LimitCone (Discrete.functor f) where
  cone :=
    { pt := ModuleCat.of R (∀ j, f j)
      π := Discrete.natTrans fun j => ofHom (LinearMap.proj j.as : (∀ j, f j) →ₗ[R] f j.as) }
  isLimit :=
    { lift := lift.{_, v} f
      fac := fun _ _ => rfl
      uniq := fun s m w => by
        /-
          R : Type u
          inst✝ : Ring R
          J : Type w
          f : J → ModuleCat R
          s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
          m : Quiver.Hom s.pt { pt := ModuleCat.of R ((j : J) → ↑(f j)), π := CategoryTh …
          w : ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp  …
          ⊢ Eq m (ModuleCat.HasLimit.lift f s)
        -/
        ext x j
        /-
          case hf.h.h
          R : Type u
          inst✝ : Ring R
          J : Type w
          f : J → ModuleCat R
          s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
          m : Quiver.Hom s.pt { pt := ModuleCat.of R ((j : J) → ↑(f j)), π := CategoryTh …
          w : ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp  …
          x : ↑s.pt
          j : J
          ⊢ Eq (m.hom x j) ((ModuleCat.HasLimit.lift f s).hom x j)
        -/
        exact congr_arg (fun g : s.pt ⟶ f j => (g : s.pt → f j) x) (w ⟨j⟩) }
        /-
          🎉 no goals
        -/


/-- We verify that the biproduct we've just defined is isomorphic to the `ModuleCat R` structure
on the dependent function type.
-/
noncomputable def biproductIsoPi [Finite J] (f : J → ModuleCat.{v} R) :
    ((⨁ f) : ModuleCat.{v} R) ≅ ModuleCat.of R (∀ j, f j) :=
  IsLimit.conePointUniqueUpToIso (biproduct.isLimit f) (productLimitCone f).isLimit


@[simp, elementwise]
theorem biproductIsoPi_inv_comp_π [Finite J] (f : J → ModuleCat.{v} R) (j : J) :
    (biproductIsoPi f).inv ≫ biproduct.π f j = ofHom (LinearMap.proj j : (∀ j, f j) →ₗ[R] f j) :=
  IsLimit.conePointUniqueUpToIso_inv_comp _ _ (Discrete.mk j)


/-- The isomorphism `A × B ≃ₗ[R] M` coming from a right split exact sequence `0 ⟶ A ⟶ M ⟶ B ⟶ 0`
of modules. -/
noncomputable def lequivProdOfRightSplitExact {f : B →ₗ[R] M} (hj : Function.Injective j)
    (exac : LinearMap.range j = LinearMap.ker g) (h : g.comp f = LinearMap.id) : (A × B) ≃ₗ[R] M :=
  ((ShortComplex.Splitting.ofExactOfSection _
    (ShortComplex.Exact.moduleCat_of_range_eq_ker (ModuleCat.ofHom j)
    (ModuleCat.ofHom g) exac) (ofHom f) (hom_ext h)
        /-
          R : Type u
          A M B : Type v
          inst✝⁶ : Ring R
          inst✝⁵ : AddCommGroup A
          inst✝⁴ : Module R A
          inst✝³ : AddCommGroup B
          inst✝² : Module R B
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          j : LinearMap (RingHom.id R) A M
          g : LinearMap (RingHom.id R) M B
          f : LinearMap (RingHom.id R) B M
          hj : Function.Injective ⇑j
          exac : Eq (LinearMap.range j) (LinearMap.ker g)
          h : Eq (g.comp f) LinearMap.id
          ⊢ CategoryTheory.Mono (CategoryTheory.ShortComplex.moduleCatMkOfKerLERange (Mo …
        -/
    (by simpa only [ModuleCat.mono_iff_injective])).isoBinaryBiproduct ≪≫
        /-
          🎉 no goals
        -/
    biprodIsoProd _ _ ).symm.toLinearEquiv


/-- The isomorphism `A × B ≃ₗ[R] M` coming from a left split exact sequence `0 ⟶ A ⟶ M ⟶ B ⟶ 0`
of modules. -/
noncomputable def lequivProdOfLeftSplitExact {f : M →ₗ[R] A} (hg : Function.Surjective g)
    (exac : LinearMap.range j = LinearMap.ker g) (h : f.comp j = LinearMap.id) : (A × B) ≃ₗ[R] M :=
  ((ShortComplex.Splitting.ofExactOfRetraction _
    (ShortComplex.Exact.moduleCat_of_range_eq_ker (ModuleCat.ofHom j)
    (ModuleCat.ofHom g) exac) (ModuleCat.ofHom f) (hom_ext h)
        /-
          R : Type u
          A M B : Type v
          inst✝⁶ : Ring R
          inst✝⁵ : AddCommGroup A
          inst✝⁴ : Module R A
          inst✝³ : AddCommGroup B
          inst✝² : Module R B
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          j : LinearMap (RingHom.id R) A M
          g : LinearMap (RingHom.id R) M B
          f : LinearMap (RingHom.id R) M A
          hg : Function.Surjective ⇑g
          exac : Eq (LinearMap.range j) (LinearMap.ker g)
          h : Eq (f.comp j) LinearMap.id
          ⊢ CategoryTheory.Epi (CategoryTheory.ShortComplex.moduleCatMkOfKerLERange (Mod …
        -/
    (by simpa only [ModuleCat.epi_iff_surjective] using hg)).isoBinaryBiproduct ≪≫
        /-
          🎉 no goals
        -/
    biprodIsoProd _ _).symm.toLinearEquiv


