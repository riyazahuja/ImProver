noncomputable instance : (forget₂ (ModuleCat.{v} R) Ab).PreservesHomology where


/-- Constructor for short complexes in `ModuleCat.{v} R` taking as inputs
linear maps `f` and `g` and the vanishing of their composition. -/
@[simps]
def moduleCatMk {X₁ X₂ X₃ : Type v} [AddCommGroup X₁] [AddCommGroup X₂] [AddCommGroup X₃]
    [Module R X₁] [Module R X₂] [Module R X₃] (f : X₁ →ₗ[R] X₂) (g : X₂ →ₗ[R] X₃)
    (hfg : g.comp f = 0) : ShortComplex (ModuleCat.{v} R) :=
  ShortComplex.mk (ModuleCat.ofHom f) (ModuleCat.ofHom g) (ModuleCat.hom_ext hfg)


@[simp]
lemma moduleCat_zero_apply (x : S.X₁) : S.g (S.f x) = 0 :=
  S.zero_apply x


lemma moduleCat_exact_iff :
    S.Exact ↔ ∀ (x₂ : S.X₂) (_ : S.g x₂ = 0), ∃ (x₁ : S.X₁), S.f x₁ = x₂ :=
  S.exact_iff_of_concreteCategory


lemma moduleCat_exact_iff_ker_sub_range :
    S.Exact ↔ LinearMap.ker S.g.hom ≤ LinearMap.range S.f.hom := by
  /-
    R : Type u
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    ⊢ Iff S.Exact (LE.le (LinearMap.ker S.g.hom) (LinearMap.range S.f.hom))
  -/
  rw [moduleCat_exact_iff]
  /-
    R : Type u
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    ⊢ Iff (∀ (x₂ : ↑S.X₂), Eq (S.g.hom x₂) 0 → Exists fun x₁ => Eq (S.f.hom x₁) x₂ …
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma moduleCat_exact_iff_range_eq_ker :
    S.Exact ↔ LinearMap.range S.f.hom = LinearMap.ker S.g.hom := by
  /-
    R : Type u
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    ⊢ Iff S.Exact (Eq (LinearMap.range S.f.hom) (LinearMap.ker S.g.hom))
  -/
  rw [moduleCat_exact_iff_ker_sub_range]
  /-
    R : Type u
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    ⊢ Iff (LE.le (LinearMap.ker S.g.hom) (LinearMap.range S.f.hom)) (Eq (LinearMap …
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma Exact.moduleCat_range_eq_ker (hS : S.Exact) :
    LinearMap.range S.f.hom = LinearMap.ker S.g.hom := by
  /-
    R : Type u
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    ⊢ Eq (LinearMap.range S.f.hom) (LinearMap.ker S.g.hom)
  -/
  simpa only [moduleCat_exact_iff_range_eq_ker] using hS
  /-
    🎉 no goals
  -/


lemma ShortExact.moduleCat_injective_f (hS : S.ShortExact) :
    Function.Injective S.f :=
  hS.injective_f


lemma ShortExact.moduleCat_surjective_g (hS : S.ShortExact) :
    Function.Surjective S.g :=
  hS.surjective_g


lemma ShortExact.moduleCat_exact_iff_function_exact :
    S.Exact ↔ Function.Exact S.f S.g := by
  /-
    R : Type u
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    ⊢ Iff S.Exact (Function.Exact ⇑S.f.hom ⇑S.g.hom)
  -/
  rw [moduleCat_exact_iff_range_eq_ker, LinearMap.exact_iff]
  /-
    R : Type u
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    ⊢ Iff (Eq (LinearMap.range S.f.hom) (LinearMap.ker S.g.hom)) (Eq (LinearMap.ke …
  -/
  tauto
  /-
    🎉 no goals
  -/


/-- Constructor for short complexes in `ModuleCat.{v} R` taking as inputs
morphisms `f` and `g` and the assumption `LinearMap.range f ≤ LinearMap.ker g`. -/
@[simps]
def moduleCatMkOfKerLERange {X₁ X₂ X₃ : ModuleCat.{v} R} (f : X₁ ⟶ X₂) (g : X₂ ⟶ X₃)
    (hfg : LinearMap.range f.hom ≤ LinearMap.ker g.hom) : ShortComplex (ModuleCat.{v} R) :=
                          /-
                            R : Type u
                            inst✝ : Ring R
                            S : CategoryTheory.ShortComplex (ModuleCat R)
                            X₁ X₂ X₃ : ModuleCat R
                            f : Quiver.Hom X₁ X₂
                            g : Quiver.Hom X₂ X₃
                            hfg : LE.le (LinearMap.range f.hom) (LinearMap.ker g.hom)
                            ⊢ Eq (CategoryTheory.CategoryStruct.comp f g) 0
                          -/
  ShortComplex.mk f g (by aesop)
                          /-
                            🎉 no goals
                          -/


lemma Exact.moduleCat_of_range_eq_ker {X₁ X₂ X₃ : ModuleCat.{v} R}
    (f : X₁ ⟶ X₂) (g : X₂ ⟶ X₃) (hfg : LinearMap.range f.hom = LinearMap.ker g.hom) :
                                     /-
                                       R : Type u
                                       inst✝ : Ring R
                                       S : CategoryTheory.ShortComplex (ModuleCat R)
                                       X₁ X₂ X₃ : ModuleCat R
                                       f : Quiver.Hom X₁ X₂
                                       g : Quiver.Hom X₂ X₃
                                       hfg : Eq (LinearMap.range f.hom) (LinearMap.ker g.hom)
                                       ⊢ LE.le (LinearMap.range f.hom) (LinearMap.ker g.hom)
                                     -/
    (moduleCatMkOfKerLERange f g (by rw [hfg])).Exact := by
                                     /-
                                       🎉 no goals
                                     -/
  /-
    R : Type u
    inst✝ : Ring R
    X₁ X₂ X₃ : ModuleCat R
    f : Quiver.Hom X₁ X₂
    g : Quiver.Hom X₂ X₃
    hfg : Eq (LinearMap.range f.hom) (LinearMap.ker g.hom)
    ⊢ (CategoryTheory.ShortComplex.moduleCatMkOfKerLERange f g ⋯).Exact
  -/
  simpa only [moduleCat_exact_iff_range_eq_ker] using hfg
  /-
    🎉 no goals
  -/


/-- The canonical linear map `S.X₁ →ₗ[R] LinearMap.ker S.g` induced by `S.f`. -/
@[simps]
def moduleCatToCycles : S.X₁ →ₗ[R] LinearMap.ker S.g.hom where
  toFun x := ⟨S.f x, S.moduleCat_zero_apply x⟩
                     /-
                       R : Type u
                       inst✝ : Ring R
                       S : CategoryTheory.ShortComplex (ModuleCat R)
                       x y : ↑S.X₁
                       ⊢ Eq ((fun x => ⟨S.f.hom x, ⋯⟩) (HAdd.hAdd x y)) (HAdd.hAdd ((fun x => ⟨S.f.ho …
                     -/
  map_add' x y := by aesop
                     /-
                       🎉 no goals
                     -/
                      /-
                        R : Type u
                        inst✝ : Ring R
                        S : CategoryTheory.ShortComplex (ModuleCat R)
                        a : R
                        x : ↑S.X₁
                        ⊢ Eq ({ toFun := fun x => ⟨S.f.hom x, ⋯⟩, map_add' := ⋯ }.toFun (HSMul.hSMul a …
                      -/
  map_smul' a x := by aesop
                      /-
                        🎉 no goals
                      -/


/-- The homology of `S`, defined as the quotient of the kernel of `S.g` by
the image of `S.moduleCatToCycles` -/
abbrev moduleCatHomology :=
  ModuleCat.of R (LinearMap.ker S.g.hom ⧸ LinearMap.range S.moduleCatToCycles)


/-- The canonical map `ModuleCat.of R (LinearMap.ker S.g) ⟶ S.moduleCatHomology`. -/
abbrev moduleCatHomologyπ : ModuleCat.of R (LinearMap.ker S.g.hom) ⟶ S.moduleCatHomology :=
  ModuleCat.ofHom (LinearMap.range S.moduleCatToCycles).mkQ


/-- The explicit left homology data of a short complex of modules that is
given by a kernel and a quotient given by the `LinearMap` API. -/
@[simps K H i π]
def moduleCatLeftHomologyData : S.LeftHomologyData where
  K := ModuleCat.of R (LinearMap.ker S.g.hom)
  H := S.moduleCatHomology
  i := ModuleCat.ofHom (LinearMap.ker S.g.hom).subtype
  π := S.moduleCatHomologyπ
           /-
             R : Type u
             inst✝ : Ring R
             S : CategoryTheory.ShortComplex (ModuleCat R)
             ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.ofHom (LinearMap.ker S.g.h …
           -/
  wi := by aesop
           /-
             🎉 no goals
           -/
  hi := ModuleCat.kernelIsLimit _
           /-
             R : Type u
             inst✝ : Ring R
             S : CategoryTheory.ShortComplex (ModuleCat R)
             ⊢ Eq (CategoryTheory.CategoryStruct.comp ((ModuleCat.kernelIsLimit S.g).lift ( …
           -/
  wπ := by aesop
           /-
             🎉 no goals
           -/
  hπ := ModuleCat.cokernelIsColimit (ModuleCat.ofHom S.moduleCatToCycles)


@[simp]
lemma moduleCatLeftHomologyData_f' :
    S.moduleCatLeftHomologyData.f' = ModuleCat.ofHom S.moduleCatToCycles := rfl


instance : Epi S.moduleCatHomologyπ :=
  (inferInstance : Epi S.moduleCatLeftHomologyData.π)


/-- Given a short complex `S` of modules, this is the isomorphism between
the abstract `S.cycles` of the homology API and the more concrete description as
`LinearMap.ker S.g`. -/
noncomputable def moduleCatCyclesIso : S.cycles ≅ ModuleCat.of R (LinearMap.ker S.g.hom) :=
  S.moduleCatLeftHomologyData.cyclesIso


@[reassoc (attr := simp, elementwise)]
lemma moduleCatCyclesIso_hom_subtype :
    S.moduleCatCyclesIso.hom ≫ ModuleCat.ofHom (LinearMap.ker S.g.hom).subtype = S.iCycles :=
  S.moduleCatLeftHomologyData.cyclesIso_hom_comp_i


@[reassoc (attr := simp, elementwise)]
lemma moduleCatCyclesIso_inv_iCycles :
    S.moduleCatCyclesIso.inv ≫ S.iCycles = ModuleCat.ofHom (LinearMap.ker S.g.hom).subtype :=
  S.moduleCatLeftHomologyData.cyclesIso_inv_comp_iCycles


@[reassoc (attr := simp, elementwise)]
lemma toCycles_moduleCatCyclesIso_hom :
    S.toCycles ≫ S.moduleCatCyclesIso.hom = ModuleCat.ofHom S.moduleCatToCycles := by
  rw [← cancel_mono S.moduleCatLeftHomologyData.i, moduleCatLeftHomologyData_i,
    Category.assoc, S.moduleCatCyclesIso_hom_subtype, toCycles_i]
  /-
    R : Type u
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    ⊢ Eq S.f (CategoryTheory.CategoryStruct.comp (ModuleCat.ofHom S.moduleCatToCyc …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Given a short complex `S` of modules, this is the isomorphism between
the abstract `S.homology` of the homology API and the more explicit
quotient of `LinearMap.ker S.g` by the image of
`S.moduleCatToCycles : S.X₁ →ₗ[R] LinearMap.ker S.g`. -/
noncomputable def moduleCatHomologyIso :
    S.homology ≅ S.moduleCatHomology :=
  S.moduleCatLeftHomologyData.homologyIso


@[reassoc (attr := simp, elementwise)]
lemma π_moduleCatCyclesIso_hom :
    S.homologyπ ≫ S.moduleCatHomologyIso.hom =
      S.moduleCatCyclesIso.hom ≫ S.moduleCatHomologyπ :=
  S.moduleCatLeftHomologyData.homologyπ_comp_homologyIso_hom


@[reassoc (attr := simp, elementwise)]
lemma moduleCatCyclesIso_inv_π :
    S.moduleCatCyclesIso.inv ≫ S.homologyπ =
       S.moduleCatHomologyπ ≫ S.moduleCatHomologyIso.inv :=
  S.moduleCatLeftHomologyData.π_comp_homologyIso_inv


lemma exact_iff_surjective_moduleCatToCycles :
    S.Exact ↔ Function.Surjective S.moduleCatToCycles := by
  rw [S.moduleCatLeftHomologyData.exact_iff_epi_f', moduleCatLeftHomologyData_f',
    ModuleCat.epi_iff_surjective]
  /-
    R : Type u
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    ⊢ Iff (Function.Surjective ⇑(ModuleCat.ofHom S.moduleCatToCycles).hom) (Functi …
  -/
  rfl
  /-
    🎉 no goals
  -/


