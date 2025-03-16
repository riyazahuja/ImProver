/-- A sheaf of modules is a presheaf of modules such that the underlying presheaf
of abelian groups is a sheaf. -/
structure SheafOfModules where
  /-- the underlying presheaf of modules of a sheaf of modules -/
  val : PresheafOfModules.{v} R.val
  isSheaf : Presheaf.IsSheaf J val.presheaf


/-- A morphism between sheaves of modules is a morphism between the underlying
presheaves of modules. -/
@[ext]
structure Hom (X Y : SheafOfModules.{v} R) where
  /-- a morphism between the underlying presheaves of modules -/
  val : X.val ⟶ Y.val


instance : Category (SheafOfModules.{v} R) where
  Hom := Hom
  id _ := ⟨𝟙 _⟩
  comp f g := ⟨f.val ≫ g.val⟩


@[ext]
lemma hom_ext {X Y : SheafOfModules.{v} R} {f g : X ⟶ Y} (h : f.val = g.val) : f = g :=
  Hom.ext h


@[simp]
lemma id_val (X : SheafOfModules.{v} R) : Hom.val (𝟙 X) = 𝟙 X.val := rfl


@[simp, reassoc]
lemma comp_val {X Y Z : SheafOfModules.{v} R} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).val = f.val ≫ g.val := rfl


/-- The forgetful functor `SheafOfModules.{v} R ⥤ PresheafOfModules R.val`. -/
@[simps]
def forget : SheafOfModules.{v} R ⥤ PresheafOfModules R.val where
  obj F := F.val
  map φ := φ.val


/-- The forget functor `SheafOfModules R ⥤ PresheafOfModules R.val` is fully faithful. -/
@[simps]
def fullyFaithfulForget : (forget.{v} R).FullyFaithful where
  preimage φ := ⟨φ⟩


instance : (forget.{v} R).Faithful := (fullyFaithfulForget R).faithful


instance : (forget.{v} R).Full := (fullyFaithfulForget R).full


instance : (forget.{v} R).ReflectsIsomorphisms := (fullyFaithfulForget R).reflectsIsomorphisms


/-- Evaluation on an object `X` gives a functor
`SheafOfModules R ⥤ ModuleCat (R.val.obj X)`. -/
def evaluation (X : Cᵒᵖ) : SheafOfModules.{v} R ⥤ ModuleCat.{v} (R.val.obj X) :=
  forget _ ⋙ PresheafOfModules.evaluation _ X


/-- The forget functor `SheafOfModules R ⥤ Sheaf J AddCommGrp`. -/
@[simps]
def toSheaf : SheafOfModules.{v} R ⥤ Sheaf J AddCommGrp.{v} where
  obj M := ⟨_, M.isSheaf⟩
  map f := { val := (forget R ⋙ PresheafOfModules.toPresheaf R.val).map f }


/--
The forgetful functor from sheaves of modules over sheaf of ring `R` to sheaves of `R(X)`-module
when `X` is initial.
-/
@[simps]
noncomputable def forgetToSheafModuleCat
      (X : Cᵒᵖ) (hX : Limits.IsInitial X)  :
    SheafOfModules.{w} R ⥤ Sheaf J (ModuleCat.{w} (R.1.obj X)) where
  obj M := ⟨(PresheafOfModules.forgetToPresheafModuleCat X hX).obj M.1,
    Presheaf.isSheaf_of_isSheaf_comp _ _
      (forget₂ (ModuleCat.{w} (R.1.obj X)) AddCommGrp.{w}) M.isSheaf⟩
  map f := { val := (PresheafOfModules.forgetToPresheafModuleCat X hX).map f.1 }


/-- The canonical isomorphism between
`SheafOfModules.toSheaf R ⋙ sheafToPresheaf J AddCommGrp.{v}`
and `SheafOfModules.forget R ⋙ PresheafOfModules.toPresheaf R.val`. -/
def toSheafCompSheafToPresheafIso :
    toSheaf R ⋙ sheafToPresheaf J AddCommGrp.{v} ≅
      forget R ⋙ PresheafOfModules.toPresheaf R.val := Iso.refl _


instance : (toSheaf.{v} R).Faithful :=
  Functor.Faithful.of_comp_iso (toSheafCompSheafToPresheafIso.{v} R)


instance (M N : SheafOfModules.{v} R) : AddCommGroup (M ⟶ N) :=
  (fullyFaithfulForget R).homEquiv.addCommGroup


@[simp]
lemma add_val {M N : SheafOfModules.{v} R} (f g : M ⟶ N) :
    (f + g).val = f.val + g.val := rfl


instance : Preadditive (SheafOfModules.{v} R) where
                 /-
                   C : Type u₁
                   inst✝ : CategoryTheory.Category.{v₁, u₁} C
                   J : CategoryTheory.GrothendieckTopology C
                   R : CategoryTheory.Sheaf J RingCat
                   ⊢ ∀ (P Q R_1 : SheafOfModules R) (f f' : Quiver.Hom P Q) (g : Quiver.Hom Q R_1 …
                 -/
  add_comp := by intros; ext1; dsimp; simp only [Preadditive.add_comp]
                                      /-
                                        🎉 no goals
                                      -/
                 /-
                   C : Type u₁
                   inst✝ : CategoryTheory.Category.{v₁, u₁} C
                   J : CategoryTheory.GrothendieckTopology C
                   R : CategoryTheory.Sheaf J RingCat
                   ⊢ ∀ (P Q R_1 : SheafOfModules R) (f : Quiver.Hom P Q) (g g' : Quiver.Hom Q R_1 …
                 -/
  comp_add := by intros; ext1; dsimp; simp only [Preadditive.comp_add]
                                      /-
                                        🎉 no goals
                                      -/


instance : (forget R).Additive where


instance : (toSheaf R).Additive where


/-- The type of sections of a sheaf of modules. -/
abbrev sections (M : SheafOfModules.{v} R) : Type _ := M.val.sections


/-- The map `M.sections → N.sections` induced by a morphisms `M ⟶ N` of sheaves of modules. -/
abbrev sectionsMap {M N : SheafOfModules.{v} R} (f : M ⟶ N) (s : M.sections) : N.sections :=
  PresheafOfModules.sectionsMap f.val s


@[simp]
lemma sectionsMap_comp {M N P : SheafOfModules.{v} R} (f : M ⟶ N) (g : N ⟶ P) (s : M.sections) :
    sectionsMap (f ≫ g) s = sectionsMap g (sectionsMap f s) := rfl


@[simp]
lemma sectionsMap_id {M : SheafOfModules.{v} R} (s : M.sections) :
    sectionsMap (𝟙 M) s = s := rfl


variable (R) in
/-- The functor which sends a sheaf of modules to its type of sections. -/
@[simps]
def sectionsFunctor : SheafOfModules.{v} R ⥤ Type _ where
  obj := sections
  map f := sectionsMap f


variable (R) in
/-- The obvious free sheaf of modules of rank `1`. -/
@[simps]
def unit : SheafOfModules R where
  val := PresheafOfModules.unit R.val
  isSheaf := ((sheafCompose J (forget₂ RingCat.{u} AddCommGrp.{u})).obj R).cond


/-- The bijection `(unit R ⟶ M) ≃ M.sections` for `M : SheafOfModules R`. -/
def unitHomEquiv (M : SheafOfModules R) :
    (unit R ⟶ M) ≃ M.sections :=
  (fullyFaithfulForget R).homEquiv.trans M.val.unitHomEquiv


@[simp]
lemma unitHomEquiv_apply_coe (M : SheafOfModules R) (f : unit R ⟶ M) (X : Cᵒᵖ) :
    (M.unitHomEquiv f).val X = f.val.app X (1 : R.val.obj X) := rfl


lemma unitHomEquiv_comp_apply {M N : SheafOfModules.{u} R}
    (f : unit R ⟶ M) (p : M ⟶ N) :
    N.unitHomEquiv (f ≫ p) = sectionsMap p (M.unitHomEquiv f) := rfl


lemma unitHomEquiv_symm_comp {M N : SheafOfModules.{u} R} (s : M.sections) (p : M ⟶ N) :
    M.unitHomEquiv.symm s ≫ p = N.unitHomEquiv.symm (sectionsMap p s) :=
                               /-
                                 C : Type u₁
                                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                 J : CategoryTheory.GrothendieckTopology C
                                 R : CategoryTheory.Sheaf J RingCat
                                 inst✝ : J.HasSheafCompose (CategoryTheory.forget₂ RingCat AddCommGrp)
                                 M N : SheafOfModules R
                                 s : M.sections
                                 p : Quiver.Hom M N
                                 ⊢ Eq (N.unitHomEquiv (CategoryTheory.CategoryStruct.comp (M.unitHomEquiv.symm  …
                               -/
  N.unitHomEquiv.injective (by simp [unitHomEquiv_comp_apply])
                               /-
                                 🎉 no goals
                               -/


/-- A morphism of presheaves of modules is locally surjective
if the underlying morphism of presheaves of abelian groups is. -/
abbrev IsLocallySurjective : Prop :=
  Presheaf.IsLocallySurjective J ((PresheafOfModules.toPresheaf R).map f)


/-- A morphism of presheaves of modules is locally injective
if the underlying morphism of presheaves of abelian groups is. -/
abbrev IsLocallyInjective : Prop :=
  Presheaf.IsLocallyInjective J ((PresheafOfModules.toPresheaf R).map f)


/-- The bijection `(M₂ ⟶ N) ≃ (M₁ ⟶ N)` induced by a locally bijective morphism
`f : M₁ ⟶ M₂` of presheaves of modules, when `N` is a sheaf. -/
@[simps]
noncomputable def homEquivOfIsLocallyBijective : (M₂ ⟶ N) ≃ (M₁ ⟶ N) where
  toFun φ := f ≫ φ
  invFun ψ := homMk (((J.W_of_isLocallyBijective
      ((PresheafOfModules.toPresheaf R).map f)).homEquiv _ hN).symm
      ((PresheafOfModules.toPresheaf R).map ψ)) (by
        obtain ⟨φ, hφ⟩ := ((J.W_of_isLocallyBijective
          ((PresheafOfModules.toPresheaf R).map f)).homEquiv _ hN).surjective
          ((PresheafOfModules.toPresheaf R).map ψ)
        /-
          case intro
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          J : CategoryTheory.GrothendieckTopology C
          R✝ : CategoryTheory.Sheaf J RingCat
          R : CategoryTheory.Functor (Opposite C) RingCat
          M₁ M₂ : PresheafOfModules R
          f : Quiver.Hom M₁ M₂
          N : PresheafOfModules R
          hN : CategoryTheory.Presheaf.IsSheaf J N.presheaf
          inst✝² : J.WEqualsLocallyBijective AddCommGrp
          inst✝¹ : PresheafOfModules.IsLocallySurjective J f
          inst✝ : PresheafOfModules.IsLocallyInjective J f
          ψ : Quiver.Hom M₁ N
          φ : Quiver.Hom ((PresheafOfModules.toPresheaf R).obj M₂) N.presheaf
          hφ : Eq ((CategoryTheory.Localization.LeftBousfield.W.homEquiv ⋯ N.presheaf hN …
          ⊢ ∀ (X : Opposite C) (r : ↑(R.obj X)) (m : ↑(M₂.obj X)), Eq ((((CategoryTheory …
        -/
        simp only [← hφ, Equiv.symm_apply_apply]
        replace hφ : ∀ (Z : Cᵒᵖ) (x : M₁.obj Z), φ.app Z (f.app Z x) = ψ.app Z x :=
          fun Z x ↦ congr_fun ((forget _).congr_map (congr_app hφ Z)) x
        /-
          case intro
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          J : CategoryTheory.GrothendieckTopology C
          R✝ : CategoryTheory.Sheaf J RingCat
          R : CategoryTheory.Functor (Opposite C) RingCat
          M₁ M₂ : PresheafOfModules R
          f : Quiver.Hom M₁ M₂
          N : PresheafOfModules R
          hN : CategoryTheory.Presheaf.IsSheaf J N.presheaf
          inst✝² : J.WEqualsLocallyBijective AddCommGrp
          inst✝¹ : PresheafOfModules.IsLocallySurjective J f
          inst✝ : PresheafOfModules.IsLocallyInjective J f
          ψ : Quiver.Hom M₁ N
          φ : Quiver.Hom ((PresheafOfModules.toPresheaf R).obj M₂) N.presheaf
          hφ : ∀ (Z : Opposite C) (x : ↑(M₁.obj Z)), Eq ((φ.app Z) ((f.app Z).hom x)) (( …
          ⊢ ∀ (X : Opposite C) (r : ↑(R.obj X)) (m : ↑(M₂.obj X)), Eq ((φ.app X) (HSMul. …
        -/
        intro X r y
        apply hN.isSeparated _ _
          (Presheaf.imageSieve_mem J ((toPresheaf R).map f) y)
        /-
          case intro.a
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          J : CategoryTheory.GrothendieckTopology C
          R✝ : CategoryTheory.Sheaf J RingCat
          R : CategoryTheory.Functor (Opposite C) RingCat
          M₁ M₂ : PresheafOfModules R
          f : Quiver.Hom M₁ M₂
          N : PresheafOfModules R
          hN : CategoryTheory.Presheaf.IsSheaf J N.presheaf
          inst✝² : J.WEqualsLocallyBijective AddCommGrp
          inst✝¹ : PresheafOfModules.IsLocallySurjective J f
          inst✝ : PresheafOfModules.IsLocallyInjective J f
          ψ : Quiver.Hom M₁ N
          φ : Quiver.Hom ((PresheafOfModules.toPresheaf R).obj M₂) N.presheaf
          hφ : ∀ (Z : Opposite C) (x : ↑(M₁.obj Z)), Eq ((φ.app Z) ((f.app Z).hom x)) (( …
          X : Opposite C
          r : ↑(R.obj X)
          y : ↑(M₂.obj X)
          ⊢ ∀ (Y : C) (f_1 : Quiver.Hom Y (Opposite.unop X)), (CategoryTheory.Presheaf.i …
        -/
        rintro Y p ⟨x : M₁.obj _, hx : f.app _ x = M₂.map p.op y⟩
        have hφ' : ∀ (z : M₂.obj X), φ.app _ (M₂.map p.op z) =
            N.map p.op (φ.app _ z) := congr_fun ((forget _).congr_map (φ.naturality p.op))
        /-
          case intro.a.intro
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          J : CategoryTheory.GrothendieckTopology C
          R✝ : CategoryTheory.Sheaf J RingCat
          R : CategoryTheory.Functor (Opposite C) RingCat
          M₁ M₂ : PresheafOfModules R
          f : Quiver.Hom M₁ M₂
          N : PresheafOfModules R
          hN : CategoryTheory.Presheaf.IsSheaf J N.presheaf
          inst✝² : J.WEqualsLocallyBijective AddCommGrp
          inst✝¹ : PresheafOfModules.IsLocallySurjective J f
          inst✝ : PresheafOfModules.IsLocallyInjective J f
          ψ : Quiver.Hom M₁ N
          φ : Quiver.Hom ((PresheafOfModules.toPresheaf R).obj M₂) N.presheaf
          hφ : ∀ (Z : Opposite C) (x : ↑(M₁.obj Z)), Eq ((φ.app Z) ((f.app Z).hom x)) (( …
          X : Opposite C
          r : ↑(R.obj X)
          y : ↑(M₂.obj X)
          Y : C
          p : Quiver.Hom Y (Opposite.unop X)
          x : ↑(M₁.obj { unop := Y })
          hx : Eq ((f.app { unop := Y }).hom x) ((M₂.map p.op).hom y)
          hφ' : ∀ (z : ↑(M₂.obj X)), Eq ((φ.app { unop := Y }) ((M₂.map p.op).hom z)) (( …
          ⊢ Eq ((N.presheaf.map p.op) ((φ.app X) (HSMul.hSMul r y))) ((N.presheaf.map p. …
        -/
        change N.map p.op (φ.app X (r • y)) = N.map p.op (r • φ.app X y)
        rw [← hφ', M₂.map_smul, ← hx, ← (f.app _).hom.map_smul, hφ, (ψ.app _).hom.map_smul,
          ← hφ, hx, N.map_smul, hφ'])
  left_inv φ := (toPresheaf _).map_injective
    (((J.W_of_isLocallyBijective
      ((PresheafOfModules.toPresheaf R).map f)).homEquiv _ hN).left_inv
      ((PresheafOfModules.toPresheaf R).map φ))
  right_inv ψ := (toPresheaf _).map_injective
    (((J.W_of_isLocallyBijective
      ((PresheafOfModules.toPresheaf R).map f)).homEquiv _ hN).right_inv
      ((PresheafOfModules.toPresheaf R).map ψ))


