/-- Given a morphism of presheaves of commutative rings `φ : S ⟶ F.op ⋙ R`,
this is the type of relative `φ`-derivation of a presheaf of `R`-modules `M`. -/
@[ext]
structure Derivation where
  /-- the underlying additive map `R.obj X →+ M.obj X` of a derivation -/
  d {X : Dᵒᵖ} : R.obj X →+ M.obj X
  d_mul {X : Dᵒᵖ} (a b : R.obj X) : d (a * b) = a • d b + b • d a := by aesop_cat
  d_map {X Y : Dᵒᵖ} (f : X ⟶ Y) (x : R.obj X) :
    d (R.map f x) = M.map f (d x) := by aesop_cat
  d_app {X : Cᵒᵖ} (a : S.obj X) : d (φ.app X a) = 0 := by aesop_cat


lemma congr_d {d d' : M.Derivation φ} (h : d = d') {X : Dᵒᵖ} (b : R.obj X) :
                         /-
                           C : Type u₁
                           inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                           D : Type u₂
                           inst✝ : CategoryTheory.Category.{v₂, u₂} D
                           S : CategoryTheory.Functor (Opposite C) CommRingCat
                           F : CategoryTheory.Functor C D
                           R : CategoryTheory.Functor (Opposite D) CommRingCat
                           M : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
                           φ : Quiver.Hom S (F.op.comp R)
                           d d' : M.Derivation φ
                           h : Eq d d'
                           X : Opposite D
                           b : ↑(R.obj X)
                           ⊢ Eq (d.d b) (d'.d b)
                         -/
    d.d b = d'.d b := by rw [h]
                         /-
                           🎉 no goals
                         -/


@[simp] lemma d_one (X : Dᵒᵖ) : d.d (X := X) 1 = 0 := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    S : CategoryTheory.Functor (Opposite C) CommRingCat
    F : CategoryTheory.Functor C D
    R : CategoryTheory.Functor (Opposite D) CommRingCat
    M : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
    φ : Quiver.Hom S (F.op.comp R)
    d : M.Derivation φ
    X : Opposite D
    ⊢ Eq (d.d 1) 0
  -/
  simpa using d.d_mul (X := X) 1 1
  /-
    🎉 no goals
  -/


/-- The postcomposition of a derivation by a morphism of presheaves of modules. -/
@[simps! d_apply]
def postcomp (f : M ⟶ N) : N.Derivation φ where
  d := (f.app _).hom.toAddMonoidHom.comp d.d
                        /-
                          C : Type u₁
                          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                          D : Type u₂
                          inst✝ : CategoryTheory.Category.{v₂, u₂} D
                          S : CategoryTheory.Functor (Opposite C) CommRingCat
                          F : CategoryTheory.Functor C D
                          S' R : CategoryTheory.Functor (Opposite D) CommRingCat
                          M N : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
                          φ : Quiver.Hom S (F.op.comp R)
                          φ' : Quiver.Hom S' R
                          d : M.Derivation φ
                          f : Quiver.Hom M N
                          X Y : Opposite D
                          g : Quiver.Hom X Y
                          x : ↑(R.obj X)
                          ⊢ Eq ((fun {X} => (f.app X).hom.toAddMonoidHom.comp d.d) ((R.map g).hom x)) (( …
                        -/
  d_map {X Y} g x := by simpa using naturality_apply f g (d.d x)
                        /-
                          🎉 no goals
                        -/
  d_app {X} a := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      S : CategoryTheory.Functor (Opposite C) CommRingCat
      F : CategoryTheory.Functor C D
      S' R : CategoryTheory.Functor (Opposite D) CommRingCat
      M N : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
      φ : Quiver.Hom S (F.op.comp R)
      φ' : Quiver.Hom S' R
      d : M.Derivation φ
      f : Quiver.Hom M N
      X : Opposite C
      a : ↑(S.obj X)
      ⊢ Eq ((fun {X} => (f.app X).hom.toAddMonoidHom.comp d.d) ((φ.app X).hom a)) 0
    -/
    dsimp
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      S : CategoryTheory.Functor (Opposite C) CommRingCat
      F : CategoryTheory.Functor C D
      S' R : CategoryTheory.Functor (Opposite D) CommRingCat
      M N : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
      φ : Quiver.Hom S (F.op.comp R)
      φ' : Quiver.Hom S' R
      d : M.Derivation φ
      f : Quiver.Hom M N
      X : Opposite C
      a : ↑(S.obj X)
      ⊢ Eq ((f.app { unop := F.obj (Opposite.unop X) }).hom (d.d ((φ.app X).hom a))) 0
    -/
    erw [d_app, map_zero]
    /-
      🎉 no goals
    -/


/-- The universal property that a derivation `d : M.Derivation φ` must
satisfy so that the presheaf of modules `M` can be considered as the presheaf of
(relative) differentials of a presheaf of commutative rings `φ : S ⟶ F.op ⋙ R`. -/
structure Universal where
  /-- An absolyte derivation of `M'` descends as a morphism `M ⟶ M'`. -/
  desc {M' : PresheafOfModules (R ⋙ forget₂ CommRingCat RingCat)}
    (d' : M'.Derivation φ) : M ⟶ M'
  fac {M' : PresheafOfModules (R ⋙ forget₂ CommRingCat RingCat)}
    (d' : M'.Derivation φ) : d.postcomp (desc d') = d' := by aesop_cat
  postcomp_injective {M' : PresheafOfModules (R ⋙ forget₂ CommRingCat RingCat)}
    (φ φ' : M ⟶ M') (h : d.postcomp φ = d.postcomp φ') : φ = φ' := by aesop_cat


instance : Subsingleton d.Universal where
  allEq h₁ h₂ := by
    suffices ∀ {M' : PresheafOfModules (R ⋙ forget₂ CommRingCat RingCat)}
      (d' : M'.Derivation φ), h₁.desc d' = h₂.desc d' by
        cases h₁
        cases h₂
        simp only [Universal.mk.injEq]
        ext : 2
        apply this
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      S : CategoryTheory.Functor (Opposite C) CommRingCat
      F : CategoryTheory.Functor C D
      S' R : CategoryTheory.Functor (Opposite D) CommRingCat
      M N : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
      φ : Quiver.Hom S (F.op.comp R)
      φ' : Quiver.Hom S' R
      d : M.Derivation φ
      h₁ h₂ : d.Universal
      ⊢ ∀ {M' : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCa …
    -/
    intro M' d'
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      S : CategoryTheory.Functor (Opposite C) CommRingCat
      F : CategoryTheory.Functor C D
      S' R : CategoryTheory.Functor (Opposite D) CommRingCat
      M N : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
      φ : Quiver.Hom S (F.op.comp R)
      φ' : Quiver.Hom S' R
      d : M.Derivation φ
      h₁ h₂ : d.Universal
      M' : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
      d' : M'.Derivation φ
      ⊢ Eq (h₁.desc d') (h₂.desc d')
    -/
    apply h₁.postcomp_injective
    /-
      case h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      S : CategoryTheory.Functor (Opposite C) CommRingCat
      F : CategoryTheory.Functor C D
      S' R : CategoryTheory.Functor (Opposite D) CommRingCat
      M N : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
      φ : Quiver.Hom S (F.op.comp R)
      φ' : Quiver.Hom S' R
      d : M.Derivation φ
      h₁ h₂ : d.Universal
      M' : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
      d' : M'.Derivation φ
      ⊢ Eq (d.postcomp (h₁.desc d')) (d.postcomp (h₂.desc d'))
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The property that there exists a universal derivation for
a morphism of presheaves of commutative rings `S ⟶ F.op ⋙ R`. -/
class HasDifferentials : Prop where
  exists_universal_derivation : ∃ (M : PresheafOfModules.{u} (R ⋙ forget₂ _ _))
      (d : M.Derivation φ), Nonempty d.Universal


/-- Given a morphism of presheaves of commutative rings `φ : S ⟶ R`,
this is the type of relative `φ`-derivation of a presheaf of `R`-modules `M`. -/
abbrev Derivation' : Type _ := M.Derivation (F := 𝟭 D) φ'


@[simp]
nonrec lemma d_app (d : M.Derivation' φ') {X : Dᵒᵖ} (a : S'.obj X) :
    d.d (φ'.app X a) = 0 :=
  d.d_app _


/-- The derivation relative to the morphism of commutative rings `φ'.app X` induced by
a derivation relative to a morphism of presheaves of commutative rings. -/
noncomputable def app (d : M.Derivation' φ') (X : Dᵒᵖ) : (M.obj X).Derivation (φ'.app X) :=
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    S : CategoryTheory.Functor (Opposite C) CommRingCat
    F : CategoryTheory.Functor C D
    S' R : CategoryTheory.Functor (Opposite D) CommRingCat
    M N : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
    φ : Quiver.Hom S (F.op.comp R)
    φ' : Quiver.Hom S' R
    d : M.Derivation' φ'
    X : Opposite D
    ⊢ ∀ (b b' : ↑(R.obj X)), Eq ((fun b => d.d b) (HAdd.hAdd b b')) (HAdd.hAdd ((f …
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  ModuleCat.Derivation.mk (fun b ↦ d.d b)
  /-
    🎉 no goals
  -/


@[simp]
lemma app_apply (d : M.Derivation' φ') {X : Dᵒᵖ} (b : R.obj X) :
    (d.app X).d b = d.d b := rfl


/-- Given a morphism of presheaves of commutative rings `φ'`, this is the
in derivation `M.Derivation' φ'` that is given by a compatible family of derivations
with values in the modules `M.obj X` for all `X`. -/
def mk (d_map : ∀ ⦃X Y : Dᵒᵖ⦄ (f : X ⟶ Y) (x : R.obj X),
    (d Y).d ((R.map f) x) = (M.map f) ((d X).d x)) : M.Derivation' φ' where
                                        /-
                                          C : Type u₁
                                          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                          D : Type u₂
                                          inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                          S : CategoryTheory.Functor (Opposite C) CommRingCat
                                          F : CategoryTheory.Functor C D
                                          S' R : CategoryTheory.Functor (Opposite D) CommRingCat
                                          M N : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
                                          φ : Quiver.Hom S (F.op.comp R)
                                          φ' : Quiver.Hom S' R
                                          d : (X : Opposite D) → (M.obj X).Derivation (φ'.app X)
                                          d_map : ∀ ⦃X Y : Opposite D⦄ (f : Quiver.Hom X Y) (x : ↑(R.obj X)), Eq ((d Y). …
                                          X : Opposite D
                                          ⊢ ∀ (a b : ↑(R.obj X)), Eq ((d X).d (HAdd.hAdd a b)) (HAdd.hAdd ((d X).d a) (( …
                                        -/
  d {X} := AddMonoidHom.mk' (d X).d (by simp)
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
lemma mk_app (X : Dᵒᵖ) : (mk d d_map).app X = d X := rfl


/-- Constructor for `Derivation.Universal` in the case `F` is the identity functor. -/
def Universal.mk {d : M.Derivation' φ'}
    (desc : ∀ {M' : PresheafOfModules (R ⋙ forget₂ _ _)}
      (_ : M'.Derivation' φ'), M ⟶ M')
    (fac : ∀ {M' : PresheafOfModules (R ⋙ forget₂ _ _)}
      (d' : M'.Derivation' φ'), d.postcomp (desc d') = d')
    (postcomp_injective : ∀ {M' : PresheafOfModules (R ⋙ forget₂ _ _)}
      (α β : M ⟶ M'), d.postcomp α = d.postcomp β → α = β) : d.Universal where
  desc := desc
  fac := fac
  postcomp_injective := postcomp_injective


/-- The presheaf of relative differentials of a morphism of presheaves of
commutative rings. -/
@[simps (config := .lemmasOnly)]
noncomputable def relativeDifferentials' :
    PresheafOfModules.{u} (R ⋙ forget₂ _ _) where
  obj X := CommRingCat.KaehlerDifferential (φ'.app X)
  map f := CommRingCat.KaehlerDifferential.map (φ'.naturality f)
  -- Without `dsimp`, `ext` doesn't pick up the right lemmas.
                 /-
                   C : Type u₁
                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                   D : Type u₂
                   inst✝ : CategoryTheory.Category.{v₂, u₂} D
                   S : CategoryTheory.Functor (Opposite C) CommRingCat
                   F : CategoryTheory.Functor C D
                   S' R : CategoryTheory.Functor (Opposite D) CommRingCat
                   M N : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
                   φ : Quiver.Hom S (F.op.comp R)
                   φ' : Quiver.Hom S' R
                   x✝ : Opposite D
                   ⊢ Eq ((fun {X Y} f => CommRingCat.KaehlerDifferential.map ⋯) (CategoryTheory.C …
                 -/
  map_id _ := by dsimp; ext; simp
                             /-
                               🎉 no goals
                             -/
                     /-
                       C : Type u₁
                       inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                       D : Type u₂
                       inst✝ : CategoryTheory.Category.{v₂, u₂} D
                       S : CategoryTheory.Functor (Opposite C) CommRingCat
                       F : CategoryTheory.Functor C D
                       S' R : CategoryTheory.Functor (Opposite D) CommRingCat
                       M N : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
                       φ : Quiver.Hom S (F.op.comp R)
                       φ' : Quiver.Hom S' R
                       X✝ Y✝ Z✝ : Opposite D
                       x✝¹ : Quiver.Hom X✝ Y✝
                       x✝ : Quiver.Hom Y✝ Z✝
                       ⊢ Eq ((fun {X Y} f => CommRingCat.KaehlerDifferential.map ⋯) (CategoryTheory.C …
                     -/
  map_comp _ _ := by dsimp; ext; simp
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
lemma relativeDifferentials'_map_d {X Y : Dᵒᵖ} (f : X ⟶ Y) (x : R.obj X) :
    DFunLike.coe (α := CommRingCat.KaehlerDifferential (φ'.app X))
      (β := fun _ ↦ CommRingCat.KaehlerDifferential (φ'.app Y))
      ((relativeDifferentials' φ').map f).hom (CommRingCat.KaehlerDifferential.d x) =
        CommRingCat.KaehlerDifferential.d (R.map f x) :=
  CommRingCat.KaehlerDifferential.map_d (φ'.naturality f) _


/-- The universal derivation. -/
noncomputable def derivation' : (relativeDifferentials' φ').Derivation' φ' :=
  Derivation'.mk (fun X ↦ CommRingCat.KaehlerDifferential.D (φ'.app X))
    (fun _ _ f x ↦ (relativeDifferentials'_map_d φ' f x).symm)


/-- The derivation `Derivation' φ'` is universal. -/
noncomputable def isUniversal' : (derivation' φ').Universal :=
  Derivation'.Universal.mk
    (fun {M'} d' ↦
      { app := fun X ↦ (d'.app X).desc
        naturality := fun {X Y} f ↦ CommRingCat.KaehlerDifferential.ext (fun b ↦ by
          /-
            C : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} D
            S : CategoryTheory.Functor (Opposite C) CommRingCat
            F : CategoryTheory.Functor C D
            S' R : CategoryTheory.Functor (Opposite D) CommRingCat
            M N : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
            φ : Quiver.Hom S (F.op.comp R)
            φ' : Quiver.Hom S' R
            M' : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
            d' : M'.Derivation' φ'
            X Y : Opposite D
            f : Quiver.Hom X Y
            b : ↑(R.obj X)
            ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((PresheafOfModules.DifferentialsCon …
          -/
          dsimp
          /-
            C : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} D
            S : CategoryTheory.Functor (Opposite C) CommRingCat
            F : CategoryTheory.Functor C D
            S' R : CategoryTheory.Functor (Opposite D) CommRingCat
            M N : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
            φ : Quiver.Hom S (F.op.comp R)
            φ' : Quiver.Hom S' R
            M' : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
            d' : M'.Derivation' φ'
            X Y : Opposite D
            f : Quiver.Hom X Y
            b : ↑(R.obj X)
            ⊢ Eq ((d'.app Y).desc.hom (((PresheafOfModules.DifferentialsConstruction.relat …
          -/
          rw [ModuleCat.Derivation.desc_d, Derivation'.app_apply]
          /-
            C : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} D
            S : CategoryTheory.Functor (Opposite C) CommRingCat
            F : CategoryTheory.Functor C D
            S' R : CategoryTheory.Functor (Opposite D) CommRingCat
            M N : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
            φ : Quiver.Hom S (F.op.comp R)
            φ' : Quiver.Hom S' R
            M' : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
            d' : M'.Derivation' φ'
            X Y : Opposite D
            f : Quiver.Hom X Y
            b : ↑(R.obj X)
            ⊢ Eq ((d'.app Y).desc.hom (((PresheafOfModules.DifferentialsConstruction.relat …
          -/
          erw [relativeDifferentials'_map_d φ' f]
          /-
            C : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} D
            S : CategoryTheory.Functor (Opposite C) CommRingCat
            F : CategoryTheory.Functor C D
            S' R : CategoryTheory.Functor (Opposite D) CommRingCat
            M N : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
            φ : Quiver.Hom S (F.op.comp R)
            φ' : Quiver.Hom S' R
            M' : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
            d' : M'.Derivation' φ'
            X Y : Opposite D
            f : Quiver.Hom X Y
            b : ↑(R.obj X)
            ⊢ Eq ((d'.app Y).desc.hom (CommRingCat.KaehlerDifferential.d ((R.map f).hom b) …
          -/
          rw [ModuleCat.Derivation.desc_d]
          /-
            C : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} D
            S : CategoryTheory.Functor (Opposite C) CommRingCat
            F : CategoryTheory.Functor C D
            S' R : CategoryTheory.Functor (Opposite D) CommRingCat
            M N : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
            φ : Quiver.Hom S (F.op.comp R)
            φ' : Quiver.Hom S' R
            M' : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
            d' : M'.Derivation' φ'
            X Y : Opposite D
            f : Quiver.Hom X Y
            b : ↑(R.obj X)
            ⊢ Eq ((d'.app Y).d ((R.map f).hom b)) ((M'.map f).hom (d'.d b))
          -/
          dsimp
          /-
            C : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} D
            S : CategoryTheory.Functor (Opposite C) CommRingCat
            F : CategoryTheory.Functor C D
            S' R : CategoryTheory.Functor (Opposite D) CommRingCat
            M N : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
            φ : Quiver.Hom S (F.op.comp R)
            φ' : Quiver.Hom S' R
            M' : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
            d' : M'.Derivation' φ'
            X Y : Opposite D
            f : Quiver.Hom X Y
            b : ↑(R.obj X)
            ⊢ Eq (d'.d ((R.map f).hom b)) ((M'.map f).hom (d'.d b))
          -/
          rw [Derivation.d_map]
          /-
            C : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} D
            S : CategoryTheory.Functor (Opposite C) CommRingCat
            F : CategoryTheory.Functor C D
            S' R : CategoryTheory.Functor (Opposite D) CommRingCat
            M N : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
            φ : Quiver.Hom S (F.op.comp R)
            φ' : Quiver.Hom S' R
            M' : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
            d' : M'.Derivation' φ'
            X Y : Opposite D
            f : Quiver.Hom X Y
            b : ↑(R.obj X)
            ⊢ Eq ((M'.map f).hom (d'.d b)) ((M'.map f).hom (d'.d b))
          -/
          dsimp) })
          /-
            🎉 no goals
          -/
    (fun {M'} d' ↦ by
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        S : CategoryTheory.Functor (Opposite C) CommRingCat
        F : CategoryTheory.Functor C D
        S' R : CategoryTheory.Functor (Opposite D) CommRingCat
        M N : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
        φ : Quiver.Hom S (F.op.comp R)
        φ' : Quiver.Hom S' R
        M' : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
        d' : M'.Derivation' φ'
        ⊢ Eq (PresheafOfModules.Derivation.postcomp (PresheafOfModules.DifferentialsCo …
      -/
      ext X b
      /-
        case d.h.h
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        S : CategoryTheory.Functor (Opposite C) CommRingCat
        F : CategoryTheory.Functor C D
        S' R : CategoryTheory.Functor (Opposite D) CommRingCat
        M N : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
        φ : Quiver.Hom S (F.op.comp R)
        φ' : Quiver.Hom S' R
        M' : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
        d' : M'.Derivation' φ'
        X : Opposite D
        b : ↑(R.obj X)
        ⊢ Eq ((PresheafOfModules.Derivation.postcomp (PresheafOfModules.DifferentialsC …
      -/
      apply ModuleCat.Derivation.desc_d)
      /-
        🎉 no goals
      -/
    (fun {M} α β h ↦ by
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        S : CategoryTheory.Functor (Opposite C) CommRingCat
        F : CategoryTheory.Functor C D
        S' R : CategoryTheory.Functor (Opposite D) CommRingCat
        M✝ N : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
        φ : Quiver.Hom S (F.op.comp R)
        φ' : Quiver.Hom S' R
        M : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
        α β : Quiver.Hom (PresheafOfModules.DifferentialsConstruction.relativeDifferen …
        h : Eq (PresheafOfModules.Derivation.postcomp (PresheafOfModules.Differentials …
        ⊢ Eq α β
      -/
      ext1 X
      /-
        case h
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        S : CategoryTheory.Functor (Opposite C) CommRingCat
        F : CategoryTheory.Functor C D
        S' R : CategoryTheory.Functor (Opposite D) CommRingCat
        M✝ N : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
        φ : Quiver.Hom S (F.op.comp R)
        φ' : Quiver.Hom S' R
        M : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
        α β : Quiver.Hom (PresheafOfModules.DifferentialsConstruction.relativeDifferen …
        h : Eq (PresheafOfModules.Derivation.postcomp (PresheafOfModules.Differentials …
        X : Opposite D
        ⊢ Eq (α.app X) (β.app X)
      -/
      exact CommRingCat.KaehlerDifferential.ext (Derivation.congr_d h))
      /-
        🎉 no goals
      -/


instance : HasDifferentials (F := 𝟭 D) φ' := ⟨_, _,  ⟨isUniversal' φ'⟩⟩


